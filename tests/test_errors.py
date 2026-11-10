"""M4.5 tests: error handling and resilience.

Plan M4.5 names these: test_errors.py (a mock Ollama that can refuse
connections, hang past timeout, and die mid-stream), test_preflight.py,
test_degraded_retrieval.py, and test_no_error_as_refusal.py -- the regression
test for the governing rule:

    A TECHNICAL FAILURE MUST NEVER BE REPORTED AS "I DON'T KNOW".

These use a stub retriever rather than the real one, so they neither need
Ollama nor contend for the Qdrant lock.
"""
from __future__ import annotations

import httpx
import pytest

from src import config
from src.chat import llm as llm_module
from src.chat.engine import ChatEngine
from src.chat.memory import Conversation, ConversationStore
from src.chat.prompts import NO_ANSWER_TEXT
from src.index.types import Hit
from src.retrieve.pipeline import RetrievalResult

# Phrases that mean "the documents do not answer this". A technical failure
# must never produce any of them.
REFUSAL_PHRASES = (
    "i don't know", "i do not know", "not in the documents",
    "could not find anything", "do not contain", "not mentioned",
)


def make_hit(doc_id: str = "employee_handbook_v3.md", text: str = "Annual leave: 18 days.",
             status: str = "current") -> Hit:
    return Hit(chunk_id=f"{doc_id}#000", score=0.9, stage="rerank",
               payload={"chunk_id": f"{doc_id}#000", "doc_id": doc_id,
                        "doc_title": "Employee Handbook", "section": "3. Leave",
                        "text": text, "embed_text": text, "status": status,
                        "effective_date": "2026-01-01"})


class StubRetriever:
    """Stands in for the real Retriever, with controllable behaviour."""

    def __init__(self, hits=None, degraded=None, ranking="rerank", raises=None):
        self._hits = [make_hit()] if hits is None else hits
        self._degraded = degraded or []
        self._ranking = ranking
        self._raises = raises

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        if self._raises:
            raise self._raises
        return RetrievalResult(hits=list(self._hits),
                               degraded=list(self._degraded),
                               stages={"ranking": self._ranking,
                                       "candidates": len(self._hits)})


def collect(engine: ChatEngine, question: str = "How many days of annual leave?",
            conversation: Conversation | None = None) -> dict:
    return engine.ask_sync(question, conversation=conversation)


# ------------------------------------------------- the governing rule ------
def _assert_no_refusal_text(answer: str) -> None:
    lowered = answer.lower()
    for phrase in REFUSAL_PHRASES:
        assert phrase not in lowered, (
            f"a technical failure produced the refusal phrase {phrase!r}; "
            f"this is exactly plan risk R11")


def test_no_error_as_refusal_connection_refused(monkeypatch):
    def dead(*_a, **_k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm_module.httpx, "stream", dead)
    engine = ChatEngine(StubRetriever())
    result = collect(engine)

    assert result["error"] is not None
    assert result["error"]["code"] == "llm_unavailable"
    _assert_no_refusal_text(result["answer"])
    assert result["answer"] == ""


def test_no_error_as_refusal_timeout(monkeypatch):
    def hang(*_a, **_k):
        raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(llm_module.httpx, "stream", hang)
    engine = ChatEngine(StubRetriever())
    result = collect(engine)

    assert result["error"]["code"] == "llm_timeout"
    _assert_no_refusal_text(result["answer"])


def test_no_error_as_refusal_retrieval_failure():
    engine = ChatEngine(StubRetriever(raises=RuntimeError("index corrupt")))
    result = collect(engine)
    assert result["error"]["code"] == "retrieval_failed"
    _assert_no_refusal_text(result["answer"])


def test_refusal_text_itself_is_not_an_error():
    """The control case: a genuine refusal is NOT an error, so the eval can
    tell "the documents do not cover this" from "the machine is broken"."""
    _assert_no_refusal_text("18 days of annual leave.")
    with pytest.raises(AssertionError):
        _assert_no_refusal_text(NO_ANSWER_TEXT)


# ------------------------------------------------------- retry policy ------
def test_connection_error_is_retried_exactly_once(monkeypatch):
    calls = {"n": 0}

    def flaky(*_a, **_k):
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm_module.httpx, "post", flaky)
    with pytest.raises(llm_module.LLMUnavailable):
        llm_module.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2, "one initial attempt plus exactly one retry"


def test_timeout_is_never_retried(monkeypatch):
    """At 60-90s per call a blind retry doubles an already-painful wait."""
    calls = {"n": 0}

    def slow(*_a, **_k):
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(llm_module.httpx, "post", slow)
    with pytest.raises(llm_module.LLMTimeout):
        llm_module.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


# ------------------------------------------------ the streaming trap -------
class FakeStream:
    """A stream that emits some tokens and then dies without a done frame."""

    def __init__(self, lines, fail_after=None):
        self._lines = lines
        self._fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for index, line in enumerate(self._lines):
            if self._fail_after is not None and index == self._fail_after:
                raise httpx.ReadError("connection dropped mid-stream")
            yield line


def test_stream_without_done_frame_is_a_failure(monkeypatch):
    """Plan R12: a truncated answer must never render as a complete one."""
    lines = ['{"message":{"content":"18 da"}}', '{"message":{"content":"ys"}}']

    monkeypatch.setattr(llm_module.httpx, "stream",
                        lambda *a, **k: FakeStream(lines))
    engine = ChatEngine(StubRetriever())
    result = collect(engine)

    assert result["error"] is not None
    assert result["error"]["code"] == "incomplete_stream"


def test_mid_stream_death_surfaces_as_an_error(monkeypatch):
    lines = ['{"message":{"content":"18"}}', '{"message":{"content":" days"}}',
             '{"done":true}']

    monkeypatch.setattr(llm_module.httpx, "stream",
                        lambda *a, **k: FakeStream(lines, fail_after=2))
    engine = ChatEngine(StubRetriever())
    result = collect(engine)

    assert result["error"] is not None
    _assert_no_refusal_text(result["answer"])


def test_complete_stream_succeeds(monkeypatch):
    lines = ['{"message":{"content":"18 days."}}',
             '{"done":true,"prompt_eval_count":100,"eval_count":5,'
             '"prompt_eval_duration":1000000000,"eval_duration":2000000000}']

    monkeypatch.setattr(llm_module.httpx, "stream",
                        lambda *a, **k: FakeStream(lines))
    engine = ChatEngine(StubRetriever())
    result = collect(engine)

    assert result["error"] is None
    assert result["answer"] == "18 days."
    assert result["citations"][0]["doc_id"] == "employee_handbook_v3.md"
    assert result["timings"]["prompt_tokens"] == 100


def test_ollama_error_field_in_stream_is_surfaced(monkeypatch):
    lines = ['{"error":"model not found"}']
    monkeypatch.setattr(llm_module.httpx, "stream",
                        lambda *a, **k: FakeStream(lines))
    engine = ChatEngine(StubRetriever())
    result = collect(engine)
    assert result["error"]["code"] == "llm_bad_response"


# --------------------------------------------------- degraded retrieval ----
def test_degraded_retrieval_still_returns_a_cited_answer(monkeypatch):
    """Plan M4.5: with the reranker down, a cited answer still arrives,
    flagged as degraded -- never a silent wrong answer, never a refusal."""
    lines = ['{"message":{"content":"18 days."}}', '{"done":true}']
    monkeypatch.setattr(llm_module.httpx, "stream",
                        lambda *a, **k: FakeStream(lines))

    engine = ChatEngine(StubRetriever(
        degraded=["reranker unavailable, using fusion order"], ranking="rrf"))
    result = collect(engine)

    assert result["error"] is None
    assert result["citations"], "a degraded answer must still be cited"
    assert any(s.get("stage") == "degraded" for s in result["stages"])


def test_zero_results_is_distinct_from_a_refusal(monkeypatch):
    """Retrieval finding nothing must not be worded like the out-of-scope
    refusal, or the eval cannot tell them apart."""
    engine = ChatEngine(StubRetriever(hits=[]))
    result = collect(engine)
    assert result["error"] is None
    assert result["answer"] != NO_ANSWER_TEXT
    assert "could not find anything in the Nodi Tech documents" in result["answer"]


# ------------------------------------------------------ input validation ---
def test_empty_question_is_rejected():
    engine = ChatEngine(StubRetriever())
    result = engine.ask_sync("   ")
    assert result["error"]["code"] == "empty_question"


def test_overlong_input_is_truncated(monkeypatch):
    lines = ['{"message":{"content":"ok"}}', '{"done":true}']
    monkeypatch.setattr(llm_module.httpx, "stream",
                        lambda *a, **k: FakeStream(lines))
    engine = ChatEngine(StubRetriever())
    result = engine.ask_sync("x" * (config.MAX_INPUT_CHARS + 500))
    assert any(s.get("stage") == "truncated" for s in result["stages"])
    assert result["error"] is None


# ------------------------------------------------------------- memory ------
def test_conversation_window_starts_on_a_user_turn():
    conversation = Conversation("c")
    for index in range(6):
        conversation.add_user(f"q{index}")
        conversation.add_assistant(f"a{index}")
    window = conversation.window(5)
    assert window[0]["role"] == "user"
    assert len(window) <= 5


def test_conversation_store_isolates_conversations():
    store = ConversationStore()
    store.get("a").add_user("one")
    assert store.get("b").turns == []
    store.reset("a")
    assert store.get("a").turns == []


def test_history_is_only_recorded_on_success(monkeypatch):
    """A failed turn must not poison the next turn's rewrite."""
    monkeypatch.setattr(llm_module.httpx, "stream",
                        lambda *a, **k: (_ for _ in ()).throw(
                            httpx.ConnectError("refused")))
    conversation = Conversation("c")
    engine = ChatEngine(StubRetriever())
    engine.ask_sync("will fail", conversation=conversation)
    assert conversation.turns == []
