"""The LangChain backend: same contract as the native client, nothing leaks.

None of these tests call Ollama. They pin the properties that, if broken,
would silently change what the model receives or send data off the machine.
"""
import os
import subprocess
import sys
from pathlib import Path

import httpx
import ollama
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src import config
from src.chat import backend, lc, llm
from src.chat.llm import LLMBadResponse, LLMTimeout, LLMUnavailable

ROOT = Path(__file__).resolve().parent.parent


# --- the switch ---------------------------------------------------------------

def test_every_test_starts_native_whatever_the_dotenv_says():
    """conftest.py pins this. Without it, USE_LANGCHAIN=1 in a developer's .env made 13
    unrelated tests (which patch the native client) fail on that machine only."""
    assert config.USE_LANGCHAIN is False


def test_default_is_native(monkeypatch):
    """Flag off = the code that produced the audited 40/40."""
    monkeypatch.setattr(config, "USE_LANGCHAIN", False)
    assert backend.name() == "native"


def test_flag_selects_the_backend_at_call_time(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "chat", lambda *a, **k: calls.append("native") or "n")
    monkeypatch.setattr(lc, "chat", lambda *a, **k: calls.append("langchain") or "l")
    monkeypatch.setattr(config, "USE_LANGCHAIN", False)
    assert backend.chat([]) == "n"
    monkeypatch.setattr(config, "USE_LANGCHAIN", True)
    assert backend.chat([]) == "l" and backend.name() == "langchain"
    assert calls == ["native", "langchain"]


def test_engine_and_rewriter_use_the_switch_not_llm_directly():
    for rel in ("src/chat/engine.py", "src/chat/rewrite.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "from .backend import" in text, rel
        assert "from .llm import LLMError, chat" not in text, rel


# --- the request that reaches Ollama ------------------------------------------

def test_every_ollama_option_that_matters_is_carried_over():
    """num_gpu=0 stops the unusable 2 GB GPU being tried; reasoning=False is
    Ollama's think:false (qwen3:1.7b returns nothing without it)."""
    m = lc._model("some-model", 30.0, tuple(sorted(config.ollama_options(40).items())))
    assert m.num_gpu == 0 == config.NUM_GPU
    assert m.num_ctx == config.NUM_CTX
    assert m.num_thread == config.NUM_THREAD
    assert m.temperature == config.TEMPERATURE
    assert m.num_predict == 40
    assert m.reasoning is False
    assert m.keep_alive == config.KEEP_ALIVE
    assert m.base_url == config.OLLAMA_HOST


def test_the_options_really_reach_the_wire(monkeypatch):
    """Not just stored on the object: captured at the client call."""
    seen = {}

    def fake_chat(self, *args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(ollama.Client, "chat", fake_chat)
    with pytest.raises(RuntimeError):
        lc.chat([{"role": "user", "content": "hi"}], model="wire-test", num_predict=7)
    assert seen["options"]["num_gpu"] == 0
    assert seen["options"]["num_predict"] == 7
    assert seen["think"] is False


def test_unknown_options_are_refused_not_dropped():
    with pytest.raises(ValueError, match="unsupported"):
        lc._chain("m", 10, None, {"top_p_typo": 1})


def test_supported_option_override_changes_the_model():
    a = lc._chain("m", 10, None)
    b = lc._chain("m", 10, None, {"temperature": 0.0})
    assert a.last.temperature == config.TEMPERATURE
    assert b.last.temperature == 0.0


# --- the prompt passes through untouched --------------------------------------

def test_messages_keep_roles_and_order():
    out = lc.to_messages([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ])
    assert [type(m) for m in out] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert [m.content for m in out] == ["S", "U1", "A1", "U2"]


def test_unknown_role_is_an_error():
    with pytest.raises(LLMBadResponse):
        lc.to_messages([{"role": "tool", "content": "x"}])


def test_braces_and_tags_in_documents_are_not_treated_as_a_template():
    """Retrieved text holds { } and <document>. The template must not parse it
    -- that would either crash or silently rewrite what the model reads."""
    nasty = '<document id="a.md">{not_a_variable} {{x}} %s</document>'
    rendered = lc._PROMPT.invoke({"messages": [HumanMessage(content=nasty)]}).to_messages()
    assert rendered[0].content == nasty


# --- errors keep the project's types ------------------------------------------

def test_exceptions_map_to_the_projects_own():
    assert isinstance(lc._translate(httpx.ReadTimeout("t"), "m", 5), LLMTimeout)
    assert isinstance(lc._translate(ConnectionError("refused"), "m", 5), LLMUnavailable)
    assert isinstance(lc._translate(httpx.ConnectError("x"), "m", 5), LLMUnavailable)
    resp = ollama.ResponseError("model not found", 404)
    assert isinstance(lc._translate(resp, "m", 5), LLMBadResponse)
    odd = ValueError("a real bug")
    assert lc._translate(odd, "m", 5) is odd          # never disguised


# --- streaming rules ----------------------------------------------------------

class _Chunk:
    def __init__(self, content="", meta=None):
        self.content, self.response_metadata = content, meta or {}


class _FakeChain:
    def __init__(self, chunks, closed):
        self._chunks, self._closed = chunks, closed

    def stream(self, _payload):
        try:
            yield from self._chunks
        finally:
            self._closed.append(True)


def _stream_with(monkeypatch, chunks):
    closed = []
    monkeypatch.setattr(lc, "_chain", lambda *a, **k: _FakeChain(chunks, closed))
    return closed


def test_stream_yields_tokens_then_one_done_with_metrics(monkeypatch):
    _stream_with(monkeypatch, [
        _Chunk("Hel"), _Chunk("lo"),
        _Chunk("", {"done": True, "prompt_eval_count": 12, "eval_count": 3,
                    "prompt_eval_duration": 2_000_000_000, "eval_duration": 1_000_000_000}),
    ])
    frames = list(lc.chat_stream([{"role": "user", "content": "x"}], model="m"))
    assert [f[0] for f in frames] == ["token", "token", "done"]
    assert "".join(f[1] for f in frames if f[0] == "token") == "Hello"
    done = frames[-1][2]
    assert done["prompt_tokens"] == 12 and done["output_tokens"] == 3
    assert done["prefill_seconds"] == 2.0 and done["generation_seconds"] == 1.0


def test_a_stream_without_done_yields_no_done(monkeypatch):
    """A truncated stream must never be dressed up as a finished answer
    (CLAUDE.md section 7). The engine treats 'no done' as a failure."""
    _stream_with(monkeypatch, [_Chunk("part"), _Chunk("ial")])
    frames = list(lc.chat_stream([{"role": "user", "content": "x"}], model="m"))
    assert [f[0] for f in frames] == ["token", "token"]


def test_closing_the_stream_closes_the_upstream_one(monkeypatch):
    """Browser disconnects -> engine closes this generator -> the HTTP request
    underneath must be released, or an orphaned generation blocks the next
    question for a minute (plan R13)."""
    closed = _stream_with(monkeypatch, [_Chunk(str(i)) for i in range(50)])
    gen = lc.chat_stream([{"role": "user", "content": "x"}], model="m")
    assert next(gen)[0] == "token"
    gen.close()
    assert closed == [True]


# --- nothing leaves the machine -----------------------------------------------

def test_cloud_tracing_is_forced_off_even_if_the_shell_turns_it_on():
    """LangSmith would send every question and passage to a third party."""
    env = {**os.environ, "LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true"}
    code = ("import src.chat.lc, os; "
            "from langsmith.utils import tracing_is_enabled; "
            "print(os.environ['LANGSMITH_TRACING'], os.environ['LANGCHAIN_TRACING_V2'], tracing_is_enabled())")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.stdout.split()[-3:] == ["false", "false", "False"], out.stdout + out.stderr


def test_dependencies_are_pinned():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "langchain-core==" in text and "langchain-ollama==" in text
