"""The conversational fast path.

"hello" cost 27.6 seconds: 4.1s of retrieval and 21.3s of prefill, because
the pipeline fetched five policy chunks and the model read all 1,535 tokens
before replying. These tests pin both halves of the fix -- that pleasantries
short-circuit, and, more importantly, that real questions never do.
"""
from __future__ import annotations

import pytest

from src.chat.smalltalk import (ACK_REPLY, FAREWELL_REPLY, GREETING_REPLY,
                                THANKS_REPLY, smalltalk_reply)


@pytest.mark.parametrize("message", [
    "hi", "Hi", "HELLO", "hello!", "Hey", "hey there", "hi there",
    "Good morning", "good evening.", "Howdy", "yo", "hiya",
    "Assalamu Alaikum", "salam",
    "হ্যালো", "নমস্কার",
    "  hello  ",
])
def test_greetings_are_answered_instantly(message):
    assert smalltalk_reply(message) == GREETING_REPLY


@pytest.mark.parametrize("message", [
    "thanks", "Thanks!", "thank you", "Thank you very much", "thx", "cheers",
    "ধন্যবাদ",
])
def test_thanks_are_acknowledged(message):
    assert smalltalk_reply(message) == THANKS_REPLY


@pytest.mark.parametrize("message", ["bye", "Goodbye", "see you", "good night"])
def test_farewells(message):
    assert smalltalk_reply(message) == FAREWELL_REPLY


@pytest.mark.parametrize("message", ["ok", "okay", "got it", "perfect", "noted"])
def test_acknowledgements(message):
    assert smalltalk_reply(message) == ACK_REPLY


def test_paired_pleasantries():
    assert smalltalk_reply("hi thanks") == GREETING_REPLY
    assert smalltalk_reply("hello hi") == GREETING_REPLY


# ------------------------------------------------------- the important half --
@pytest.mark.parametrize("message", [
    # A greeting ATTACHED to a question is a question.
    "hello, how many days of annual leave do I get?",
    "hi, what does ND-1042 mean?",
    "thanks, but what is the hotel limit?",
    "good morning, can I work remotely?",
    # Short real questions that could look like noise.
    "sick leave?",
    "ND-1042",
    "SSO?",
    "annual leave",
    "hotel limit",
    "password length",
    "অফিস বাস কখন ছাড়ে?",
    # Words that merely CONTAIN a pleasantry.
    "high availability",
    "thanksgiving holiday policy",
    "okta integration",
    "who approves my claim",
])
def test_real_questions_are_never_short_circuited(message):
    assert smalltalk_reply(message) is None, message


def test_empty_and_whitespace_are_not_smalltalk():
    """Empty input is rejected earlier, with its own error."""
    assert smalltalk_reply("") is None
    assert smalltalk_reply("   ") is None


def test_long_messages_are_never_smalltalk():
    assert smalltalk_reply("hello " * 20) is None


def test_emoji_only_greeting_still_matches():
    assert smalltalk_reply("hello \U0001f44b") == GREETING_REPLY


# ------------------------------------------------------------ end to end -----
def test_engine_answers_a_greeting_without_retrieval_or_llm():
    """The whole point: no Ollama call, no Qdrant call, no prefill."""
    from src.chat.engine import ChatEngine
    from tests.test_errors import StubRetriever

    engine = ChatEngine(StubRetriever(
        raises=AssertionError("retrieval must not run for a greeting")))
    result = engine.ask_sync("hello")

    assert result["error"] is None
    assert result["answer"] == GREETING_REPLY
    assert result["citations"] == []


def test_engine_records_the_greeting_in_history():
    """It is still a turn: the next question's rewrite sees a sane history."""
    from src.chat.engine import ChatEngine
    from src.chat.memory import Conversation
    from tests.test_errors import StubRetriever

    conversation = Conversation("c")
    ChatEngine(StubRetriever()).ask_sync("hi", conversation=conversation)
    assert [t["role"] for t in conversation.turns] == ["user", "assistant"]
    assert conversation.title == "hi"


def test_engine_still_runs_the_pipeline_for_a_greeting_plus_question(monkeypatch):
    from src.chat import llm as llm_module
    from src.chat.engine import ChatEngine
    from tests.test_errors import FakeStream, StubRetriever

    lines = ['{"message":{"content":"18 days."}}', '{"done":true}']
    monkeypatch.setattr(llm_module.httpx, "stream",
                        lambda *a, **k: FakeStream(lines))

    result = ChatEngine(StubRetriever()).ask_sync(
        "hello, how many days of annual leave do I get?")
    assert result["answer"] == "18 days."
    assert result["citations"], "a real question must still be grounded"


# ----------------------------------- phrases that LOOK like questions --------
@pytest.mark.parametrize("message", [
    "how are you", "How are you?", "how are you doing?", "how's it going",
    "hows it going", "what's up", "sup", "are you ok", "how do you do",
    "কেমন আছেন?",
])
def test_wellbeing_is_answered_not_refused(message):
    """Regression: "how are you?" was answered with the out-of-scope refusal.

    The fast path rejected it (it contains "how", "are" and a question mark),
    retrieval then scored below the relevance floor, and the floor cannot
    tell a pleasantry from a question the documents genuinely do not cover.
    """
    from src.chat.smalltalk import WELLBEING_REPLY
    assert smalltalk_reply(message) == WELLBEING_REPLY


@pytest.mark.parametrize("message", [
    "who are you", "Who are you?", "what are you", "what can you do?",
    "what do you do", "how can you help", "help", "what is this",
    "what can I ask you?",
])
def test_capability_questions_get_a_real_answer(message):
    from src.chat.smalltalk import CAPABILITY_REPLY
    assert smalltalk_reply(message) == CAPABILITY_REPLY


def test_capability_reply_describes_the_actual_corpus():
    from src.chat.smalltalk import CAPABILITY_REPLY
    lowered = CAPABILITY_REPLY.lower()
    for topic in ("leave", "security", "expense", "nodidesk", "api"):
        assert topic in lowered, topic
    # It must not promise anything beyond the documents.
    assert "cite" in lowered or "document" in lowered


def test_greeting_plus_pleasantry_combinations():
    from src.chat.smalltalk import CAPABILITY_REPLY, WELLBEING_REPLY
    assert smalltalk_reply("hi, how are you?") == WELLBEING_REPLY
    assert smalltalk_reply("Hello! How are you?") == WELLBEING_REPLY
    assert smalltalk_reply("hey, what can you do?") == CAPABILITY_REPLY


@pytest.mark.parametrize("message", [
    # The dangerous near-misses: a pleasantry PREFIX on a real question.
    "how are you going to handle my claim?",
    "what can you do about my expired claim?",
    "who are the approvers for a BDT 30,000 claim?",
    "what are you supposed to do when a ticket is P1?",
    "how are leave days carried forward?",
    "how are passwords stored?",
    "what are the office hours?",
    "help me find the API rate limit",
    "what do you do when a webhook fails?",
])
def test_near_misses_still_run_the_full_pipeline(message):
    """Exact-set matching is what keeps these safe. A prefix match here would
    silently refuse real questions."""
    assert smalltalk_reply(message) is None, message
