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
