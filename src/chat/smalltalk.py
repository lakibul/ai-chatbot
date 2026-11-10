"""Conversational openers answered without retrieval or the LLM.

Measured: "hello" cost 27.6 seconds. Retrieval returned five policy chunks,
the prompt came to 1,535 tokens, and the model spent 21.3 s of prefill
READING EXPENSE POLICY before it could say hello back.

The relevance floor does not prevent this. The floor is calibrated on
questions, and a cross-encoder scores a greeting middlingly against almost
any text, so "hello" clears it comfortably. Nothing downstream can fix that
either: by the time a chunk has been retrieved, the prompt has been built.

So greetings are answered here, deterministically, before any of it runs.

The matcher is deliberately strict: it fires only when the ENTIRE message is
a pleasantry. "hello, how many days of leave do I get?" is a real question
and must go through the normal pipeline, so a prefix match would be a bug.
"""
from __future__ import annotations

import re
import unicodedata

GREETING_REPLY = (
    "Hello. I answer questions about Nodi Tech's internal documents — HR "
    "policy, IT security, expenses, and the NodiDesk product and API. "
    "What would you like to know?")

THANKS_REPLY = "You're welcome. Ask me anything else about Nodi Tech."

FAREWELL_REPLY = "Goodbye. I'm here whenever you need the documents checked."

ACK_REPLY = "Noted. What else can I look up for you?"

_GREETINGS = {
    "hi", "hii", "hiii", "hello", "helo", "hallo", "hey", "heya", "hiya", "yo",
    "hi there", "hello there", "hey there",
    "good morning", "good afternoon", "good evening", "good day",
    "greetings", "howdy",
    "assalamu alaikum", "assalamualaikum", "salam", "salaam",
    "as salamu alaikum", "walaikum assalam",
    # Bangla
    "হ্যালো", "হায়", "নমস্কার", "আসসালামু আলাইকুম",
}

_THANKS = {
    "thanks", "thank you", "thank u", "thanks a lot", "thanks very much",
    "thank you very much", "many thanks", "thx", "ty", "cheers", "appreciated",
    "much appreciated", "thanks so much",
    "ধন্যবাদ", "ধন্যবাদ।", "অসংখ্য ধন্যবাদ",
}

_FAREWELLS = {
    "bye", "byebye", "bye bye", "goodbye", "good bye", "see you", "see ya",
    "cya", "later", "good night", "goodnight",
    "বিদায়", "আল্লাহ হাফেজ",
}

_ACKS = {
    "ok", "okay", "k", "kk", "got it", "understood", "alright", "all right",
    "sure", "fine", "great", "nice", "cool", "perfect", "noted",
    "আচ্ছা", "ঠিক আছে",
}

# Anything that makes a message a real request, however short, disqualifies it.
_SUBSTANTIVE = re.compile(r"[?؟৷।]|\b(what|how|when|where|who|why|which|"
                          r"can|does|do|is|are|should|tell|show|list|find|give)\b", re.I)

_STRIP = re.compile(r"[!.,…~\-–—*\"'()\[\]:;]+")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").strip().lower()
    # Drop emoji and symbols; a waving hand is still just a greeting.
    text = "".join(ch for ch in text
                   if not unicodedata.category(ch).startswith("S"))
    text = _STRIP.sub(" ", text)
    return " ".join(text.split())


def smalltalk_reply(message: str) -> str | None:
    """A canned reply when the whole message is a pleasantry, else None."""
    cleaned = _normalise(message)
    if not cleaned or len(cleaned) > 40:
        return None
    if _SUBSTANTIVE.search(cleaned):
        return None

    if cleaned in _GREETINGS:
        return GREETING_REPLY
    if cleaned in _THANKS:
        return THANKS_REPLY
    if cleaned in _FAREWELLS:
        return FAREWELL_REPLY
    if cleaned in _ACKS:
        return ACK_REPLY

    # "hi thanks" and the like: a pair of pleasantries is still a pleasantry.
    words = cleaned.split()
    if 2 <= len(words) <= 4:
        for first in range(1, len(words)):
            left, right = " ".join(words[:first]), " ".join(words[first:])
            if left in _GREETINGS and right in (_THANKS | _GREETINGS | _ACKS):
                return GREETING_REPLY
    return None
