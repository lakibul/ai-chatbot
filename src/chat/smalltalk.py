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

WELLBEING_REPLY = (
    "I'm working fine, thank you. Ask me anything about Nodi Tech's policies "
    "or NodiDesk and I'll find it in the documents.")

CAPABILITY_REPLY = (
    "I answer questions from Nodi Tech's internal documents and cite where "
    "each answer came from. I cover HR policy (leave, probation, remote work, "
    "notice periods), IT security, expense reimbursement, office transport and "
    "canteen, and the NodiDesk product — plans and pricing, the user guide, "
    "the REST API, error codes and release notes.\n\n"
    "I only use those documents. If something isn't in them I'll say so rather "
    "than guess. You can ask in English or Bangla.")

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

# Phrases that LOOK like questions but are conversational. These are matched
# in full, before the substantive-content filter below, because that filter
# rejects anything containing "how", "are", "what" or a question mark -- which
# is every one of these.
#
# This is why "how are you?" used to be answered with the out-of-scope
# refusal: the fast path rejected it, retrieval found nothing above the
# relevance floor, and the floor cannot tell a pleasantry from a question
# about something the documents genuinely do not cover.
_WELLBEING = {
    "how are you", "how are you doing", "how are you today", "how r u",
    "how do you do", "hows it going", "how is it going", "how's it going",
    "you ok", "you good", "are you ok", "are you there", "whats up",
    "what's up", "sup", "কেমন আছেন", "কেমন আছো",
}

_CAPABILITY = {
    "who are you", "what are you", "what can you do", "what do you do",
    "what can i ask", "what can i ask you", "what can you help with",
    "how can you help", "how can you help me", "what is this", "help",
    "what do you know", "what documents do you have", "how do you work",
    "আপনি কে", "আপনি কি করতে পারেন",
}

# Anything that makes a message a real request, however short, disqualifies it.
# Question marks are NOT a signal here: every conversational phrase above
# carries one, and the match is exact-set anyway, so a real question can only
# short-circuit if it is literally one of those phrases.
_SUBSTANTIVE = re.compile(r"\b(what|how|when|where|who|why|which|"
                          r"can|does|do|is|are|should|tell|show|list|find|give)\b", re.I)

# Apostrophes are DELETED rather than turned into a space: replacing them
# splits "how's it going" into "how s it going", which matches nothing.
_APOSTROPHE = re.compile(r"['‘’ʼ`]")
_STRIP = re.compile(r"[!?.,…~\-–—*\"()\[\]:;؟৷।]+")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").strip().lower()
    # Drop emoji and symbols; a waving hand is still just a greeting.
    text = "".join(ch for ch in text
                   if not unicodedata.category(ch).startswith("S"))
    text = _APOSTROPHE.sub("", text)
    text = _STRIP.sub(" ", text)
    return " ".join(text.split())


def smalltalk_reply(message: str) -> str | None:
    """A canned reply when the whole message is a pleasantry, else None."""
    cleaned = _normalise(message)
    if not cleaned or len(cleaned) > 40:
        return None

    # Checked FIRST, on the whole message, because every one of these would be
    # thrown out by the substantive-content filter below.
    if cleaned in _WELLBEING:
        return WELLBEING_REPLY
    if cleaned in _CAPABILITY:
        return CAPABILITY_REPLY

    paired = _paired(cleaned)
    if paired is not None:
        return paired

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

    return None


def _paired(cleaned: str) -> str | None:
    """"hi how are you" is a greeting followed by a pleasantry, not a question."""
    words = cleaned.split()
    if not 2 <= len(words) <= 7:
        return None
    for first in range(1, len(words)):
        left, right = " ".join(words[:first]), " ".join(words[first:])
        if left not in _GREETINGS:
            continue
        if right in _WELLBEING:
            return WELLBEING_REPLY
        if right in _CAPABILITY:
            return CAPABILITY_REPLY
        if right in (_THANKS | _GREETINGS | _ACKS):
            return GREETING_REPLY
    return None
