"""History-aware query rewriting.

Plan 3 and M4: C1 turn 2 ("Can I carry some forward?") must become a
standalone query, and C2's "Does it have SSO?" must resolve "it" to the
Growth plan. Done with the 1.7b model and a <=40 token budget, because this
runs before every retrieval and must not add seconds to the answer.

Plan M4.5: if the rewrite fails, use the raw question unchanged -- "No --
silent, safe". A failed rewrite degrades retrieval slightly; surfacing it
would be noise.
"""
from __future__ import annotations

import re

from .. import config
from .llm import LLMError, chat
from .prompts import build_rewrite_messages

# The small model sometimes decorates its answer despite being told not to.
_PREFIXES = re.compile(
    r"^\s*(standalone\s+(search\s+)?query|query|rewritten(\s+query)?|answer)\s*[:\-]\s*",
    re.I)


def _clean(text: str, fallback: str) -> str:
    candidate = text.strip()
    if not candidate:
        return fallback

    # Take the first non-empty line: the model occasionally adds commentary.
    for line in candidate.split("\n"):
        if line.strip():
            candidate = line.strip()
            break

    candidate = _PREFIXES.sub("", candidate).strip()
    candidate = candidate.strip('"“”\'')

    # A rewrite that collapsed to nothing, or that ballooned, is not trusted.
    if not candidate or len(candidate) > 300:
        return fallback
    return candidate


def rewrite_query(question: str, history: list[dict[str, str]] | None = None,
                  model: str | None = None) -> tuple[str, bool]:
    """Return (query, was_rewritten).

    With no history there is nothing to resolve, so the question is returned
    untouched and no LLM call is made at all.
    """
    if not history:
        return question, False

    messages = build_rewrite_messages(question, history)
    try:
        result = chat(messages,
                      model=model or config.REWRITE_MODEL,
                      num_predict=config.REWRITE_MAX_TOKENS,
                      timeout=config.REWRITE_TIMEOUT)
    except LLMError:
        return question, False          # silent, safe (plan M4.5)

    rewritten = _clean(result.text, question)
    return rewritten, rewritten != question
