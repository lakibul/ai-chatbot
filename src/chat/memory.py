"""Conversation memory: a bounded window of recent turns, persisted to disk.

Plan 5.2 mitigation 3: summarise history beyond 6 turns rather than appending
it. With prefill measured at ~20-24 tok/s the pressure to do that is real, so
the window is truncated and the oldest turns are dropped. The hook for
summarising is left in place.

Conversations are persisted so the sidebar survives a server restart. Without
that, reopening an old chat would show its transcript while the model had no
memory of it -- the follow-up rewriter would silently lose "it" and "some",
which is exactly the failure the rewriter exists to prevent.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from .. import config

TITLE_MAX_CHARS = 64

# time.time() has ~15 ms resolution on Windows, so two conversations touched in
# the same tick carry identical timestamps and the sidebar would order them
# arbitrarily. A monotonic counter breaks the tie by actual order of use.
_SEQ = itertools.count(1)


def _make_title(text: str) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= TITLE_MAX_CHARS:
        return cleaned or "New conversation"
    return cleaned[:TITLE_MAX_CHARS - 1].rstrip() + "…"


@dataclass
class Conversation:
    conversation_id: str
    turns: list[dict[str, str]] = field(default_factory=list)
    title: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    seq: int = field(default_factory=lambda: next(_SEQ))

    def add_user(self, text: str, ts: float | None = None) -> None:
        # `ts` is when the question was ASKED. The turn is only appended once
        # the answer is complete, so defaulting to now() would stamp a
        # 60-second answer as if the user had typed it when it finished.
        self.turns.append({"role": "user", "content": text,
                           "ts": float(ts) if ts is not None else time.time()})
        if not self.title:
            self.title = _make_title(text)
        self.updated_at = time.time()
        self.seq = next(_SEQ)

    def add_assistant(self, text: str, citations: list[dict] | None = None) -> None:
        turn: dict[str, Any] = {"role": "assistant", "content": text,
                                "ts": time.time()}
        if citations:
            # Kept beside the message, never inside `content`: this is for the
            # reader, so that a reopened conversation still shows its sources,
            # and it must not leak into the prompt. window() projects it away.
            turn["citations"] = citations
        self.turns.append(turn)
        self.updated_at = time.time()
        self.seq = next(_SEQ)

    def attach_trace(self, steps: list[dict] | None,
                     timings: dict[str, Any] | None) -> None:
        """Record how the last answer was produced, for the UI only.

        Stored beside the message exactly like `citations`, and projected
        away by window() for the same reason: it is a record for the reader,
        and must never be fed back to the model as if it were conversation.
        """
        if not self.turns or self.turns[-1].get("role") != "assistant":
            return
        if steps:
            self.turns[-1]["steps"] = steps
        if timings:
            self.turns[-1]["timings"] = timings

    def window(self, turns: int | None = None) -> list[dict[str, str]]:
        """The last N messages, oldest first.

        Always starts on a user message: a window beginning with an assistant
        reply reads as if the model spoke first, which confuses the rewriter.
        """
        turns = turns or config.HISTORY_TURNS
        recent = self.turns[-turns:]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]
        # Project to exactly what the chat API accepts. Anything stored for
        # the UI's benefit (citations) must not reach the model.
        return [{"role": t["role"], "content": t["content"]} for t in recent]

    def reset(self) -> None:
        self.turns.clear()
        self.title = ""
        self.updated_at = time.time()
        self.seq = next(_SEQ)

    # ------------------------------------------------------- serialisation --
    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": self.turns,
        }

    def summary(self) -> dict[str, Any]:
        """What the sidebar needs: no transcript, so the list stays small."""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title or "New conversation",
            "updated_at": self.updated_at,
            "messages": len(self.turns),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        return cls(
            conversation_id=data["conversation_id"],
            turns=list(data.get("turns") or []),
            title=data.get("title", ""),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        )


class ConversationStore:
    """In-process store with a JSON file behind it.

    Writes are whole-file and happen once per completed answer, which at one
    answer per ~60s is nothing. A partial write is avoided by writing to a
    temporary file and replacing, so a crash mid-save cannot leave the sidebar
    pointing at a truncated file.
    """

    def __init__(self, path: Path | None = None, autoload: bool = True) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._lock = RLock()
        self.path = path if path is not None else config.CONVERSATIONS_PATH
        if autoload:
            self.load()

    # ------------------------------------------------------------- access --
    def get(self, conversation_id: str) -> Conversation:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                conversation = Conversation(conversation_id)
                self._conversations[conversation_id] = conversation
            return conversation

    def find(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def reset(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            existed = self._conversations.pop(conversation_id, None) is not None
        if existed:
            self.save()
        return existed

    def summaries(self) -> list[dict[str, Any]]:
        """Most recently updated first. Empty conversations are not listed:
        opening the page creates one, and an unused id is not history."""
        with self._lock:
            live = [c for c in self._conversations.values() if c.turns]
        return [c.summary() for c in
                sorted(live, key=lambda c: (c.updated_at, c.seq), reverse=True)]

    def __len__(self) -> int:
        return len(self._conversations)

    # -------------------------------------------------------- persistence --
    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            payload = [c.to_dict() for c in self._conversations.values() if c.turns]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            # History is a convenience. Losing it must never break an answer.
            pass

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, list):
            return
        with self._lock:
            for item in raw:
                try:
                    conversation = Conversation.from_dict(item)
                except (KeyError, TypeError, ValueError):
                    continue
                self._conversations[conversation.conversation_id] = conversation
