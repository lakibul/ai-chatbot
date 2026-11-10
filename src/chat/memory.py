"""Conversation memory: a bounded window of recent turns.

Plan 5.2 mitigation 3: summarise history beyond 6 turns rather than appending
it. With prefill measured at ~4,100 tok/s the pressure to do that is far
lower than the plan assumed, so the window is simply truncated and the
oldest turns are dropped. The hook for summarising is left in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from .. import config


@dataclass
class Conversation:
    conversation_id: str
    turns: list[dict[str, str]] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.turns.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.turns.append({"role": "assistant", "content": text})

    def window(self, turns: int | None = None) -> list[dict[str, str]]:
        """The last N messages, oldest first.

        Always starts on a user message: a window beginning with an assistant
        reply reads as if the model spoke first, which confuses the rewriter.
        """
        turns = turns or config.HISTORY_TURNS
        recent = self.turns[-turns:]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]
        return recent

    def reset(self) -> None:
        self.turns.clear()


class ConversationStore:
    """In-process conversation store, safe for the single-slot server."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._lock = Lock()

    def get(self, conversation_id: str) -> Conversation:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                conversation = Conversation(conversation_id)
                self._conversations[conversation_id] = conversation
            return conversation

    def reset(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)

    def __len__(self) -> int:
        return len(self._conversations)
