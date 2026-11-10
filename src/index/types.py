"""Shared retrieval types, so the lexical and vector sides speak one language."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hit:
    """One retrieved chunk with the score that retrieved it.

    `score` means different things at different stages (cosine, BM25, RRF,
    cross-encoder), so `stage` records which -- otherwise the relevance floor
    in M5 would be comparing incomparable numbers.
    """
    chunk_id: str
    score: float
    payload: dict[str, Any]
    stage: str = ""
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.payload.get("doc_id", "")

    @property
    def section(self) -> str:
        return self.payload.get("section", "")

    @property
    def text(self) -> str:
        return self.payload.get("text", "")

    @property
    def embed_text(self) -> str:
        return self.payload.get("embed_text", self.text)

    @property
    def status(self) -> str:
        return self.payload.get("status", "current")

    @property
    def context_line(self) -> str:
        return self.payload.get("context_line", "")
