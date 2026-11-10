"""Shared ingestion data types."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Section:
    """One header-delimited run of a document."""
    header_path: list[str]           # e.g. ["3. Leave", "Carry forward"]
    text: str
    is_table: bool = False

    @property
    def header(self) -> str:
        return " > ".join(self.header_path) if self.header_path else ""


@dataclass
class Document:
    doc_id: str                      # the filename, as the eval CSV cites it
    title: str
    sections: list[Section] = field(default_factory=list)
    version: str = ""
    effective_date: str = ""         # ISO yyyy-mm-dd, "" when the doc states none
    owner: str = ""
    status: str = "current"          # or "superseded"
    superseded_by: str = ""
    language: str = "en"             # "bn" for the Bangla document
    contains_injection: bool = False


@dataclass
class Chunk:
    """A retrievable unit. One chunk never splits a table (plan 4.4)."""
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str                        # the body, script-preserving
    context_line: str                # prepended provenance, embedded with the body
    is_table: bool = False
    version: str = ""
    effective_date: str = ""
    owner: str = ""
    status: str = "current"
    superseded_by: str = ""
    language: str = "en"
    contains_injection: bool = False

    @property
    def embed_text(self) -> str:
        """What actually goes to the embedder and the reranker.

        Plan 4.4: "Every table chunk is prefixed with its document and section
        so column meaning survives retrieval." We do it for every chunk, not
        just tables -- a bare '### Carry forward' body is meaningless alone.
        """
        return f"{self.context_line}\n\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["embed_text"] = self.embed_text
        return data
