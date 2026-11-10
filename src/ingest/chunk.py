"""Chunking.

Two rules carry almost all the weight here.

Plan 4.4 -- TABLES ARE NEVER SPLIT. A chunk either contains a whole table,
header row included, or none of it. Splitting the pricing matrix mid-way makes
"BDT 2,500" unattributable to "Growth".

Plan 3 / 4.1 -- EVERY CHUNK CARRIES A CONTEXT LINE, embedded with the body.
For a superseded document that line is also where the warning lives, so the
"this is outdated" signal survives into the vector space itself rather than
sitting only in a payload field the embedder never sees.
"""
from __future__ import annotations

from .. import config
from .models import Chunk, Document, Section


def build_context_line(doc: Document, section: Section) -> str:
    """The provenance prefix prepended to every chunk before embedding."""
    name = doc.title
    if doc.version:
        name = f"{name} v{doc.version}"

    parts = [name]
    if doc.status == "superseded":
        warning = "STATUS: SUPERSEDED"
        if doc.superseded_by:
            warning += f", replaced by {doc.superseded_by}"
        parts.append(warning)
    if section.header:
        parts.append(f"Section: {section.header}")
    if doc.effective_date:
        parts.append(f"Effective {doc.effective_date}")
    return "[" + " | ".join(parts) + "]"


def _split_prose(text: str) -> list[str]:
    """Paragraph-aware split for sections above CHUNK_MAX_CHARS.

    The corpus is small enough that this never fires today (the largest prose
    section is ~420 chars). It exists so that adding a long document later
    degrades gracefully instead of producing one giant chunk.
    """
    if len(text) <= config.CHUNK_MAX_CHARS:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > config.CHUNK_TARGET_CHARS and current:
            pieces.append(current)
            tail = current[-config.CHUNK_OVERLAP_CHARS:] if config.CHUNK_OVERLAP_CHARS else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [text]


def chunk_document(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        # A table section is emitted whole, however long it is.
        bodies = [section.text] if section.is_table else _split_prose(section.text)
        context_line = build_context_line(doc, section)
        for part in bodies:
            if not part.strip():
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks):03d}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                section=section.header,
                text=part.strip(),
                context_line=context_line,
                is_table=section.is_table,
                version=doc.version,
                effective_date=doc.effective_date,
                owner=doc.owner,
                status=doc.status,
                superseded_by=doc.superseded_by,
                language=doc.language,
                contains_injection=doc.contains_injection,
            ))
    return chunks
