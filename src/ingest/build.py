"""The ingestion pipeline: read -> sanitise -> parse -> chunk, with asserts.

Plan M4.5 sets the philosophy for this file, and it is the OPPOSITE of the
one that governs the request path: "ingestion is offline, so silence is the
enemy." A run that indexes 11 of 12 files exits non-zero. Silently shipping an
index missing the PDF would cost a 45-minute eval run to discover.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import config
from .chunk import chunk_document
from .models import Chunk, Document
from .parse_md import parse_markdown
from .parse_pdf import parse_pdf
from .sanitise import Strip, log_strips, sanitise

# Facts about the corpus that ingestion refuses to proceed without.
PDF_NAME = "expense_reimbursement_policy.pdf"
PDF_REQUIRED_LIMITS = ("BDT 1,500", "BDT 800", "BDT 6,000", "BDT 1,000", "BDT 700")


class IngestError(RuntimeError):
    """Any violated corpus invariant. Always fatal."""


def _sanitise_document(doc: Document) -> list[Strip]:
    """Strip injection payloads from every section, in place.

    Plan 4.2: the payload must be removed while the legitimate content
    survives, because Q39 asks a real question of the poisoned document.
    """
    all_strips: list[Strip] = []
    for section in doc.sections:
        clean, strips = sanitise(section.text, doc.doc_id)
        if strips:
            section.text = clean
            all_strips.extend(strips)
    if all_strips:
        doc.contains_injection = True
    doc.sections = [s for s in doc.sections if s.text.strip()]
    return all_strips


def load_documents(data_dir: Path | None = None) -> tuple[list[Document], list[Strip]]:
    data_dir = data_dir or config.DATA_DIR
    if not data_dir.is_dir():
        raise IngestError(f"data directory not found: {data_dir}")

    paths = sorted(p for p in data_dir.iterdir()
                   if p.suffix.lower() in {".md", ".pdf"} and p.is_file())

    if len(paths) != config.EXPECTED_SOURCE_FILES:
        raise IngestError(
            f"expected exactly {config.EXPECTED_SOURCE_FILES} source files in "
            f"{data_dir}, found {len(paths)}: {[p.name for p in paths]}")

    documents: list[Document] = []
    strips: list[Strip] = []
    for path in paths:
        doc = parse_pdf(path) if path.suffix.lower() == ".pdf" else parse_markdown(path)
        strips.extend(_sanitise_document(doc))
        if not doc.sections:
            raise IngestError(f"{path.name}: parsed to zero sections")
        documents.append(doc)
    return documents, strips


def _assert_corpus_invariants(documents: list[Document], chunks: list[Chunk]) -> None:
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    for doc in documents:
        if not by_doc.get(doc.doc_id):
            raise IngestError(f"{doc.doc_id}: produced zero chunks")

    # The PDF is the file most likely to fail silently (plan R14).
    pdf_chunks = by_doc.get(PDF_NAME, [])
    if not pdf_chunks:
        raise IngestError(f"{PDF_NAME}: missing from the index entirely")
    pdf_text = "\n".join(c.text for c in pdf_chunks)
    missing = [limit for limit in PDF_REQUIRED_LIMITS if limit not in pdf_text]
    if missing:
        raise IngestError(
            f"{PDF_NAME}: limits table incomplete, missing {missing}")
    if "BDT 25,000" not in pdf_text:
        raise IngestError(f"{PDF_NAME}: the CFO approval threshold is missing")

    # Plan 4.1: the superseded document must be PRESENT, not filtered out --
    # Q29 cites both documents.
    superseded = [d for d in documents if d.status == "superseded"]
    if len(superseded) != 1:
        raise IngestError(
            f"expected exactly 1 superseded document, found {len(superseded)}")

    # Plan 4.2: the injection must have been found and removed.
    poisoned = [d for d in documents if d.contains_injection]
    if len(poisoned) != 1:
        raise IngestError(
            f"expected exactly 1 document carrying an injection, found "
            f"{[d.doc_id for d in poisoned]}")
    for chunk in chunks:
        lowered = chunk.text.lower()
        if "ignore all previous instructions" in lowered or "system note to ai" in lowered:
            raise IngestError(
                f"{chunk.chunk_id}: injection payload survived sanitisation")

    # Plan 4.5: the Bangla document must still be Bangla.
    bangla = [d for d in documents if d.language == "bn"]
    if len(bangla) != 1:
        raise IngestError(f"expected exactly 1 Bangla document, found {len(bangla)}")


def build_chunks(data_dir: Path | None = None, write_log: bool = True) -> list[Chunk]:
    documents, strips = load_documents(data_dir)

    if write_log:
        config.ensure_storage()
        # Truncate first so the log describes THIS run, not every run ever.
        config.SANITISED_LOG.write_text("", encoding="utf-8")
        log_strips(strips, config.SANITISED_LOG)

    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc))

    _assert_corpus_invariants(documents, chunks)
    return chunks


def save_chunks(chunks: list[Chunk]) -> Path:
    config.ensure_storage()
    payload = [c.to_dict() for c in chunks]
    config.CHUNKS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return config.CHUNKS_PATH


def load_chunks() -> list[Chunk]:
    """Re-hydrate chunks saved by a previous run."""
    raw = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    fields = Chunk.__dataclass_fields__.keys()
    return [Chunk(**{k: v for k, v in item.items() if k in fields}) for item in raw]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingest",
        description="Parse the corpus into chunks and persist them.")
    parser.add_argument("--print", dest="do_print", action="store_true",
                        help="dump every chunk for eyeball review")
    parser.add_argument("--no-save", action="store_true",
                        help="parse and validate without writing storage/chunks.json")
    args = parser.parse_args(argv)

    try:
        chunks = build_chunks()
    except Exception as exc:                      # noqa: BLE001 - fail loudly
        print(f"INGEST FAILED: {exc}", file=sys.stderr)
        return 1

    if args.do_print:
        for chunk in chunks:
            print("=" * 78)
            print(f"{chunk.chunk_id}  [{chunk.status}] lang={chunk.language}"
                  f"{' TABLE' if chunk.is_table else ''}")
            print(chunk.context_line)
            print("-" * 78)
            print(chunk.text)
        print("=" * 78)

    by_doc: dict[str, int] = {}
    for chunk in chunks:
        by_doc[chunk.doc_id] = by_doc.get(chunk.doc_id, 0) + 1
    for doc_id in sorted(by_doc):
        print(f"  {by_doc[doc_id]:3d}  {doc_id}")

    tables = sum(1 for c in chunks if c.is_table)
    print(f"\nOK: {len(by_doc)} documents -> {len(chunks)} chunks "
          f"({tables} table chunks)")

    if not args.no_save:
        path = save_chunks(chunks)
        print(f"Saved: {path}")
        if config.SANITISED_LOG.exists():
            lines = [ln for ln in
                     config.SANITISED_LOG.read_text(encoding="utf-8").splitlines() if ln]
            print(f"Sanitised: {len(lines)} line(s) stripped -> {config.SANITISED_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
