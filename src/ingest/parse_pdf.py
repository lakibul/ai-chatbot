"""PDF parser for expense_reimbursement_policy.pdf.

Plan 2.6 established by decompressing the file that it is Chromium-printed
HTML: a real text layer with working ToUnicode CMaps (so no OCR), but table
borders drawn as 48 FILLED RECTANGLES with zero stroked lines.

Measured outcome: pdfplumber derives cell edges from those rects on its own,
and the default "lines" strategy recovers the header plus all five limit rows
with correct columns. The coordinate-clustering plan B from 2.6 is therefore
not needed, and R3 is closed.

The remaining work is mechanical:
  * cells wrap mid-phrase ("BDT 6,000 per\\nnight"), so intra-cell newlines
    are joined back into spaces;
  * extract_text() would re-emit the table as mangled prose, so prose is read
    from the regions ABOVE and BELOW the table bbox instead of filtered after
    the fact.
"""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from .models import Document, Section
from .normalise import normalise_text
from .parse_md import parse_date

# The corpus PDF is a single page with a single table of exactly these
# dimensions. Ingestion asserts this (plan M4.5: ingestion must fail loudly).
EXPECTED_TABLE_ROWS = 6          # 1 header + 5 limit rows
EXPECTED_TABLE_COLS = 3

_HEADING = re.compile(r"^(\d+)\.\s+(\S.*)$")
_VERSION = re.compile(r"\bVersion\s+([0-9]+(?:\.[0-9]+)*)", re.I)
_EFFECTIVE = re.compile(r"\bEffective\s+([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4})", re.I)
_OWNER = re.compile(r"\bOwner:\s*([^\n·]+)", re.I)


class PdfParseError(RuntimeError):
    """Raised when the PDF does not match its known shape."""


def _clean_cell(cell: str | None) -> str:
    """Join the mid-phrase wraps the HTML-to-PDF printer introduced."""
    if not cell:
        return ""
    return re.sub(r"\s+", " ", cell.replace("\n", " ")).strip()


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Render extracted cells as a GitHub-style markdown table.

    Markdown is the right target: every other document in the corpus already
    uses it, so one chunking rule and one prompt format cover all 12 files.
    """
    cleaned = [[_clean_cell(c) for c in row] for row in rows]
    header, *body = cleaned
    width = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in body:
        padded = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


def _sections_from_text(text: str, keep_empty_tail: bool = False) -> list[tuple[list[str], str]]:
    """Split flat prose into (header_path, body) on '1. Heading' lines.

    keep_empty_tail matters for the region above the table: "2. Limits" is a
    heading whose entire body IS the table, so it arrives here with an empty
    body. Dropping it would silently file the limits table under "1. How to
    claim" -- which is exactly the kind of mis-attribution that makes
    "BDT 6,000" unanswerable for Q23.
    """
    out: list[tuple[list[str], str]] = []
    current: list[str] = []
    buffer: list[str] = []
    saw_heading = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            out.append((list(current), body))
        buffer.clear()

    for line in text.split("\n"):
        heading = _HEADING.match(line.strip())
        if heading:
            flush()
            current = [f"{heading.group(1)}. {heading.group(2).strip()}"]
            saw_heading = True
            continue
        buffer.append(line)

    tail_is_bare_heading = saw_heading and not "\n".join(buffer).strip()
    flush()
    if keep_empty_tail and tail_is_bare_heading and (not out or out[-1][0] != current):
        out.append((list(current), ""))
    return out


def parse_pdf(path: Path) -> Document:
    with pdfplumber.open(str(path)) as pdf:
        if len(pdf.pages) != 1:
            raise PdfParseError(
                f"{path.name}: expected 1 page, found {len(pdf.pages)}")
        page = pdf.pages[0]

        tables = page.find_tables()
        if len(tables) != 1:
            raise PdfParseError(
                f"{path.name}: expected exactly 1 table, found {len(tables)}")
        table = tables[0]
        rows = table.extract()
        if len(rows) != EXPECTED_TABLE_ROWS:
            raise PdfParseError(
                f"{path.name}: expected {EXPECTED_TABLE_ROWS} table rows "
                f"(1 header + 5 limits), found {len(rows)}")
        if any(len(r) != EXPECTED_TABLE_COLS for r in rows):
            raise PdfParseError(
                f"{path.name}: expected {EXPECTED_TABLE_COLS} columns in every row, "
                f"found {[len(r) for r in rows]}")

        table_md = table_to_markdown(rows)

        # Prose from above and below the table, never through it.
        _, top, _, bottom = table.bbox
        above = page.crop((0, 0, page.width, top)).extract_text() or ""
        below = page.crop((0, bottom, page.width, page.height)).extract_text() or ""

    above = normalise_text(above)
    below = normalise_text(below)

    lines = above.split("\n")
    title = lines[0].strip() if lines else path.stem
    front = "\n".join(lines[1:3]).strip()   # the "Version ... Effective ... Owner:" line

    doc = Document(doc_id=path.name, title=title)
    version = _VERSION.search(front)
    if version:
        doc.version = version.group(1)
    effective = _EFFECTIVE.search(front)
    if effective:
        doc.effective_date = parse_date(effective.group(1))
    owner = _OWNER.search(front)
    if owner:
        doc.owner = owner.group(1).strip()

    sections: list[Section] = []

    # Everything above the table, minus the H1 line, split on numbered headings.
    above_body = "\n".join(lines[1:]).strip()
    above_sections = _sections_from_text(above_body, keep_empty_tail=True)

    # The last section above the table is the one the table belongs to
    # ("2. Limits"). Its prose and the table become ONE chunk -- plan 4.4:
    # a chunk either contains a whole table or none of it, and the heading
    # must travel with it or "BDT 2,500" is unattributable.
    for index, (header_path, body) in enumerate(above_sections):
        is_last = index == len(above_sections) - 1
        if is_last:
            merged = f"{body}\n\n{table_md}".strip() if body else table_md
            sections.append(Section(header_path=header_path, text=merged, is_table=True))
        else:
            sections.append(Section(header_path=header_path, text=body))

    if not above_sections:          # defensive: table with no preceding heading
        sections.append(Section(header_path=["Limits"], text=table_md, is_table=True))

    for header_path, body in _sections_from_text(below):
        sections.append(Section(header_path=header_path, text=body))

    doc.sections = sections
    return doc
