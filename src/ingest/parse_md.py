"""Header-aware markdown parser.

Produces one Section per '##' / '###' heading, plus the front-matter block.
Documents with no headings at all (the FAQ, the error-code table, the vendor
notice) yield a single section -- they are small enough that this is correct
rather than lazy.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .models import Document, Section
from .normalise import normalise_text, bengali_digits_to_ascii


def _nfc(text: str) -> str:
    """Bengali 'ya-phala' U+09DF is a composition exclusion: NFC DECOMPOSES it
    to U+09AF + U+09BC. Since parse_markdown NFC-normalises the document first,
    every Bangla literal in this module must be folded the same way or it will
    silently never match. This bit us once; it is now impossible to forget."""
    return unicodedata.normalize("NFC", text)


_H1 = re.compile(r"^#\s+(.*)$")
_H2 = re.compile(r"^##\s+(.*)$")
_H3 = re.compile(r"^###\s+(.*)$")

_VERSION_EN = re.compile(r"\bVersion\s+([0-9]+(?:\.[0-9]+)*)", re.I)
_VERSION_BN = re.compile(_nfc(r"সংস্করণ\s+([০-৯0-9.]+)"))
_OWNER_EN = re.compile(r"\bOwner:\s*([^\n·]+)", re.I)
_OWNER_BN = re.compile(_nfc(r"দায়িত্বে:\s*([^\n·]+)"))
_EFFECTIVE_EN = re.compile(r"\bEffective\s+([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4})", re.I)
_EFFECTIVE_BN = re.compile(
    _nfc(r"কার্যকর:\s*([০-৯0-9]{1,2}\s+\S+\s+[০-৯0-9]{4})"))
_STATUS = re.compile(r"^STATUS:\s*([A-Z]+)", re.M)
# [^,.]+ would stop at the dot in "v3.0", so allow a dot that is followed by a digit.
_REPLACED_BY = re.compile(r"replaced by the ((?:[^,.]|\.(?=\d))+)", re.I)

_MONTHS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
# Bangla month names, NFC-folded for the reason in _nfc above.
_MONTHS_BN = {
    _nfc(name): number for name, number in {
        "জানুয়ারি": 1, "ফেব্রুয়ারি": 2, "মার্চ": 3, "এপ্রিল": 4,
        "মে": 5, "জুন": 6, "জুলাই": 7, "আগস্ট": 8,
        "সেপ্টেম্বর": 9, "অক্টোবর": 10, "নভেম্বর": 11, "ডিসেম্বর": 12,
    }.items()
}


def parse_date(raw: str) -> str:
    """'1 January 2026' or '১ জানুয়ারি ২০২৬' -> '2026-01-01'. '' when unparseable.

    The ISO form is what makes "the later effective date wins" (plan 4.1) a
    comparison rather than a guess.

    The input is NFC-folded here rather than assumed: callers outside
    parse_markdown (tests, the PDF parser) may hand us a precomposed U+09DF,
    which would otherwise miss every Bangla month name.
    """
    parts = bengali_digits_to_ascii(_nfc(raw.strip())).split()
    if len(parts) != 3:
        return ""
    day, month_raw, year = parts
    month = _MONTHS_EN.get(month_raw.lower()) or _MONTHS_BN.get(month_raw)
    if not month or not day.isdigit() or not year.isdigit():
        return ""
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _looks_bangla(text: str) -> bool:
    bangla = sum(1 for ch in text if "ঀ" <= ch <= "৿")
    return bangla > max(20, len(text) * 0.15)


def parse_markdown(path: Path) -> Document:
    raw = normalise_text(path.read_text(encoding="utf-8"))
    lines = raw.split("\n")

    title = ""
    for line in lines:
        match = _H1.match(line)
        if match:
            title = match.group(1).strip()
            break
    if not title:
        title = path.stem

    doc = Document(doc_id=path.name, title=title)
    doc.language = "bn" if _looks_bangla(raw) else "en"

    # --- front matter: everything between the H1 and the first H2 ----------
    front: list[str] = []
    for line in lines:
        if _H2.match(line):
            break
        if not _H1.match(line):
            front.append(line)
    front_text = "\n".join(front).strip()

    version = _VERSION_EN.search(front_text) or _VERSION_BN.search(front_text)
    if version:
        doc.version = bengali_digits_to_ascii(version.group(1)).strip()
    owner = _OWNER_EN.search(front_text) or _OWNER_BN.search(front_text)
    if owner:
        doc.owner = owner.group(1).strip()
    effective = _EFFECTIVE_EN.search(front_text) or _EFFECTIVE_BN.search(front_text)
    if effective:
        doc.effective_date = parse_date(effective.group(1))

    # Plan 4.1 layer 1: status from the STATUS: line AND from the filename.
    # Either alone is enough; both is the normal case.
    status_match = _STATUS.search(raw)
    if (status_match and status_match.group(1).upper() == "SUPERSEDED") \
            or "SUPERSEDED" in path.name.upper():
        doc.status = "superseded"
        replaced = _REPLACED_BY.search(raw)
        if replaced:
            doc.superseded_by = replaced.group(1).strip()

    # --- sections ----------------------------------------------------------
    sections: list[Section] = []
    current_path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(
                header_path=list(current_path),
                text=body,
                is_table=any(_is_table_line(ln) for ln in body.split("\n")),
            ))
        buffer.clear()

    for line in lines:
        if _H1.match(line):
            continue
        h2 = _H2.match(line)
        h3 = _H3.match(line)
        if h2:
            flush()
            current_path = [h2.group(1).strip()]
            continue
        if h3:
            flush()
            current_path = current_path[:1] + [h3.group(1).strip()]
            continue
        buffer.append(line)
    flush()

    doc.sections = sections
    return doc
