"""Ingest-time prompt-injection stripping -- guardrail defence #1.

Plan 4.2. The subtlety: Q39 requires the bot to STILL correctly answer
"when is printer maintenance?" from the very file that carries the payload.
Blacklisting the document fails the eval. So the payload is removed while the
legitimate content survives untouched.

Every strip is logged to storage/sanitised.log for the report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Each pattern is named so the log says WHY a line was removed.
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("system_note_to_ai", re.compile(
        r"system\s+note\s+to\s+(ai|assistant|llm|bot)", re.I)),
    # The temporal word ("previous", "above") is OPTIONAL: a bare "ignore your
    # instructions" is just as much an injection. A determiner or possessive
    # is required instead, which is what keeps ordinary prose like
    # "Instructions for creating a ticket are in the user guide" clear of it.
    ("ignore_previous", re.compile(
        r"\b(ignore|disregard|forget)\s+"
        r"(?:(?:all|any|the|these|those)\s+)*"
        r"(?:of\s+)?(?:your|my|its)?\s*"
        r"(?:(?:previous|prior|above|preceding|earlier|former)\s+)?"
        r"(instruction|direction|rule|prompt|command)s?\b", re.I)),
    ("reveal_prompt", re.compile(
        r"\b(print|show|reveal|output|repeat|display|dump)\b[^.\n]{0,40}?"
        r"\b(system\s+prompt|your\s+instructions|the\s+text\s+above)", re.I)),
    ("role_override", re.compile(
        r"\byou\s+are\s+now\s+(a|an|the)\b|\bnew\s+instructions?\s*:", re.I)),
    ("tell_every_user", re.compile(
        r"\btell\s+(every|all|each)\s+(user|customer|person)s?\b", re.I)),
    ("override_marker", re.compile(
        r"^\s*(\[|<)?\s*(important|urgent|admin|override|developer)\s*"
        r"(note|message|instruction)s?\s*(\]|>)?\s*(to\s+ai)?\s*:", re.I)),
)


@dataclass(frozen=True)
class Strip:
    """One removed line, kept for the audit log."""
    doc: str
    line_no: int
    pattern: str
    text: str


def find_injections(text: str, doc: str = "") -> list[Strip]:
    """Report, without modifying. Used by tests and by the report."""
    found: list[Strip] = []
    for line_no, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        for name, pattern in INJECTION_PATTERNS:
            if pattern.search(line):
                found.append(Strip(doc, line_no, name, line.strip()))
                break
    return found


def sanitise(text: str, doc: str = "") -> tuple[str, list[Strip]]:
    """Return (clean_text, strips).

    Whole lines are removed. The surrounding legitimate content -- the
    maintenance schedule, the vendor email -- is left byte-identical.
    """
    strips = find_injections(text, doc)
    if not strips:
        return text, []

    drop = {s.line_no for s in strips}
    kept = [line for n, line in enumerate(text.split("\n"), start=1) if n not in drop]

    # Collapse the blank run the removal may have left behind.
    cleaned: list[str] = []
    for line in kept:
        if not line.strip() and cleaned and not cleaned[-1].strip():
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip(), strips


def log_strips(strips: list[Strip], log_path: Path) -> None:
    """Append-only audit trail. Written even when the list is empty is wrong --
    callers only call this with content, so the file's existence is meaningful."""
    if not strips:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as handle:
        for s in strips:
            handle.write(f"{stamp}\t{s.doc}\tline {s.line_no}\t{s.pattern}\t{s.text}\n")
