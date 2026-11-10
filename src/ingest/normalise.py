"""Unicode and digit normalisation.

Plan 4.5: "Unicode NFC, and Bengali digits mapped to ASCII in the BM25 index
ONLY. The original Bangla text is preserved for display and for the LLM, so
answers can be given in Bangla as Q32/Q33 expect."

So this module offers two different functions and the caller must pick the
right one. `normalise_text` is safe everywhere. `fold_for_lexical` is
destructive of script and must never touch stored chunk text.
"""
from __future__ import annotations

import re
import unicodedata

# Bengali (Bangla) digits -> ASCII. U+09E6..U+09EF.
BENGALI_DIGITS = {
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
}
_BENGALI_DIGIT_RE = re.compile("[" + "".join(BENGALI_DIGITS) + "]")

# Typographic characters that break exact-string matching on codes and money.
_PUNCT_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "", "﻿": "",
}


def normalise_text(text: str) -> str:
    """NFC + whitespace tidy. Script-preserving: safe for stored chunk text.

    Deliberately does NOT touch digits -- Q32's expected answer is written in
    Bengali numerals, so the displayed text must keep them.
    """
    text = unicodedata.normalize("NFC", text)
    for bad, good in _PUNCT_FOLD.items():
        text = text.replace(bad, good)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Trailing spaces only -- blank lines are structure, never collapse them.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def bengali_digits_to_ascii(text: str) -> str:
    """'8:15' from '৮:১৫'. Lexical index only."""
    return _BENGALI_DIGIT_RE.sub(lambda m: BENGALI_DIGITS[m.group()], text)


def fold_for_lexical(text: str) -> str:
    """Aggressive fold for the BM25 side: NFC, ASCII digits, lowercase.

    Never store the result as chunk text -- it is an index key, not content.
    """
    return bengali_digits_to_ascii(normalise_text(text)).lower()
