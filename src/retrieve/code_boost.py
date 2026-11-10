"""Exact-identifier boost.

Plan 4.3 and plan 3: this sits BEFORE the reranker, deliberately. "A
cross-encoder has no special respect for ND-1042" -- it scores semantic
relatedness, and every row of a 6-row error table looks about equally related
to "what does error ND-1042 mean". Lexical certainty has to be applied while
it still means something.

The boost is additive and large (config.CODE_BOOST) so that a chunk literally
containing the queried identifier cannot be pushed out of the rerank window by
RRF scores that live in the 0.0-0.03 range.
"""
from __future__ import annotations

from .. import config
from ..index.lexical import extract_codes
from ..index.types import Hit
from ..ingest.normalise import fold_for_lexical


def apply_code_boost(query: str, hits: list[Hit],
                     boost: float | None = None) -> list[Hit]:
    """Re-sort `hits`, lifting any chunk that literally contains a queried code.

    No-op when the query contains no identifier, which is the common case --
    so this cannot distort ordinary questions.
    """
    boost = config.CODE_BOOST if boost is None else boost
    codes = extract_codes(query)
    if not codes:
        return hits

    for hit in hits:
        haystack = fold_for_lexical(hit.embed_text)
        matched = [code for code in codes if code in haystack]
        if matched:
            # Scale by how many distinct queried codes the chunk contains, so
            # a chunk matching both "ND-1042" and "Retry-After" outranks one
            # matching only the first.
            hit.score += boost * len(matched)
            hit.debug["code_boost"] = matched

    return sorted(hits, key=lambda h: h.score, reverse=True)
