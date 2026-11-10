"""Reciprocal Rank Fusion.

Plan 3: fuse BM25 top-20 and vector top-20 with k=60.

RRF is used rather than score averaging because BM25 scores and cosine
similarities live on incomparable scales -- a BM25 score of 33 and a cosine of
0.74 cannot be added. RRF throws the magnitudes away and keeps only the ranks,
which is exactly what makes it safe here.
"""
from __future__ import annotations

from typing import Iterable

from .. import config
from ..index.types import Hit


def reciprocal_rank_fusion(rankings: Iterable[list[Hit]],
                           k: int | None = None) -> list[Hit]:
    """Fuse ranked lists. Returns one Hit per unique chunk, best first.

    score(d) = sum over lists of 1 / (k + rank(d)), rank 1-based.
    A document missing from a list contributes nothing from that list.
    """
    k = config.RRF_K if k is None else k

    fused: dict[str, Hit] = {}
    contributions: dict[str, dict[str, float]] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            contribution = 1.0 / (k + rank)
            existing = fused.get(hit.chunk_id)
            if existing is None:
                existing = Hit(chunk_id=hit.chunk_id, score=0.0,
                               payload=hit.payload, stage="rrf",
                               debug={"sources": {}})
                fused[hit.chunk_id] = existing
                contributions[hit.chunk_id] = {}
            existing.score += contribution
            contributions[hit.chunk_id][hit.stage or "unknown"] = float(rank)

    for chunk_id, hit in fused.items():
        hit.debug["sources"] = contributions[chunk_id]

    return sorted(fused.values(), key=lambda h: h.score, reverse=True)
