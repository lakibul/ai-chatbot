"""The retrieval pipeline.

Plan 3:
    BM25 top-20  +  vector top-20
      -> RRF fuse (k=60)
        -> exact-code boost
          -> rerank (ONNX cross-encoder) -> top 5
            -> superseded down-rank
              -> relevance floor -> below? "I don't know"

Two ordering decisions matter and are load-bearing (plan 3):
  * the exact-code boost sits BEFORE the reranker, because a cross-encoder
    has no special respect for ND-1042;
  * the relevance floor sits AFTER the reranker, because its calibrated score
    is the only trustworthy signal for "I don't know".

Plan M4.5 governs failure: each stage has ONE defined fallback, so a partial
failure still produces a cited answer rather than a stack trace. Every
degradation is recorded in `RetrievalResult.degraded` so the UI can show a
banner -- never silently.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from .. import config
from ..index.embed import EmbeddingError
from ..index.lexical import BM25Index, extract_codes, tokenize
from ..index.types import Hit
from ..index.vector import search as vector_search
from .code_boost import apply_code_boost
from .rerank import RerankerUnavailable, rerank
from .rrf import reciprocal_rank_fusion


@dataclass
class RetrievalResult:
    hits: list[Hit]
    degraded: list[str] = field(default_factory=list)
    stages: dict[str, Any] = field(default_factory=dict)
    # Raw data for the UI's per-message pipeline graph. Purely additive: it is
    # filled from values already computed, and removing every line that
    # touches it leaves the hits, their order and their scores unchanged.
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def is_degraded(self) -> bool:
        return bool(self.degraded)

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


def _compact(hit: Hit, **extra: Any) -> dict[str, Any]:
    """One row of a ranked list, small enough to store with every answer."""
    row: dict[str, Any] = {"id": hit.chunk_id, "doc": hit.doc_id,
                           "section": hit.section, "score": round(hit.score, 4)}
    row.update(extra)
    return row


class Retriever:
    """Owns the long-lived index handles for a process.

    Qdrant local mode takes an exclusive file lock, so exactly one Retriever
    may exist per process and it must be closed before another opens.
    """

    def __init__(self, embed_model: str | None = None,
                 collection: str | None = None,
                 reranker_model: str | None = None,
                 use_reranker: bool = True):
        self.embed_model = embed_model or config.EMBED_MODEL
        self.collection = collection or config.COLLECTION_NAME
        self.reranker_model = reranker_model or config.RERANKER_MODEL
        self.use_reranker = use_reranker

        self.client = QdrantClient(path=str(config.QDRANT_PATH))
        self.bm25 = BM25Index.load()

    # ------------------------------------------------------------ stages --
    def _lexical(self, query: str, result: RetrievalResult) -> list[Hit]:
        try:
            return self.bm25.search(query, top_k=config.BM25_TOP_K)
        except Exception as exc:                  # noqa: BLE001
            result.degraded.append(f"lexical search failed ({exc})")
            return []

    def _vector(self, query: str, result: RetrievalResult) -> list[Hit]:
        try:
            return vector_search(query, top_k=config.VECTOR_TOP_K,
                                 model=self.embed_model,
                                 collection=self.collection, client=self.client,
                                 trace=result.trace.setdefault("vector_query", {}))
        except EmbeddingError as exc:
            result.degraded.append(f"embeddings unavailable, BM25 only ({exc})")
            return []
        except Exception as exc:                  # noqa: BLE001
            result.degraded.append(f"vector search failed, BM25 only ({exc})")
            return []

    def _select_candidates(self, ranked: list[Hit], lexical: list[Hit],
                           vector: list[Hit]) -> list[Hit]:
        """Choose the rerank window, reserving seats for each retriever's best.

        Plain `ranked[:N]` loses a document that one retriever is CERTAIN
        about whenever the other retriever returns plausible noise, because
        RRF rewards appearing in both lists over ranking first in one. That is
        not a hypothetical: for "What time does the return office bus leave?"
        the Bangla document is vector rank 1 at 0.724 -- far clear of the 0.532
        runner-up -- while BM25, which cannot match a single Bangla token,
        still returns twenty English documents. Fusion buried the right answer
        at rank 13.

        The cross-encoder resolves it decisively once it sees it (0.674 against
        0.069), so the fix is simply to guarantee it gets looked at.
        """
        limit = config.RERANK_CANDIDATES
        by_id = {hit.chunk_id: hit for hit in ranked}
        position = {hit.chunk_id: i for i, hit in enumerate(ranked)}

        picked: list[str] = []

        def take(chunk_id: str) -> None:
            if chunk_id in by_id and chunk_id not in picked and len(picked) < limit:
                picked.append(chunk_id)

        # An exact identifier match is certainty, not a guess: never drop it.
        for hit in ranked:
            if hit.debug.get("code_boost"):
                take(hit.chunk_id)

        for source in (vector, lexical):
            for hit in source[:config.GUARANTEED_PER_RETRIEVER]:
                take(hit.chunk_id)

        for hit in ranked:
            take(hit.chunk_id)

        # Hand them to the reranker in fused order, for determinism.
        return [by_id[cid] for cid in sorted(picked, key=lambda c: position[c])]

    # ----------------------------------------------------------- retrieve --
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or config.FINAL_TOP_K
        result = RetrievalResult(hits=[])

        # Per-stage wall clock, recorded only so the UI can show WHICH part
        # of a slow search was slow. Nothing here changes what is returned --
        # the hits, their order and their scores are identical with the
        # timers removed.
        mark = time.perf_counter()
        lexical_hits = self._lexical(query, result)
        result.stages["bm25_ms"] = round((time.perf_counter() - mark) * 1000, 1)

        mark = time.perf_counter()
        vector_hits = self._vector(query, result)
        result.stages["vector_ms"] = round((time.perf_counter() - mark) * 1000, 1)

        result.stages["bm25_count"] = len(lexical_hits)
        result.stages["vector_count"] = len(vector_hits)

        kept = config.TRACE_HITS_KEPT
        result.trace["bm25"] = {
            "query_tokens": tokenize(query)[:60],
            "hits": [_compact(h) for h in lexical_hits[:kept]],
        }
        result.trace["vector"] = {
            **result.trace.pop("vector_query", {}),
            "hits": [_compact(h) for h in vector_hits[:kept]],
        }

        if not lexical_hits and not vector_hits:
            # Genuinely nothing found. Distinct from a refusal and from a
            # failure -- the caller must keep all three apart (plan M4.5).
            result.stages["reason"] = "no candidates"
            return result

        mark = time.perf_counter()
        fused = reciprocal_rank_fusion([lexical_hits, vector_hits])
        # Captured BEFORE the boost, which edits these same objects in place.
        result.trace["fuse"] = {
            "k": config.RRF_K,
            "unique": len(fused),
            "hits": [_compact(h, ranks=dict(h.debug.get("sources", {})))
                     for h in fused[:kept]],
        }
        boosted = apply_code_boost(query, fused)
        candidates = self._select_candidates(boosted, lexical_hits, vector_hits)
        result.stages["fuse_ms"] = round((time.perf_counter() - mark) * 1000, 1)
        result.stages["fused_count"] = len(fused)
        # How many chunks carried a literal identifier match. Shown in the
        # inspector because this boost is the step that rescues ND-1042.
        result.stages["code_boosted"] = sum(
            1 for hit in boosted if hit.debug.get("code_boost"))
        result.stages["candidates"] = len(candidates)

        result.trace["boost"] = {
            "codes_in_question": extract_codes(query),
            "boost": config.CODE_BOOST,
            "matched": [_compact(h, matched=h.debug.get("code_boost"))
                        for h in boosted if h.debug.get("code_boost")],
        }
        # Why each passage earned a seat in the rerank window. Derived from the
        # same lists _select_candidates used; it does not influence the choice.
        guaranteed = config.GUARANTEED_PER_RETRIEVER
        top_vector = {h.chunk_id for h in vector_hits[:guaranteed]}
        top_lexical = {h.chunk_id for h in lexical_hits[:guaranteed]}

        def _why(hit: Hit) -> str:
            if hit.debug.get("code_boost"):
                return "exact identifier"
            if hit.chunk_id in top_vector:
                return "top meaning match"
            if hit.chunk_id in top_lexical:
                return "top keyword match"
            return "fusion rank"
        result.trace["window"] = {
            "limit": config.RERANK_CANDIDATES,
            "reserved_per_search": guaranteed,
            "candidates": [_compact(h, why=_why(h)) for h in candidates],
        }

        if not self.use_reranker:
            result.hits = candidates[:top_k]
            result.stages["ranking"] = "rrf"
            return result

        try:
            mark = time.perf_counter()
            result.hits = rerank(query, candidates, top_k=top_k,
                                 model=self.reranker_model)
            result.stages["rerank_ms"] = round((time.perf_counter() - mark) * 1000, 1)
            result.stages["ranking"] = "rerank"
            final_ids = {h.chunk_id for h in result.hits}
            result.trace["rerank"] = {
                "available": True,
                "superseded_multiplier": config.SUPERSEDED_MULTIPLIER,
                "candidates": sorted((
                    {"id": h.chunk_id, "doc": h.doc_id, "section": h.section,
                     "logit": round(h.debug.get("rerank_logit", 0.0), 3),
                     "probability": round(h.debug.get("rerank_score", 0.0), 4),
                     "score": round(h.score, 4),
                     "demoted": bool(h.debug.get("superseded_demoted")),
                     "kept": h.chunk_id in final_ids}
                    for h in candidates), key=lambda r: r["score"], reverse=True),
            }
        except RerankerUnavailable as exc:
            result.degraded.append(f"reranker unavailable, using fusion order ({exc})")
            result.hits = candidates[:top_k]
            result.stages["ranking"] = "rrf"
            result.trace["rerank"] = {"available": False, "reason": str(exc)}
        return result

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:                         # noqa: BLE001
            pass

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_hybrid_retriever(model: str | None = None, collection: str | None = None,
                           use_reranker: bool = True):
    """Adapter for the eval harness: query -> list[Hit]."""
    retriever = Retriever(embed_model=model, collection=collection,
                          use_reranker=use_reranker)

    def retrieve(query: str, top_k: int):
        return retriever.retrieve(query, top_k=top_k).hits

    retrieve.retriever = retriever                # keep it alive and closable
    return retrieve
