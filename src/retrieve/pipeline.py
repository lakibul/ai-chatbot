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

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from .. import config
from ..index.embed import EmbeddingError
from ..index.lexical import BM25Index
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

    @property
    def is_degraded(self) -> bool:
        return bool(self.degraded)

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


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
                                 collection=self.collection, client=self.client)
        except EmbeddingError as exc:
            result.degraded.append(f"embeddings unavailable, BM25 only ({exc})")
            return []
        except Exception as exc:                  # noqa: BLE001
            result.degraded.append(f"vector search failed, BM25 only ({exc})")
            return []

    # ----------------------------------------------------------- retrieve --
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or config.FINAL_TOP_K
        result = RetrievalResult(hits=[])

        lexical_hits = self._lexical(query, result)
        vector_hits = self._vector(query, result)
        result.stages["bm25_count"] = len(lexical_hits)
        result.stages["vector_count"] = len(vector_hits)

        if not lexical_hits and not vector_hits:
            # Genuinely nothing found. Distinct from a refusal and from a
            # failure -- the caller must keep all three apart (plan M4.5).
            result.stages["reason"] = "no candidates"
            return result

        fused = reciprocal_rank_fusion([lexical_hits, vector_hits])
        boosted = apply_code_boost(query, fused)
        candidates = boosted[:config.RERANK_CANDIDATES]
        result.stages["candidates"] = len(candidates)

        if not self.use_reranker:
            result.hits = candidates[:top_k]
            result.stages["ranking"] = "rrf"
            return result

        try:
            result.hits = rerank(query, candidates, top_k=top_k,
                                 model=self.reranker_model)
            result.stages["ranking"] = "rerank"
        except RerankerUnavailable as exc:
            result.degraded.append(f"reranker unavailable, using fusion order ({exc})")
            result.hits = candidates[:top_k]
            result.stages["ranking"] = "rrf"
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
