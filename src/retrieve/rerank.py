"""Cross-encoder reranking via FastEmbed (ONNX, no PyTorch).

Plan 2.5. Verified in M0 that FastEmbed 0.5.1 really does list
`jinaai/jina-reranker-v2-base-multilingual` (1.11 GB, cc-by-nc-4.0), so plan
risk R8 is closed and R9's licence flag is confirmed: NON-COMMERCIAL ONLY.
`BAAI/bge-reranker-base` (MIT) is the fallback and is also listed.

ONE DEVIATION FROM THE PLAN, deliberate. Plan 4.1 layer 3 says to apply a
"post-rerank score multiplier x0.5" to superseded chunks. Cross-encoder
outputs are raw LOGITS and are frequently negative, and multiplying a
negative number by 0.5 makes it LARGER -- the demotion would silently become
a promotion. Scores are therefore squashed through a sigmoid into (0, 1)
first. Sigmoid is monotonic, so it changes no ordering; it just makes the
multiplier mean what the plan intended. It also hands M5 a calibrated 0-1
number for the relevance floor, which is far easier to threshold than a logit.
"""
from __future__ import annotations

import math
from typing import Any

from .. import config
from ..index.types import Hit


class RerankerUnavailable(RuntimeError):
    """Model could not be loaded. Callers fall back to fusion order."""


_ENCODER: Any = None
_ENCODER_NAME: str = ""


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)                 # avoid overflow for large negative x
    return exp_x / (1.0 + exp_x)


def load_reranker(model: str | None = None):
    """Load once and cache. First call downloads ~1.1 GB."""
    global _ENCODER, _ENCODER_NAME
    model = model or config.RERANKER_MODEL
    if _ENCODER is not None and _ENCODER_NAME == model:
        return _ENCODER

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:
        raise RerankerUnavailable(f"fastembed not installed: {exc}") from exc

    try:
        config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _ENCODER = TextCrossEncoder(model_name=model,
                                    cache_dir=str(config.MODEL_CACHE_DIR))
        _ENCODER_NAME = model
    except Exception as exc:                      # noqa: BLE001
        raise RerankerUnavailable(f"cannot load reranker {model}: {exc}") from exc
    return _ENCODER


def rerank(query: str, hits: list[Hit], top_k: int | None = None,
           model: str | None = None,
           superseded_multiplier: float | None = None) -> list[Hit]:
    """Score candidates against the query, demote superseded, return top_k.

    Raises RerankerUnavailable so the caller can degrade to fusion order
    (plan M4.5: "Reranker fails -> fall back to RRF fusion order", with a
    banner, never a silent wrong answer).
    """
    if not hits:
        return []
    top_k = top_k or config.FINAL_TOP_K
    multiplier = (config.SUPERSEDED_MULTIPLIER
                  if superseded_multiplier is None else superseded_multiplier)

    encoder = load_reranker(model)
    documents = [hit.embed_text for hit in hits]
    try:
        raw_scores = list(encoder.rerank(query, documents))
    except Exception as exc:                      # noqa: BLE001
        raise RerankerUnavailable(f"rerank call failed: {exc}") from exc

    if len(raw_scores) != len(hits):
        raise RerankerUnavailable(
            f"reranker returned {len(raw_scores)} scores for {len(hits)} documents")

    for hit, raw in zip(hits, raw_scores):
        probability = _sigmoid(float(raw))
        hit.debug["rerank_logit"] = float(raw)
        hit.debug["rerank_score"] = probability
        if hit.status == "superseded":
            # Demoted, never deleted -- Q29 cites BOTH documents (plan 4.1).
            probability *= multiplier
            hit.debug["superseded_demoted"] = True
        hit.score = probability
        hit.stage = "rerank"

    return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
