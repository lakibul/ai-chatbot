"""Embeddings via Ollama's native /api/embed.

Plan 2.2: embeddings go through Ollama rather than PyTorch. That avoids a
~1.5 GB torch install and keeps ONE model server for the whole project, which
also means OLLAMA_MAX_LOADED_MODELS=2 can hold the embedder and the LLM
co-resident instead of letting them evict each other between query-embed and
generate (plan 5.3 -- an eviction costs a 10-20 s reload PER MESSAGE).
"""
from __future__ import annotations

import httpx

from .. import config


class EmbeddingError(RuntimeError):
    """Embedding call failed. Distinct from "no results" -- plan M4.5 keeps
    failure and refusal on separate channels from the first line of code."""


def embed(texts: list[str], model: str | None = None,
          timeout: float | None = None) -> list[list[float]]:
    """Embed a batch. Returns one vector per input, in order.

    Retries once on a connection error (cheap -- usually Ollama swapping a
    model in) but never on a timeout, per the plan's retry policy.
    """
    if not texts:
        return []

    model = model or config.EMBED_MODEL
    payload = {"model": model, "input": texts,
               "options": {"num_gpu": config.NUM_GPU, "num_thread": config.NUM_THREAD},
               "keep_alive": config.KEEP_ALIVE}
    limits = httpx.Timeout(timeout or config.EMBED_TIMEOUT,
                           connect=config.CONNECT_TIMEOUT)

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = httpx.post(config.OLLAMA_EMBED_URL, json=payload, timeout=limits)
            response.raise_for_status()
            vectors = response.json().get("embeddings")
            if not vectors or len(vectors) != len(texts):
                raise EmbeddingError(
                    f"{model}: expected {len(texts)} vectors, got "
                    f"{0 if not vectors else len(vectors)}")
            return vectors
        except httpx.TimeoutException as exc:
            raise EmbeddingError(f"{model}: embedding timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"{model}: Ollama returned {exc.response.status_code} "
                f"-- {exc.response.text[:200]}") from exc
        except httpx.RequestError as exc:
            last_error = exc
            if attempt == 0:
                continue
    raise EmbeddingError(
        f"{model}: cannot reach Ollama at {config.OLLAMA_HOST} ({last_error})")


def embed_one(text: str, model: str | None = None) -> list[float]:
    return embed([text], model=model)[0]


def embedding_dimension(model: str | None = None) -> int:
    """Ask the model itself rather than hard-coding 1024 for bge-m3 --
    the nomic comparison in plan 2.2 uses a different width."""
    return len(embed_one("dimension probe", model=model))
