"""Vector store: Qdrant in local (embedded) mode.

Plan 2.3 -- the single most important infrastructure decision. Docker is not
installed, Docker Desktop would cost ~2 GB of the 16 GB we need for the model,
and Qdrant ships no native Windows server binary. Local mode gives the
IDENTICAL Python API persisted to disk with no server process, so moving to a
real Qdrant server later is a one-line constructor change.

Local mode holds an exclusive file lock, so exactly one client may be open at
a time. Every entry point here is a context manager for that reason.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from qdrant_client import QdrantClient, models

from .. import config
from ..ingest.models import Chunk
from .embed import embed
from .types import Hit

__all__ = ["Hit", "build_vector_index", "search", "open_store", "collection_info"]


@contextmanager
def open_store(path: str | None = None) -> Iterator[QdrantClient]:
    client = QdrantClient(path=str(path or config.QDRANT_PATH))
    try:
        yield client
    finally:
        client.close()


def build_vector_index(chunks: list[Chunk], model: str | None = None,
                       collection: str | None = None,
                       batch_size: int = 16) -> int:
    """(Re)create the collection and embed every chunk. Returns the count.

    The collection is recreated rather than updated: the corpus is ~50 chunks
    and a stale vector from a previous chunking strategy is a silent,
    expensive bug.
    """
    model = model or config.EMBED_MODEL
    collection = collection or config.COLLECTION_NAME
    config.ensure_storage()

    if not chunks:
        raise ValueError("refusing to build an empty vector index")

    texts = [c.embed_text for c in chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embed(texts[start:start + batch_size], model=model))

    dimension = len(vectors[0])
    if any(len(v) != dimension for v in vectors):
        raise ValueError("embedding model returned inconsistent dimensions")

    with open_store() as client:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE),
        )
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id=i, vector=vectors[i], payload=chunks[i].to_dict())
                for i in range(len(chunks))
            ],
        )
        return client.count(collection, exact=True).count


def search(query: str, top_k: int | None = None, model: str | None = None,
           collection: str | None = None, client: QdrantClient | None = None) -> list[Hit]:
    """Embed the query and return the nearest chunks."""
    top_k = top_k or config.VECTOR_TOP_K
    collection = collection or config.COLLECTION_NAME
    vector = embed([query], model=model)[0]

    def _run(c: QdrantClient) -> list[Hit]:
        found = c.query_points(
            collection_name=collection, query=vector, limit=top_k, with_payload=True
        ).points
        return [Hit(chunk_id=p.payload.get("chunk_id", str(p.id)),
                    score=float(p.score), payload=p.payload, stage="vector")
                for p in found]

    if client is not None:
        return _run(client)
    with open_store() as owned:
        return _run(owned)


def collection_info(collection: str | None = None) -> dict[str, Any]:
    collection = collection or config.COLLECTION_NAME
    with open_store() as client:
        if not client.collection_exists(collection):
            return {"exists": False, "count": 0, "dimension": 0}
        info = client.get_collection(collection)
        return {
            "exists": True,
            "count": client.count(collection, exact=True).count,
            "dimension": info.config.params.vectors.size,
        }
