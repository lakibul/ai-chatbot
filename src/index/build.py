"""CLI: build the search indexes from storage/chunks.json.

    python -m src.index            # vector index with the default embedder
    python -m src.index --model nomic-embed-text --collection nodi_nomic
"""
from __future__ import annotations

import argparse
import sys
import time

from .. import config
from ..ingest.build import build_chunks, save_chunks
from .lexical import build_bm25_index
from .vector import build_vector_index, collection_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.index")
    parser.add_argument("--model", default=config.EMBED_MODEL,
                        help="embedding model tag (default: %(default)s)")
    parser.add_argument("--collection", default=config.COLLECTION_NAME,
                        help="Qdrant collection name (default: %(default)s)")
    parser.add_argument("--reuse-chunks", action="store_true",
                        help="use storage/chunks.json instead of re-parsing the corpus")
    args = parser.parse_args(argv)

    try:
        if args.reuse_chunks:
            from ..ingest.build import load_chunks
            chunks = load_chunks()
        else:
            chunks = build_chunks()
            save_chunks(chunks)
    except Exception as exc:                      # noqa: BLE001 - fail loudly
        print(f"INGEST FAILED: {exc}", file=sys.stderr)
        return 1

    # BM25 first: it is instant, and if it fails there is no point spending
    # 45 s on embeddings.
    try:
        bm25 = build_bm25_index(chunks)
    except Exception as exc:                      # noqa: BLE001
        print(f"BM25 FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(bm25)} documents in the BM25 index -> {config.BM25_PATH}")

    print(f"Embedding {len(chunks)} chunks with {args.model} ...")
    started = time.time()
    try:
        count = build_vector_index(chunks, model=args.model, collection=args.collection)
    except Exception as exc:                      # noqa: BLE001
        print(f"INDEX FAILED: {exc}", file=sys.stderr)
        return 1
    elapsed = time.time() - started

    info = collection_info(args.collection)
    print(f"OK: {count} vectors in '{args.collection}' "
          f"(dim {info['dimension']}) in {elapsed:.1f}s "
          f"-- {elapsed / max(len(chunks), 1):.2f}s per chunk")

    if count != len(chunks):
        print(f"MISMATCH: {len(chunks)} chunks but {count} vectors", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
