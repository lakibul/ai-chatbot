"""BM25 lexical search with a code-preserving tokeniser.

Plan 2.4 -- rank_bm25 rather than Postgres FTS: pure Python, no server, and
decisively, Postgres has no Bengali text-search configuration, so it would
tokenise the Bangla document as `simple` anyway. Rolling our own tokeniser
gives full control over the thing that actually matters here: keeping
`ND-1042` as a token.

Plan 4.3 -- standard tokenisers destroy identifiers: ND-1042 -> nd + 1042,
SEC-EXC -> sec + exc. We emit the whole code AND its parts, so a query for
either form matches.

ONE DEVIATION FROM THE PLAN, deliberate: plan 4.3 proposes the pattern
`[A-Za-z]{2,}-[A-Za-z0-9]{2,}`. That requires two or more letters before the
hyphen, which would MISS `X-Nodi-Key` -- the exact string Q18 asks for, and
`X-Request-Id` which Q19 needs. The pattern below allows a single leading
letter and any number of hyphenated parts.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from .. import config
from ..ingest.models import Chunk
from ..ingest.normalise import fold_for_lexical
from .types import Hit

# ND-1042, SEC-EXC, X-Nodi-Key, X-Request-Id, X-Nodi-Signature, X-Api-Key.
CODE_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")

# Word characters including Bangla; \w under re.UNICODE already covers it.
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

# A bare identifier a query might use without the hyphen ("ND 1042", "ND1042").
BARE_CODE_PATTERN = re.compile(r"\b([A-Za-z]{2,4})[\s-]?(\d{3,6})\b")


def extract_codes(text: str) -> list[str]:
    """Identifier-shaped strings in `text`, lowercased.

    Used by the exact-code boost (plan 4.3) as well as by the tokeniser.
    """
    folded = fold_for_lexical(text)
    codes = {m.group(0) for m in CODE_PATTERN.finditer(folded)}
    # "what does ND 1042 mean" and "error ND1042" should both reach nd-1042.
    for match in BARE_CODE_PATTERN.finditer(folded):
        codes.add(f"{match.group(1)}-{match.group(2)}")
    return sorted(codes)


def tokenize(text: str) -> list[str]:
    """Tokens for BM25: words, plus whole codes AND their parts."""
    folded = fold_for_lexical(text)
    tokens: list[str] = []

    for match in CODE_PATTERN.finditer(folded):
        code = match.group(0)
        tokens.append(code)                       # "nd-1042"
        tokens.extend(p for p in code.split("-") if p)   # "nd", "1042"

    tokens.extend(WORD_PATTERN.findall(folded))
    for match in BARE_CODE_PATTERN.finditer(folded):
        tokens.append(f"{match.group(1)}-{match.group(2)}")
    return tokens


class BM25Index:
    """BM25 over chunk embed_text, persisted as a pickle."""

    def __init__(self, chunk_ids: list[str], payloads: list[dict[str, Any]],
                 corpus_tokens: list[list[str]]):
        if not chunk_ids:
            raise ValueError("refusing to build an empty BM25 index")
        self.chunk_ids = chunk_ids
        self.payloads = payloads
        self.corpus_tokens = corpus_tokens
        self.bm25 = BM25Okapi(corpus_tokens)

    # ------------------------------------------------------------- build --
    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        payloads = [c.to_dict() for c in chunks]
        tokens = [tokenize(c.embed_text) for c in chunks]
        return cls([c.chunk_id for c in chunks], payloads, tokens)

    # -------------------------------------------------------- persistence --
    def save(self, path: Path | None = None) -> Path:
        path = path or config.BM25_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        # BM25Okapi is cheap to rebuild, so only the inputs are persisted.
        # That keeps the pickle insensitive to a rank_bm25 version bump.
        with path.open("wb") as handle:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "payloads": self.payloads,
                "corpus_tokens": self.corpus_tokens,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Index":
        path = path or config.BM25_PATH
        with path.open("rb") as handle:
            data = pickle.load(handle)
        return cls(data["chunk_ids"], data["payloads"], data["corpus_tokens"])

    # ------------------------------------------------------------ search --
    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or config.BM25_TOP_K
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[Hit] = []
        for rank, index in enumerate(order[:top_k]):
            # BM25 legitimately returns 0.0 for a query with no lexical overlap
            # at all -- which is exactly Q34, an English question against a
            # Bangla-only document. Those are dropped rather than ranked, so
            # RRF does not reward an arbitrary tie-break order.
            if scores[index] <= 0:
                break
            hits.append(Hit(chunk_id=self.chunk_ids[index],
                            score=float(scores[index]),
                            payload=self.payloads[index],
                            stage="bm25",
                            debug={"bm25_rank": rank + 1}))
        return hits

    def __len__(self) -> int:
        return len(self.chunk_ids)


def build_bm25_index(chunks: list[Chunk], path: Path | None = None) -> BM25Index:
    index = BM25Index.build(chunks)
    index.save(path)
    return index
