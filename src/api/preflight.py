"""Startup checks. Plan M4.5: fail fast and loudly, before serving traffic.

Each check returns the exact command that fixes it. A server that starts
without its index would answer every question with a refusal, which looks
like a tuning problem and costs hours.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..chat.llm import LLMUnavailable, list_models


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""

    def render(self) -> str:
        mark = "OK  " if self.ok else "FAIL"
        line = f"[{mark}] {self.name}: {self.detail}"
        if not self.ok and self.fix:
            line += f"\n         fix: {self.fix}"
        return line


def _normalise(tag: str) -> str:
    """Ollama reports 'bge-m3:latest' for a pull of 'bge-m3'."""
    return tag[:-7] if tag.endswith(":latest") else tag


def check_ollama() -> tuple[Check, list[str]]:
    try:
        tags = list_models()
    except LLMUnavailable as exc:
        return Check("Ollama reachable", False, str(exc),
                     "ollama serve"), []
    return Check("Ollama reachable", True,
                 f"{config.OLLAMA_HOST}, {len(tags)} model(s)"), tags


def check_models(tags: list[str]) -> list[Check]:
    available = {_normalise(t) for t in tags} | set(tags)
    checks: list[Check] = []
    for required in config.REQUIRED_MODELS:
        present = required in available or _normalise(required) in available
        checks.append(Check(f"model {required}", present,
                            "present" if present else "missing",
                            f"ollama pull {required}"))
    return checks


def check_indexes() -> list[Check]:
    checks: list[Check] = []

    chunks_ok = config.CHUNKS_PATH.exists()
    checks.append(Check("chunks.json", chunks_ok,
                        str(config.CHUNKS_PATH) if chunks_ok else "missing",
                        "python -m src.ingest"))

    bm25_ok = config.BM25_PATH.exists()
    checks.append(Check("bm25.pkl", bm25_ok,
                        str(config.BM25_PATH) if bm25_ok else "missing",
                        "python -m src.index"))

    qdrant_ok = config.QDRANT_PATH.exists()
    checks.append(Check("qdrant store", qdrant_ok,
                        str(config.QDRANT_PATH) if qdrant_ok else "missing",
                        "python -m src.index"))

    # The two indexes must agree on how many chunks exist. A mismatch means
    # one was rebuilt and the other was not, which silently corrupts fusion.
    if chunks_ok and bm25_ok:
        try:
            import json

            from ..index.lexical import BM25Index
            chunk_count = len(json.loads(
                config.CHUNKS_PATH.read_text(encoding="utf-8")))
            bm25_count = len(BM25Index.load())
            agree = chunk_count == bm25_count
            checks.append(Check("index agreement", agree,
                                f"{chunk_count} chunks, {bm25_count} BM25 documents",
                                "python -m src.index"))
        except Exception as exc:                  # noqa: BLE001
            checks.append(Check("index agreement", False, f"unreadable: {exc}",
                                "python -m src.index"))
    return checks


def run_preflight(require_reranker: bool = False) -> list[Check]:
    ollama_check, tags = check_ollama()
    checks = [ollama_check]
    if ollama_check.ok:
        checks.extend(check_models(tags))
    checks.extend(check_indexes())

    if require_reranker:
        try:
            from ..retrieve.rerank import load_reranker
            load_reranker()
            checks.append(Check("reranker", True, config.RERANKER_MODEL))
        except Exception as exc:                  # noqa: BLE001
            checks.append(Check("reranker", False, str(exc),
                                "it downloads on first use; check the network"))
    return checks


def assert_ready(checks: list[Check]) -> None:
    failed = [c for c in checks if not c.ok]
    if failed:
        raise SystemExit(
            "Preflight failed. Refusing to start:\n"
            + "\n".join(c.render() for c in failed))
