"""The FAST evaluation harness -- no LLM at all.

Plan 5.1: the full answer eval costs ~45 minutes, which kills the iteration
loop the brief asks for ("after every change to chunking, embeddings or
prompts, rerun the evaluation"). This harness runs in seconds and covers
retrieval, which is where most of the traps live.

    python -m eval.eval_retrieval                    # default retriever
    python -m eval.eval_retrieval --mode vector      # vector only
    python -m eval.eval_retrieval --model nomic-embed-text --collection nodi_nomic

Reports hit rate BROKEN OUT BY QUESTION TYPE, because the averaged number
hides exactly the failures we care about (plan M3).
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

# Q35/Q36/Q37/Q40 have no source document -- they are the refusal cases and
# belong to the answer eval, not the retrieval eval.
OUT_OF_SCOPE_MARKER = "none"


@dataclass
class Question:
    qid: str
    qtype: str
    question: str
    expected_answer: str
    sources: list[str]

    @property
    def in_scope(self) -> bool:
        return bool(self.sources)


def load_questions(path: Path | None = None) -> list[Question]:
    path = path or config.EVAL_QUESTIONS
    rows = list(csv.DictReader(io.open(path, encoding="utf-8-sig")))
    questions: list[Question] = []
    for row in rows:
        raw = (row.get("source_documents") or "").strip()
        sources = [] if raw.lower() == OUT_OF_SCOPE_MARKER else [
            s.strip() for s in raw.split(";") if s.strip()]
        questions.append(Question(
            qid=row["id"].strip(),
            qtype=row["type"].strip(),
            question=row["question"].strip(),
            expected_answer=(row.get("expected_answer") or "").strip(),
            sources=sources,
        ))
    return questions


@dataclass
class Result:
    question: Question
    hit: bool
    rank: int | None          # 1-based rank of the first correct document
    retrieved: list[str]      # doc_id per retrieved chunk, in order


def evaluate(retriever, questions: list[Question], top_k: int) -> list[Result]:
    results: list[Result] = []
    for question in questions:
        if not question.in_scope:
            continue
        hits = retriever(question.question, top_k)
        docs = [h.doc_id for h in hits]
        rank = None
        for position, doc in enumerate(docs[:top_k], start=1):
            if doc in question.sources:
                rank = position
                break
        results.append(Result(question, rank is not None, rank, docs))
    return results


def report(results: list[Result], top_k: int, label: str) -> float:
    hits = sum(1 for r in results if r.hit)
    total = len(results)
    rate = hits / total if total else 0.0

    by_type: "OrderedDict[str, list[Result]]" = OrderedDict()
    for result in results:
        by_type.setdefault(result.question.qtype, []).append(result)

    print(f"\n=== Retrieval hit rate @{top_k} -- {label} ===")
    print(f"{'type':<18}{'hit':>5}{'/':^3}{'n':<5}{'rate':>8}")
    print("-" * 42)
    for qtype, group in by_type.items():
        got = sum(1 for r in group if r.hit)
        print(f"{qtype:<18}{got:>5}{'/':^3}{len(group):<5}{got / len(group):>7.0%}")
    print("-" * 42)
    print(f"{'OVERALL':<18}{hits:>5}{'/':^3}{total:<5}{rate:>7.0%}")

    misses = [r for r in results if not r.hit]
    if misses:
        print(f"\nMISSES ({len(misses)}):")
        for result in misses:
            print(f"  {result.question.qid} [{result.question.qtype}] "
                  f"{result.question.question[:58]}")
            print(f"      want one of: {result.question.sources}")
            print(f"      got        : {result.retrieved[:top_k]}")

    # Hit rate @5 over a 12-document corpus is a LOOSE metric -- top-5 covers
    # nearly half the corpus. Rank-1 is the number that actually discriminates
    # between retrieval strategies here, so it is reported alongside.
    ranks = [r.rank for r in results if r.rank]
    if ranks:
        print(f"\nMean rank of first correct doc: {sum(ranks) / len(ranks):.2f} "
              f"| rank-1: {sum(1 for r in ranks if r == 1)}/{len(results)}")
    not_first = [r for r in results if r.hit and r.rank != 1]
    if not_first:
        print(f"\nNOT AT RANK 1 ({len(not_first)}):")
        for result in not_first:
            print(f"  {result.question.qid} [{result.question.qtype}] rank={result.rank}"
                  f"  {result.question.question[:52]}")
            print(f"      ahead of it: {result.retrieved[:result.rank - 1]}")
    return rate


def build_retriever(mode: str, model: str, collection: str):
    if mode == "vector":
        from qdrant_client import QdrantClient

        from src.index.vector import search

        # Own the client outright for the whole run. Going through the
        # open_store context manager and calling __enter__ by hand lets the
        # generator be collected, which closes the client under us.
        client = QdrantClient(path=str(config.QDRANT_PATH))

        def retrieve(query: str, top_k: int):
            return search(query, top_k=top_k, model=model,
                          collection=collection, client=client)
        return retrieve

    if mode == "hybrid":
        from src.retrieve.pipeline import build_hybrid_retriever
        return build_hybrid_retriever(model=model, collection=collection)

    raise SystemExit(f"unknown mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.eval_retrieval")
    parser.add_argument("--mode", default="hybrid", choices=("vector", "hybrid"))
    parser.add_argument("--model", default=config.EMBED_MODEL)
    parser.add_argument("--collection", default=config.COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=config.FINAL_TOP_K)
    parser.add_argument("--target", type=float, default=0.90,
                        help="hit-rate gate (plan M3: >= 90%%)")
    args = parser.parse_args(argv)

    questions = load_questions()
    in_scope = [q for q in questions if q.in_scope]
    print(f"{len(questions)} questions, {len(in_scope)} in scope, "
          f"{len(questions) - len(in_scope)} refusal cases (answer eval only)")

    retriever = build_retriever(args.mode, args.model, args.collection)
    results = evaluate(retriever, questions, args.top_k)
    label = f"mode={args.mode} embed={args.model}"
    rate = report(results, args.top_k, label)

    hits = sum(1 for r in results if r.hit)
    print(f"\nGate: {hits}/{len(results)} >= {args.target:.0%} "
          f"({int(args.target * len(results) + 0.999)} needed) -> "
          f"{'PASS' if rate >= args.target else 'FAIL'}")
    return 0 if rate >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
