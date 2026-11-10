"""Relevance-floor calibration sweep. Plan 4.6 / M5.

"A minimum top-1 reranker score, CALIBRATED AGAINST THE EVAL SET (sweep the
threshold, plot in-scope recall against out-of-scope refusal, pick the knee).
This number is measured, never guessed."

Retrieval only -- no LLM -- so the whole sweep costs well under a minute.

    python -m eval.calibrate_floor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_retrieval import load_questions     # noqa: E402
from src import config                             # noqa: E402
from src.retrieve.pipeline import Retriever        # noqa: E402

# Q37 is in the refusal set but is NOT off-domain: the pricing table really
# does discuss channels. Its score is reported separately because a floor
# tuned to refuse it would also refuse genuine questions.
NOT_MENTIONED = "Q37"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.calibrate_floor")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    questions = load_questions()
    rows: list[dict] = []

    with Retriever() as retriever:
        for question in questions:
            result = retriever.retrieve(question.question)
            rows.append({
                "id": question.qid,
                "type": question.qtype,
                "question": question.question,
                "in_scope": question.in_scope,
                "top_score": round(result.top_score, 4),
                "top_doc": result.hits[0].doc_id if result.hits else "",
            })
            print(f"  {question.qid} {'IN ' if question.in_scope else 'OUT'} "
                  f"{result.top_score:.4f}  {question.question[:50]}", flush=True)

    in_scope = [r for r in rows if r["in_scope"]]
    out_scope = [r for r in rows if not r["in_scope"]]

    print("\n=== top-1 reranker score distribution ===")
    lo = min(r["top_score"] for r in in_scope)
    print(f"in-scope  ({len(in_scope):2d}): min {lo:.4f}  "
          f"max {max(r['top_score'] for r in in_scope):.4f}")
    print(f"out-scope ({len(out_scope):2d}): "
          + ", ".join(f"{r['id']}={r['top_score']:.4f}" for r in out_scope))

    print("\n=== threshold sweep ===")
    print(f"{'floor':>8}{'in-scope kept':>15}{'out refused':>13}   note")
    best = None
    for step in range(0, 100, 2):
        floor = step / 100
        kept = sum(1 for r in in_scope if r["top_score"] >= floor)
        refused = sum(1 for r in out_scope if r["top_score"] < floor)
        note = ""
        if kept == len(in_scope) and best is None and refused > 0:
            best = floor
        if kept < len(in_scope):
            note = "LOSES in-scope questions"
        print(f"{floor:>8.2f}{kept:>9}/{len(in_scope):<5}{refused:>8}/{len(out_scope):<4}   {note}")
        if kept < len(in_scope) - 2:
            break

    # The knee: the highest floor that still keeps EVERY in-scope question.
    # Anything higher trades a real answer for a refusal, which is a bad deal
    # when the reranker already separates the two populations cleanly.
    safe = max((s / 100 for s in range(0, 100)
                if all(r["top_score"] >= s / 100 for r in in_scope)), default=0.0)
    margin = round(safe * 0.8, 4)

    print(f"\nLowest in-scope score : {lo:.4f}")
    print(f"Highest floor keeping all in-scope: {safe:.2f}")
    print(f"Recommended RELEVANCE_FLOOR (80% of that, for headroom): {margin}")
    refused_at = [r["id"] for r in out_scope if r["top_score"] < margin]
    print(f"At that floor, these refuse on score alone: {refused_at or 'none'}")
    print(f"The rest must be refused by the prompt rules (plan 4.6), "
          f"notably {NOT_MENTIONED}.")

    path = Path(args.out) if args.out else config.EVAL_RESULTS_DIR / "floor_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"rows": rows, "lowest_in_scope": lo, "safe_floor": safe,
         "recommended": margin}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
