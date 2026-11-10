"""Build the markdown evaluation report. Plan M6.

Reads the newest answers-*.csv produced by eval_answers.py, plus the floor
calibration if present, and writes eval/results/REPORT.md.

    python -m eval.report
    python -m eval.report --answers eval/results/answers-20261110-180000.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config                            # noqa: E402

TARGETS = {
    "retrieval_hit_rate": (0.90, "36 in-scope questions"),
    "answer_correctness": (0.85, "all 40 questions"),
}


def newest_answers() -> Path | None:
    candidates = sorted(config.EVAL_RESULTS_DIR.glob("answers-*.csv"))
    return candidates[-1] if candidates else None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build(answers_path: Path, retrieval: dict | None = None) -> str:
    rows = load_rows(answers_path)
    total = len(rows)
    correct = [r for r in rows if r["verdict"] == "CORRECT"]
    wrong = [r for r in rows if r["verdict"] == "INCORRECT"]
    errors = [r for r in rows if r["verdict"] == "ERROR"]

    out: list[str] = []
    out.append("# Evaluation report — Nodi Tech RAG Chatbot\n")
    out.append(f"Generated {datetime.now().isoformat(timespec='seconds')} · "
               f"source `{answers_path.name}`\n")

    # Plan M6: an ERROR count above zero INVALIDATES the run, and the report
    # must say so at the TOP rather than quietly publishing a depressed score.
    if errors:
        out.append(f"> ## RUN INVALID\n>\n> {len(errors)} question(s) failed with a "
                   f"**technical error**, not a wrong answer. Targets are measured "
                   f"only against runs with zero errors. Fix the infrastructure and "
                   f"re-run before reading the numbers below.\n")
    else:
        out.append("> **Run valid** — zero technical errors.\n")

    out.append("## Headline\n")
    rate = len(correct) / total if total else 0.0
    headline = [["Answer correctness", f"{len(correct)}/{total}", f"{rate:.0%}",
                 "85%", "PASS" if rate >= 0.85 else "FAIL"]]
    if retrieval:
        hit = retrieval.get("hit_rate")
        if hit is not None:
            headline.insert(0, [
                "Retrieval hit rate @5", retrieval.get("hits", ""),
                f"{hit:.0%}", "90%", "PASS" if hit >= 0.90 else "FAIL"])
    headline.append(["Technical errors", str(len(errors)), "", "0",
                     "PASS" if not errors else "FAIL"])
    out.append(table(["Metric", "Count", "Rate", "Target", "Result"], headline))
    out.append("")

    out.append("## By question type\n")
    by_type: "OrderedDict[str, list[dict]]" = OrderedDict()
    for row in rows:
        by_type.setdefault(row["type"], []).append(row)
    type_rows = []
    for qtype, group in by_type.items():
        ok = sum(1 for r in group if r["verdict"] == "CORRECT")
        bad = sum(1 for r in group if r["verdict"] == "INCORRECT")
        err = sum(1 for r in group if r["verdict"] == "ERROR")
        type_rows.append([qtype, ok, bad, err, len(group), f"{ok / len(group):.0%}"])
    out.append(table(["Type", "Correct", "Incorrect", "Error", "n", "Rate"], type_rows))
    out.append("")

    out.append("## The six traps\n")
    trap_rows = []
    trap_map = {
        "Superseded policy (Q29, Q30)": ["Q29", "Q30"],
        "Prompt injection (Q38, Q39)": ["Q38", "Q39"],
        "Exact codes (Q15-Q19)": ["Q15", "Q16", "Q17", "Q18", "Q19"],
        "Tables incl. PDF (Q20-Q24)": ["Q20", "Q21", "Q22", "Q23", "Q24"],
        "Bangla + cross-language (Q32-Q34)": ["Q32", "Q33", "Q34"],
        "Out of scope (Q35-Q37, Q40)": ["Q35", "Q36", "Q37", "Q40"],
    }
    index = {r["id"]: r for r in rows}
    for label, ids in trap_map.items():
        present = [index[i] for i in ids if i in index]
        if not present:
            continue
        ok = sum(1 for r in present if r["verdict"] == "CORRECT")
        trap_rows.append([label, f"{ok}/{len(present)}",
                          "PASS" if ok == len(present) else "see below"])
    out.append(table(["Trap", "Correct", "Result"], trap_rows))
    out.append("")

    if wrong:
        out.append(f"## Incorrect answers ({len(wrong)})\n")
        for row in wrong:
            out.append(f"**{row['id']}** ({row['type']}) — {row['question']}\n")
            out.append(f"- expected: {row['expected_answer']}")
            out.append(f"- got: {row['actual_answer'][:300]}")
            out.append(f"- reason: {row['reason']}\n")

    if errors:
        out.append(f"## Technical errors ({len(errors)})\n")
        for row in errors:
            out.append(f"- **{row['id']}**: `{row['error_code']}` — {row['reason']}")
        out.append("")

    times = [float(r["seconds"]) for r in rows if r["seconds"]]
    if times:
        out.append("## Latency\n")
        out.append(table(["Metric", "Seconds"], [
            ["mean", f"{sum(times) / len(times):.1f}"],
            ["median", f"{sorted(times)[len(times) // 2]:.1f}"],
            ["max", f"{max(times):.1f}"],
            ["total (minutes)", f"{sum(times) / 60:.1f}"],
        ]))
        out.append("")

    floor_path = config.EVAL_RESULTS_DIR / "floor_calibration.json"
    if floor_path.exists():
        data = json.loads(floor_path.read_text(encoding="utf-8"))
        out.append("## Relevance floor calibration\n")
        out.append(f"- lowest in-scope top-1 reranker score: "
                   f"**{data['lowest_in_scope']:.4f}**")
        out.append(f"- highest floor keeping every in-scope question: "
                   f"**{data['safe_floor']:.2f}**")
        out.append(f"- configured `RELEVANCE_FLOOR`: "
                   f"**{config.RELEVANCE_FLOOR}** (recommended "
                   f"{data['recommended']})\n")
        out_rows = [[r["id"], r["type"], f"{r['top_score']:.4f}", r["top_doc"]]
                    for r in data["rows"] if not r["in_scope"]]
        out.append("Out-of-scope questions and their top-1 scores:\n")
        out.append(table(["id", "type", "top score", "top document"], out_rows))
        out.append("")

    out.append("## Audit\n")
    out.append(f"Every verdict, including the judge's reasoning, is in "
               f"`{answers_path.relative_to(ROOT).as_posix()}`. A 4B judge is not "
               f"trustworthy unaudited (plan R7), so hard assertions on the exact "
               f"numbers (18, 6000, 1000, 25000, ND-1042, X-Nodi-Key) run first "
               f"and override it.\n")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.report")
    parser.add_argument("--answers", default="")
    parser.add_argument("--hit-rate", type=float, default=None,
                        help="retrieval hit rate from eval_retrieval, e.g. 1.0")
    parser.add_argument("--hits", default="", help="e.g. 36/36")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    path = Path(args.answers) if args.answers else newest_answers()
    if path is None or not path.exists():
        print("no answers-*.csv found; run python -m eval.eval_answers first",
              file=sys.stderr)
        return 1

    retrieval = None
    if args.hit_rate is not None:
        retrieval = {"hit_rate": args.hit_rate, "hits": args.hits}

    report = build(path, retrieval)
    out_path = Path(args.out) if args.out else config.EVAL_RESULTS_DIR / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
