"""Build the data file behind the "Accuracy" page in the web UI.

    python -m eval.build_accuracy_data                  # rebuild from stored results
    python -m eval.build_accuracy_data --retrieval      # ALSO measure retrieval now
    python -m eval.build_accuracy_data --csv eval/results/answers-XXXX.csv

Output: src/api/static/accuracy-data.js  (window.ACCURACY_DATA = {...})

THE RULE FOR THIS FILE: every number on the page is read from something that was
measured and stored -- an answers CSV, floor_calibration.json, a retrieval
snapshot, src/config.py, or the live Ollama server. Nothing is typed in by hand
and nothing is estimated. If a source is missing the field is None and the page
says "not measured" instead of guessing.

--retrieval runs the retrieval eval against a COPY of storage/qdrant, so it works
while the web server is running (Qdrant local mode allows one process on the
real folder). It needs no LLM, only the embedder and reranker.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

OUT_JS = ROOT / "src" / "api" / "static" / "accuracy-data.js"
RESULTS = config.EVAL_RESULTS_DIR

# Files whose change can alter an answer. Used to say, with evidence, whether
# the code that produced the stored result is the code that is running now.
ANSWER_AFFECTING = (
    "src/chat/prompts.py", "src/chat/engine.py", "src/chat/memory.py",
    "src/chat/smalltalk.py", "src/chat/rewrite.py", "src/chat/llm.py",
    "src/config.py", "src/retrieve/pipeline.py", "src/retrieve/rrf.py",
    "src/retrieve/rerank.py", "src/retrieve/code_boost.py",
    "src/ingest/chunk.py", "src/ingest/sanitise.py", "src/index/embed.py",
)

# The "six traps" and which question ids belong to each (plan section 4).
TRAPS = OrderedDict([
    ("Old policy that contradicts the current one", ["Q29", "Q30"]),
    ("Hidden instruction attack inside a document", ["Q38", "Q39"]),
    ("Exact codes (ND-1042, SEC-EXC, X-Nodi-Key)", ["Q15", "Q16", "Q17", "Q18", "Q19"]),
    ("Tables, including one inside a PDF", ["Q20", "Q21", "Q22", "Q23", "Q24"]),
    ("Bangla and cross-language questions", ["Q32", "Q33", "Q34"]),
    ("Questions the documents cannot answer", ["Q35", "Q36", "Q37", "Q40"]),
])


def _git(*args: str) -> str | None:
    """Read-only git only (project rule: never write to history)."""
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def latest(pattern: str) -> Path | None:
    files = sorted(RESULTS.glob(pattern))
    return files[-1] if files else None


def stamp_of(path: Path) -> datetime | None:
    m = re.search(r"(\d{8}-\d{6})", path.name)
    return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S") if m else None


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(io.open(path, encoding="utf-8-sig")))


# ------------------------------------------------------------------ answers --
def answers_block(path: Path) -> dict:
    rows = read_csv(path)
    secs = [float(r["seconds"]) for r in rows if r.get("seconds")]
    by_type: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        t = by_type.setdefault(r["type"], {"correct": 0, "incorrect": 0, "error": 0, "n": 0})
        t["n"] += 1
        t[r["verdict"].lower()] += 1
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("CORRECT", "INCORRECT", "ERROR")}
    by_id = {r["id"]: r for r in rows}
    traps = []
    for name, ids in TRAPS.items():
        ok = sum(1 for i in ids if by_id.get(i, {}).get("verdict") == "CORRECT")
        traps.append({"name": name, "ids": ids, "correct": ok, "n": len(ids)})
    return {
        "file": path.name,
        "run_started": (stamp_of(path).isoformat() if stamp_of(path) else None),
        "n": len(rows),
        "counts": counts,
        "by_type": by_type,
        "traps": traps,
        "latency": {
            "mean": round(statistics.mean(secs), 1) if secs else None,
            "median": round(statistics.median(secs), 1) if secs else None,
            "max": round(max(secs), 1) if secs else None,
            "min": round(min(secs), 1) if secs else None,
            "total_minutes": round(sum(secs) / 60, 1) if secs else None,
        },
        "rows": [{
            "id": r["id"], "type": r["type"], "question": r["question"],
            "expected": r["expected_answer"], "actual": r["actual_answer"],
            "verdict": r["verdict"], "reason": r["reason"],
            "citations": [c.strip() for c in (r.get("citations") or "").split(";") if c.strip()],
            "seconds": float(r["seconds"]) if r.get("seconds") else None,
        } for r in rows],
    }


def run_history() -> list[dict]:
    """Every stored full run, oldest first, with what changed between runs."""
    runs, prev = [], None
    for path in sorted(RESULTS.glob("answers-*.csv")):
        rows = read_csv(path)
        verdicts = {r["id"]: r["verdict"] for r in rows}
        entry = {
            "file": path.name,
            "when": stamp_of(path).isoformat() if stamp_of(path) else None,
            "correct": sum(1 for v in verdicts.values() if v == "CORRECT"),
            "n": len(rows),
            "errors": sum(1 for v in verdicts.values() if v == "ERROR"),
            "wrong_ids": [i for i, v in verdicts.items() if v == "INCORRECT"],
            "changed": [],
        }
        if prev:
            entry["changed"] = [{"id": i, "from": prev[i], "to": v}
                                for i, v in verdicts.items() if i in prev and prev[i] != v]
        runs.append(entry)
        prev = verdicts
    return runs


# ---------------------------------------------------------------- provenance --
def code_status(answers_file: Path, evaluated_override: str = "") -> dict:
    """Is the code that made the stored result the code in the folder now?

    The run is started from a working tree and committed a little later, so the
    commit that matches it is the first one made within 15 minutes AFTER the run
    started; failing that, the last one before. --evaluated-commit overrides it.
    """
    ts = stamp_of(answers_file)
    evaluated = evaluated_override or None
    guessed = False
    if not evaluated and ts:
        log = _git("log", "--format=%h|%at") or ""
        start = ts.timestamp()
        commits = [(h, int(t)) for h, t in (ln.split("|") for ln in log.splitlines() if "|" in ln)]
        after = sorted((t, h) for h, t in commits if start <= t <= start + 15 * 60)
        before = sorted((t, h) for h, t in commits if t < start)
        evaluated = after[0][1] if after else (before[-1][1] if before else None)
        guessed = True
    head = _git("rev-parse", "--short", "HEAD")
    changed: list[str] = []
    if evaluated:
        diff = _git("diff", "--name-only", evaluated, "HEAD", "--", "src") or ""
        changed = [f for f in diff.splitlines() if f in ANSWER_AFFECTING]
    dirty = [ln[3:].strip() for ln in (_git("status", "--short", "--", "src") or "").splitlines()
             if ln[3:].strip().replace("\\", "/") in ANSWER_AFFECTING]
    return {"evaluated_commit": evaluated, "evaluated_commit_guessed": guessed,
            "head_commit": head, "changed_since": changed, "uncommitted": dirty}


def models_block() -> list[dict]:
    """Model facts straight from the running Ollama server (not typed in)."""
    tags: dict[str, dict] = {}
    try:
        import httpx
        data = httpx.get(config.OLLAMA_TAGS_URL, timeout=8).json()
        tags = {m["name"]: m for m in data.get("models", [])}
    except Exception:
        pass

    def find(name: str) -> dict | None:
        return tags.get(name) or tags.get(name + ":latest")

    roles = [
        ("Writes the answer", config.LLM_MODEL, "Ollama (CPU only)",
         "Apache-2.0"),
        ("Rewrites follow-up questions", config.REWRITE_MODEL, "Ollama (CPU only)",
         "Apache-2.0"),
        ("Turns text into numbers (search by meaning)", config.EMBED_MODEL, "Ollama (CPU only)",
         "MIT"),
        ("Re-ranks the shortlist", config.RERANKER_MODEL, "ONNX Runtime via FastEmbed (CPU)",
         "CC-BY-NC-4.0 (non-commercial only)"),
    ]
    out = []
    for role, name, runtime, licence in roles:
        t = find(name)
        d = (t or {}).get("details", {})
        out.append({
            "role": role, "name": name, "runtime": runtime, "licence": licence,
            "params": d.get("parameter_size"), "quant": d.get("quantization_level"),
            "family": d.get("family"),
            "context_max": d.get("context_length"),
            "size_gb": round(t["size"] / 1e9, 2) if t else None,
            "installed": bool(t) if tags else None,
        })
    return out


def settings_block() -> list[dict]:
    rows = [
        ("Answer-writing temperature", config.TEMPERATURE, "Low = repeatable, factual answers"),
        ("Context window used (num_ctx)", config.NUM_CTX, "Tokens the model can read per request"),
        ("CPU threads (num_thread)", config.NUM_THREAD, "Measured best on both machines"),
        ("GPU layers (num_gpu)", config.NUM_GPU, "0 = CPU only, by design"),
        ("Passages kept for the answer", config.FINAL_TOP_K, "After re-ranking"),
        ("Passages re-ranked", config.RERANK_CANDIDATES, "Shortlist size"),
        ("Search depth, each search", config.VECTOR_TOP_K, "Word search and meaning search"),
        ("Rank fusion constant (RRF k)", config.RRF_K, "Merges the two searches"),
        ("Relevance floor", config.RELEVANCE_FLOOR, "Below this score: say 'not in the documents'"),
        ("Old-policy score multiplier", config.SUPERSEDED_MULTIPLIER, "Applied after a 0-1 squash"),
        ("Conversation turns remembered", config.HISTORY_TURNS, "One turn = a question and its answer"),
        ("Max answer length (tokens)", config.ANSWER_MAX_TOKENS, ""),
        ("Passage size target (characters)", config.CHUNK_TARGET_CHARS, "Tables are never split"),
    ]
    return [{"name": n, "value": v, "note": note} for n, v, note in rows]


def corpus_block() -> dict:
    try:
        chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
        chunks = chunks if isinstance(chunks, list) else chunks.get("chunks", [])
        docs = sorted({c.get("doc_id") for c in chunks if c.get("doc_id")})
        return {"documents": len(docs), "passages": len(chunks)}
    except Exception:
        return {"documents": None, "passages": None}


# ----------------------------------------------------------------- retrieval --
def measure_retrieval() -> Path:
    """Run the real retrieval pipeline on all in-scope questions, no LLM."""
    tmp = Path(tempfile.mkdtemp(prefix="qdrant_copy_"))
    copy = tmp / "qdrant"
    shutil.copytree(config.QDRANT_PATH, copy, ignore=shutil.ignore_patterns(".lock"))
    config.QDRANT_PATH = copy        # Retriever reads this at construction
    from eval.eval_retrieval import load_questions
    from src.retrieve.pipeline import Retriever

    questions = load_questions()
    retriever = Retriever()
    rows, t0 = [], time.time()
    try:
        for q in questions:
            if not q.in_scope:
                continue
            res = retriever.retrieve(q.question)
            docs = [h.doc_id for h in res.hits][:config.FINAL_TOP_K]
            rank = next((i for i, d in enumerate(docs, 1) if d in q.sources), None)
            rows.append({"id": q.qid, "type": q.qtype, "question": q.question,
                         "want": q.sources, "got": docs, "rank": rank})
    finally:
        retriever.close()
        shutil.rmtree(tmp, ignore_errors=True)
    out = RESULTS / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "top_k": config.FINAL_TOP_K, "seconds": round(time.time() - t0, 1),
        "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def retrieval_block(path: Path | None) -> dict | None:
    if not path:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    by_type: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        t = by_type.setdefault(r["type"], {"hit": 0, "n": 0})
        t["n"] += 1
        t["hit"] += 1 if r["rank"] else 0
    ranks = [r["rank"] for r in rows if r["rank"]]
    return {
        "file": path.name, "measured_at": data.get("measured_at"), "top_k": data.get("top_k"),
        "n": len(rows), "hits": len(ranks),
        "rank1": sum(1 for r in ranks if r == 1),
        "mean_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "by_type": by_type,
        "not_rank1": [{"id": r["id"], "rank": r["rank"], "ahead": r["got"][:r["rank"] - 1]}
                      for r in rows if r["rank"] and r["rank"] != 1],
        "misses": [{"id": r["id"], "want": r["want"], "got": r["got"]} for r in rows if not r["rank"]],
    }


def floor_block() -> dict | None:
    path = RESULTS / "floor_calibration.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    ins = [r for r in rows if r["in_scope"]]
    outs = [r for r in rows if not r["in_scope"]]
    return {
        "lowest_in_scope": min(r["top_score"] for r in ins) if ins else None,
        "configured": config.RELEVANCE_FLOOR,
        "out_of_scope": [{"id": r["id"], "score": r["top_score"], "doc": r["top_doc"]} for r in outs],
    }


# ---------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.build_accuracy_data")
    ap.add_argument("--csv", default="", help="answers CSV to show (default: newest)")
    ap.add_argument("--evaluated-commit", default="",
                    help="commit the answers CSV was measured on (default: inferred from its time)")
    ap.add_argument("--retrieval", action="store_true",
                    help="measure retrieval now (uses a copy of the index)")
    args = ap.parse_args(argv)

    csv_path = Path(args.csv) if args.csv else latest("answers-*.csv")
    if not csv_path or not csv_path.exists():
        print("no answers-*.csv found: run python -m eval.eval_answers first", file=sys.stderr)
        return 2
    if args.retrieval:
        print("measuring retrieval against a copy of the index ...", flush=True)
        print("  wrote", measure_retrieval().name)

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "answers": answers_block(csv_path),
        "history": run_history(),
        "code": code_status(csv_path, args.evaluated_commit),
        "models": models_block(),
        "settings": settings_block(),
        "corpus": corpus_block(),
        "retrieval": retrieval_block(latest("retrieval-*.json")),
        "floor": floor_block(),
    }
    OUT_JS.write_text(
        "/* GENERATED by eval/build_accuracy_data.py -- do not edit by hand. */\n"
        "window.ACCURACY_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8")
    a = data["answers"]
    print(f"wrote {OUT_JS.relative_to(ROOT)}  ({a['counts']['CORRECT']}/{a['n']} correct from {a['file']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
