"""Hardware benchmark. Plan M2: "replaces section 5's estimates with facts."

Measures, on a realistic ~1,800-token RAG prompt:
  * prefill (prompt-reading) speed -- plan 5 calls this THE bottleneck
  * generation speed in tokens/second
  * num_thread 4 vs 8 -- "SMT typically hurts llama.cpp on Zen+; measure,
    do not guess" (plan 2.1)
  * whether num_gpu:0 really keeps the GT 1030 idle (plan R6)

    python bench.py                      # default model, threads 4 and 8
    python bench.py --model qwen3:1.7b-q4_K_M --threads 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config                      # noqa: E402
from src.chat.llm import LLMError, chat     # noqa: E402
from src.ingest.build import load_chunks    # noqa: E402

SYSTEM = (
    "You are a helpful assistant for Nodi Tech Ltd. Answer only from the "
    "documents provided. Cite the document you used. If the documents do not "
    "contain the answer, say so plainly."
)
QUESTION = "How many days of annual leave do employees get, and how many can be carried forward?"


def build_prompt(target_chunks: int = 5) -> list[dict[str, str]]:
    """A prompt shaped like the real thing: system rules + fenced chunks."""
    try:
        chunks = load_chunks()
    except FileNotFoundError:
        print("storage/chunks.json missing -- run: python -m src.ingest", file=sys.stderr)
        raise SystemExit(1)

    picked = chunks[:target_chunks]
    blocks = "\n\n".join(
        f'<document id="{c.doc_id}" section="{c.section}">\n{c.embed_text}\n</document>'
        for c in picked
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{blocks}\n\nQuestion: {QUESTION}"},
    ]


def gpu_memory_used() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True)
        return out.stdout.strip()
    except Exception:                       # noqa: BLE001 - nvidia-smi is optional
        return "unavailable"


def gpu_processes() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True)
        return out.stdout.strip() or "(no compute processes)"
    except Exception:                       # noqa: BLE001
        return "unavailable"


def run_once(messages: list[dict[str, str]], model: str, threads: int,
             num_predict: int) -> dict:
    started = time.time()
    result = chat(messages, model=model, num_predict=num_predict,
                  options={"num_thread": threads})
    wall = time.time() - started
    return {
        "threads": threads,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "load_s": result.load_seconds,
        "prefill_s": result.prefill_seconds,
        "gen_s": result.generation_seconds,
        "wall_s": wall,
        "prefill_tps": result.prefill_tokens_per_second,
        "gen_tps": result.output_tokens_per_second,
        "text": result.text,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python bench.py")
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--threads", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--num-predict", type=int, default=120)
    parser.add_argument("--chunks", type=int, default=5)
    args = parser.parse_args(argv)

    messages = build_prompt(args.chunks)
    approx_chars = sum(len(m["content"]) for m in messages)

    print(f"Model      : {args.model}")
    print(f"Prompt     : {approx_chars} chars across {args.chunks} chunks")
    print(f"Options    : num_gpu={config.NUM_GPU} num_ctx={config.NUM_CTX} "
          f"temperature={config.TEMPERATURE}")
    print(f"GPU before : {gpu_memory_used()}")
    print()

    rows: list[dict] = []
    for threads in args.threads:
        print(f"--- num_thread={threads} (warm-up + measured run) ---")
        try:
            run_once(messages, args.model, threads, 8)       # warm up / load
            row = run_once(messages, args.model, threads, args.num_predict)
        except LLMError as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            return 1
        rows.append(row)
        print(f"  prompt {row['prompt_tokens']} tok -> prefill {row['prefill_s']:.1f}s "
              f"({row['prefill_tps']:.1f} tok/s)")
        print(f"  output {row['output_tokens']} tok -> generate {row['gen_s']:.1f}s "
              f"({row['gen_tps']:.2f} tok/s)")
        print(f"  wall {row['wall_s']:.1f}s")
        print(f"  GPU during/after: {gpu_memory_used()} | {gpu_processes()}")
        print()

    print("=" * 68)
    print(f"{'threads':>8}{'prompt':>9}{'prefill s':>11}{'prefill t/s':>13}"
          f"{'gen t/s':>10}{'wall s':>9}")
    for row in rows:
        print(f"{row['threads']:>8}{row['prompt_tokens']:>9}{row['prefill_s']:>11.1f}"
              f"{row['prefill_tps']:>13.1f}{row['gen_tps']:>10.2f}{row['wall_s']:>9.1f}")
    print("=" * 68)

    if len(rows) > 1:
        best = max(rows, key=lambda r: r["gen_tps"])
        print(f"\nFastest generation: num_thread={best['threads']} "
              f"({best['gen_tps']:.2f} tok/s). Set NUM_THREAD={best['threads']} in .env")

    print("\nSample output:")
    print(rows[-1]["text"][:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
