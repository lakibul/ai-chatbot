# CLAUDE.md — working notes for this repository

Read this before changing anything. It is the handoff document: what exists,
why it is built this way, what has already been measured, and the specific
mistakes that cost time here so they are not repeated.

**Project:** a fully local, citation-backed RAG chatbot over 12 internal
documents of a fictional company, Nodi Tech Ltd. No cloud LLM, no API keys.

**Status:** complete and evaluated. 39/40 answer correctness, 36/36 retrieval,
220 tests passing. See `eval/results/REPORT.md`.

---

## 1. Start here on a new machine

Full setup is in [`README.md`](README.md) → **Setup from a fresh clone**.
The short version, because nothing works until these are done:

```bash
ollama serve                     # must be running, with 3 model slots
python -m src.ingest             # parse corpus      -> storage/chunks.json
python -m src.index              # BM25 + vectors    -> storage/{bm25.pkl,qdrant}
python -m uvicorn src.api.app:app --port 8000
```

`storage/`, `.venv/`, `.env` and `eval/results/` are **gitignored**. A fresh
clone has no indexes, so steps 2 and 3 are mandatory, not optional. The server
refuses to start without them and prints the exact command to fix it.

---

## 2. Three things that will waste your time if you don't know them

**Qdrant local mode takes an exclusive file lock.** Exactly one process may
touch `storage/qdrant` at a time. If the server is running, `pytest` and every
`eval/*` script fail with `AlreadyLocked` and 14 errors in `test_retrieve.py`.
That is not a bug — stop the server first. Within one process, only one
`Retriever` may exist; tests share a module-scoped fixture for this reason.

**`OLLAMA_MAX_LOADED_MODELS` must be 3, not 2.** The pipeline needs the 4B
generator, `bge-m3` for embedding, and `qwen3:1.7b` for query rewriting
resident at once. With two slots, measured, every conversational turn evicted
and reloaded a model twice, adding 20-40 s per answer. The plan originally said
2; it was wrong.

**Python 3.10 or 3.11.** The `py` launcher on the original machine registers
only 3.14, for which `onnxruntime` / `fastembed` wheels are a hunt. Create the
venv from a 3.10 interpreter by full path.

On Windows also set `PYTHONIOENCODING=utf-8` when running anything that prints
Bangla, or `cp1252` raises `UnicodeEncodeError` on perfectly good output.

---

## 3. Architecture

```
INGEST   data/*.md ─ parse_md ─┐
         data/*.pdf ─ parse_pdf ┤─ sanitise ─ chunk ─ assert ─ chunks.json
                                │
INDEX    ├─ bge-m3 /api/embed ──→ Qdrant local (storage/qdrant)
         └─ custom tokeniser ───→ rank_bm25   (storage/bm25.pkl)

CHAT     question
           ├─ smalltalk?        → canned reply, no retrieval, no LLM
           ├─ injection?        → refusal, no retrieval, no LLM
           ├─ rewrite (1.7b, history-aware)
           ├─ BM25 top-20  +  vector top-20
           ├─ RRF fuse (k=60)
           ├─ exact-code boost                      ← BEFORE rerank
           ├─ candidate window (12, reserved seats)
           ├─ rerank (ONNX cross-encoder) → top 5
           ├─ superseded down-rank (sigmoid × 0.5)
           ├─ relevance floor 0.088                 ← AFTER rerank
           └─ Ollama /api/chat (stream) → SSE → browser
```

Two orderings are load-bearing: the **code boost sits before the reranker**
(a cross-encoder has no special respect for `ND-1042`), and the **relevance
floor sits after it** (only the calibrated score is trustworthy).

### Module map

| Path | Holds |
|---|---|
| `src/config.py` | **Every tunable.** Nothing is hard-coded elsewhere. |
| `src/ingest/` | `parse_md`, `parse_pdf`, `normalise`, `chunk`, `sanitise`, `build` |
| `src/index/` | `embed` (Ollama), `vector` (Qdrant), `lexical` (BM25), `types` |
| `src/retrieve/` | `rrf`, `code_boost`, `rerank`, `pipeline` |
| `src/chat/` | `llm`, `prompts`, `rewrite`, `smalltalk`, `memory`, `engine` |
| `src/api/` | `app` (FastAPI+SSE), `preflight`, `errors`, `static/index.html` |
| `eval/` | `eval_retrieval` (fast), `eval_answers` (full), `eval_conversations`, `calibrate_floor`, `report` |

---

## 4. The six traps and how each is handled

The dataset in `rag_chatbot_kit/data/` is deliberately booby-trapped. **It is
read-only — never edit it.** `rag_chatbot_kit.zip` is the pristine copy.

| Trap | Mechanism | Guarded by |
|---|---|---|
| **Superseded policy** — leave is 18, not the old 15 | Status parsed from filename *and* `STATUS:` line; warning embedded *into* the chunk text; post-rerank score halved. Demoted, never deleted, because Q29 must cite **both** documents | `test_superseded_*` |
| **Prompt injection** in `vendor_notice_printer.md` | Stripped at ingest, logged to `storage/sanitised.log`; documents fenced as untrusted; output asserted. The file **stays indexed** so Q39 still answers | `test_injection_*` |
| **Exact codes** `ND-1042`, `SEC-EXC`, `X-Nodi-Key` | Tokeniser emits `nd-1042` *and* `nd` *and* `1042`; literal match gets a large additive boost **before** reranking and a guaranteed seat in the window | `test_nd1042_rank1` |
| **Tables** incl. one inside a PDF | Never split. A chunk holds a whole table or none of it, always with its heading | `test_no_split_tables` |
| **Bangla + cross-language** | `bge-m3` carries the English→Bangla question; Bengali digits folded to ASCII **in the BM25 index only** | `test_bangla_preserved` |
| **Out of scope** — four different refusals | Relevance floor (measured) + explicit prompt rules; absence is never inferred as denial | `test_guardrails.py` |

---

## 5. Measured facts — do not re-derive these

Hardware: Ryzen 5 3400G (4c/8t, Zen+), 16 GB DDR4-2666, GT 1030 2 GB
(**unusable** — a 4B Q4 needs ~2.5 GB, so `num_gpu: 0` is forced; `nvidia-smi`
confirms Ollama never appears as a GPU process).

| | Measured |
|---|---|
| Prefill, 4B, cold | **23.9 tok/s** (1,162 tok → 48.6 s) |
| Generation, 4B | **5.8 tok/s** |
| Prefill, 1.7b, cold | 56.3 tok/s · generation 10.3 tok/s |
| Bare question, no documents | prefill **0.8 s** |
| Repeat question (prefix cached) | prefill **0.2 s**, total ~9 s |
| Rerank 20 candidates | 8.6 s → **cut to 12 candidates** (~5 s) |
| BM25 / vector+embed | ~0 ms / 205 ms |
| `num_thread` 4 vs 8 | **8 wins**: 48.6 s vs 60.5 s prefill. SMT helps here |
| Index build | ~45 s (50 chunks) |

**Latency is RAG, not the model.** A bare question prefills in 0.8 s; five
retrieved documents add 15-25 s of reading before the first token. A 2.4×
smaller model is only ~2× faster. The real fix is a GPU with ≥4 GB VRAM —
prefill is the embarrassingly-parallel phase GPUs exist for.

The system prompt (~740 tok) is byte-identical every turn and llama.cpp
caches it. Documents (~390 tok) are new each question — that is what you pay
for. **Never interpolate anything into `SYSTEM_PROMPT`.**

### Eval runtimes

| Command | Time |
|---|---|
| `pytest` | ~75 s |
| `pytest -m slow` | ~4 min (real LLM calls) |
| `eval_retrieval` | ~6-9 min (reranker dominates) |
| `eval_answers` | ~30 min |
| `eval_conversations` | ~8 min |

---

## 6. Deviations from `plan.md` — all deliberate, all measured

`plan.md` §1-10 is the approved plan; **§11 is the build log and supersedes it
wherever they disagree.**

1. **Plan's identifier regex would miss `X-Nodi-Key`.** It required two letters
   before the hyphen. The tokeniser allows one.
2. **Plan's `×0.5` superseded multiplier would have *promoted* superseded
   chunks.** Cross-encoder outputs are logits and often negative (measured
   −0.31 and −3.04); ×0.5 makes a negative *larger*. Scores pass through a
   monotonic sigmoid first.
3. **RRF buried the cross-language question.** The plan assumed BM25 would
   "score zero" on an English question against a Bangla document. It scores
   *noise*, and RRF rewards appearing in both lists over ranking first in one,
   so the right document fell from vector rank 1 to fused rank 13. The rerank
   window now **reserves seats** for each retriever's own top results.
4. **`qwen3:1.7b` is a thinking model.** Without `"think": false` it spent all
   40 rewrite tokens reasoning and returned empty content, so the rewriter
   silently fell back to the raw question on every turn. `think:false` **is**
   honoured on native `/api/chat` (verified, Ollama 0.40.16) — this is why the
   plan forbids the `/v1` path.
5. **Rerank candidates 20 → 12**, on measured cost, with no accuracy loss.

---

## 7. Bugs already fixed — do not reintroduce

- **SSE is CRLF.** `sse_starlette` separates frames with `\r\n\r\n`. The browser
  split on `\n\n`, matched nothing, and *every* answer displayed as "cut off".
  Pinned by `tests/test_sse_format.py`, which runs the browser's exact regexes
  against real bytes.
- **NFC decomposes Bengali `য়` (U+09DF).** Every Bangla literal must be folded
  through the same normaliser or it silently never matches. See `_nfc()` in
  `parse_md.py`.
- **A benchmark that warms up on the prompt it measures reports the KV cache.**
  The first `bench.py` claimed 4,100 tok/s prefill — a 200× overstatement. It
  now warms on an unrelated prompt. llama.cpp matches the *longest common
  prefix*, so even a similar prompt contaminates a measurement.
- **The engine used to fabricate its own `done` frame** when Ollama's stream
  ended without one, dressing a truncated answer as complete. A stream with no
  `done` is a **failure**, never an answer.
- **An assertion list is code and can be buggy.** Q37 was failed by the eval for
  the wrong reason because `REFUSAL_MARKERS` lacked "no mention of".
- **Apostrophes replaced by spaces** turned `how's it going` into `how s it
  going`. They are deleted, not spaced.

---

## 8. Conventions

- **Config, not constants.** Every threshold lives in `src/config.py` so the
  eval can sweep it.
- **Refusal and failure are different channels.** A technical failure must
  **never** be reported as "I don't know" — that is plan risk R11. Refusals
  travel as tokens; failures travel as `error` events. `eval_answers.py` scores
  a third outcome, `ERROR`, and any `ERROR` **invalidates the run**.
- **Ingestion fails loudly** (offline, so silence is the enemy); **the request
  path degrades quietly** (each stage has one fallback, flagged in the UI).
- **Comments explain *why*, especially where the code looks odd.** Every
  deviation above is annotated at its call site.
- Tests that need a real LLM are marked `@pytest.mark.slow` and excluded by
  default.
- Filenames never reach the reader: the model cites `[employee_handbook_v3.md]`,
  the UI strips it and lists clean document titles instead.

---

## 9. Known limitations

- **Q33 is the one failing eval question.** Asked in Bangla what canteen lunch
  costs, it gives the employee price but omits the guest price. Both are in the
  retrieved chunk and the completeness rule names this exact shape of mistake.
  Left as-is rather than patched with a Q33-shaped instruction, which would be
  tuning to the test.
- **Small talk is a fixed phrase list** (`src/chat/smalltalk.py`). An unlisted
  greeting falls through to the relevance floor and gets the out-of-scope
  refusal. Add phrases to the sets; do **not** loosen to prefix matching — that
  starts swallowing real questions like "how are you going to handle my claim?".
- **The reranker is CC-BY-NC** (`jinaai/jina-reranker-v2-base-multilingual`),
  non-commercial only. `BAAI/bge-reranker-base` (MIT) is the configured
  fallback and must be swapped before any commercial use.
- **Single user.** `OLLAMA_NUM_PARALLEL=1` and one Qdrant lock mean one
  question at a time; the API queues and then returns an explicit "busy".
- **A 4B judge is not trustworthy unaudited.** It graded two correct answers
  wrong in the first full run. Every verdict goes to `eval/results/*.csv`.

---

## 10. If you are asked to make it faster

In order of effect, with what it costs:

1. **GPU with ≥4 GB VRAM** — the real fix, nothing else is close.
2. **`LLM_MODEL=qwen3:1.7b-q4_K_M`** in `.env` — measured ~2× faster. Costs
   accuracy; re-run `eval_answers` (~30 min) before recommending it.
3. **`FINAL_TOP_K` 5 → 3** — prefill scales with document tokens. Risks the
   multi-document questions; re-run `eval_retrieval` (~6 min).
4. Trimming `SYSTEM_PROMPT` helps only the *first* question — it is cached
   afterwards.

Measure before and after. Never report a speed-up without a fresh eval.
