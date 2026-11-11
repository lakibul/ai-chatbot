# CLAUDE.md — working notes for this repository

Read this before changing anything. It is the handoff document: what exists,
why it is built this way, what has already been measured, and the specific
mistakes that cost time here so they are not repeated.

**Project:** a fully local, citation-backed RAG chatbot over 12 internal
documents of a fictional company, Nodi Tech Ltd. No cloud LLM, no API keys.

**Status:** complete and evaluated. 39/40 answer correctness, 36/36 retrieval,
220 tests passing. See `eval/results/REPORT.md`.

**Reproduced on a second machine 2026-11-11** — same code, same model tags,
same config: 39/40, 36/36, 225/225 tests, ~8× faster. `plan.md` §12 is the
portability log; §5 below now carries both machines' numbers side by side.

---

## 1. Setting this up on a new machine

A clone has **no virtual environment, no indexes and no `.env`** — all four
are generated and gitignored. Run every step below. Budget ~20 minutes, almost
all of it downloading ~6.5 GB of models.

Needs **Python 3.10 or 3.11** (not 3.12+ — the `onnxruntime` wheels do not
exist), ~10 GB of disk and 16 GB of RAM.

### Step 1 — get the code

```bash
git clone <repo-url> ai-chatbot
cd ai-chatbot
```

### Step 2 — Ollama, with three model slots

```powershell
winget install Ollama.Ollama          # macOS: brew install ollama

setx OLLAMA_MAX_LOADED_MODELS 3
setx OLLAMA_NUM_PARALLEL 1
setx OLLAMA_KEEP_ALIVE 30m
```

**If winget hangs at 0 bytes** (it may demand a two-letter geographic region),
download the installer directly and run it with **Inno Setup** flags — it is
not NSIS, so `/S` is not silent, it opens a GUI wizard and waits forever:

```powershell
# https://github.com/ollama/ollama/releases/download/v0.40.16/OllamaSetup.exe
.\OllamaSetup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
```

The installer then launches the Ollama tray app, **which never exits** — so
`Start-Process -Wait` blocks even after a successful install. Check for
`%LOCALAPPDATA%\Programs\Ollama\ollama.exe` on disk instead of waiting on the
exit code. GitHub's asset host may also throttle a single connection hard
(113 KB/s measured while the link did 3.9 MB/s); parallel HTTP Range requests
fix that.

These are read when the server launches, so open a **new terminal** and leave
this running:

```bash
ollama serve
```

Three slots, not two: the generator, the embedder and the rewriter must all
stay resident. With two, measured, every conversational turn evicted and
reloaded a model twice — 20-40 s of pure waste per answer.

**Verify it, do not assume it.** A server started *by the installer* inherits
the installer's environment, which can predate your `setx`. Kill it, run
`ollama serve` yourself, and confirm the server's own startup line says
`OLLAMA_MAX_LOADED_MODELS:3` — it prints its whole config as `msg="server
config"`. A silently-2-slot server looks fine and just runs slowly.

### Step 3 — pull the models (~5.4 GB)

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M    # 2.5 GB  writes the answers
ollama pull bge-m3                           # 1.2 GB  multilingual embeddings
ollama pull qwen3:1.7b-q4_K_M                # 1.4 GB  rewrites follow-ups
ollama pull nomic-embed-text                 # 274 MB  optional, comparison only
ollama list                                  # confirm all four
```

The two **language** models are already quantised -- `q4_K_M` *is* 4-bit. The two
**embedders** are not: `ollama show bge-m3` and `nomic-embed-text` both report
`F16`. Nothing is quantised here, and on these tags nothing can be:

```
$ ollama create probe -q q4_0 -f Modelfile      # FROM qwen3:4b-...-q4_K_M
Error: create-time quantization is only supported for safetensors imports;
       quantize GGUF models before importing
```

Ollama's own `create -q` is therefore a dead end. It is **not** the only route,
though: Ollama bundles llama.cpp's quantiser at
`%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-quantize.exe`, which can
quantise a GGUF directly, with no download.

So the honest position is *could, but should not*:

| Model | Source precision | Quantisable with the bundled tool? |
|---|---|---|
| `qwen3:4b-instruct-...-q4_K_M` | already Q4_K_M | only with `--allow-requantize`, whose own help says it "can severely reduce quality" |
| `qwen3:1.7b-q4_K_M` | already Q4_K_M | same |
| `bge-m3` | **F16** | yes, cleanly |
| `nomic-embed-text` | **F16** | yes, cleanly |

The reason not to is not feasibility, it is consequence. A requantised
generator is a different model artefact, so every number in §5 and §11 stops
applying. A quantised `bge-m3` rewrites **every vector** in `storage/qdrant`,
so the index must be rebuilt and `eval_retrieval` re-run before the 36/36 can
be quoted again. And the upside is not there: a warm query embed is **29 ms**
of a ~9 s answer (0.3%). The time is in prefill, which AVX-512 already fixed.

Measure first. If a future machine makes embedding the bottleneck, quantise
`bge-m3` to `q8_0`, rebuild the index, and re-run the retrieval eval -- in that
order.

### Step 4 — Python environment

```bash
python -m venv .venv
# if the default python is not 3.10/3.11, use a full path, e.g.
#   "C:\laragon\bin\python\python-3.10\python.exe" -m venv .venv

.venv\Scripts\activate          # Windows
source .venv/bin/activate        # macOS / Linux

python -m pip install --upgrade pip
pip install -r requirements.txt

copy .env.example .env           # cp on macOS / Linux
```

### Step 5 — build the indexes (~45 s)

```bash
python -m src.ingest     # -> storage/chunks.json   "12 documents -> 50 chunks"
python -m src.index      # -> storage/bm25.pkl + storage/qdrant
```

Both fail loudly and exit non-zero if a corpus invariant breaks, naming which.

### Step 6 — run it

```bash
python -m uvicorn src.api.app:app --port 8000
```

Open <http://127.0.0.1:8000>. Eight preflight checks must print `[OK]`; the
server refuses to start otherwise and prints the exact fix.

The **first thing that uses the reranker downloads it** — ~1.1 GB into
`storage/models/`. That is whichever comes first: a question, `pytest`, or an
eval script.

### Step 7 — verify

```bash
pytest                                 # 220 tests; ~3-4 min first run, ~75 s after
curl http://127.0.0.1:8000/health      # "ok": true
```

### Do not copy between machines

`.venv/` holds absolute paths and `storage/` is regenerated in a minute. Copy
neither. Everything needed is tracked in git.

This still stands even though a wholesale copy to machine B happened to work:
`pyvenv.cfg` pins `home = C:\laragon\bin\python\python-3.10`, and that exact
path existed on both boxes. That is luck, not portability. The copy also
dragged stale `__pycache__` across, so tracebacks printed the *old* machine's
paths (`C:\ai-chatbot\...`) and sent debugging to a directory that does not
exist here. Delete `__pycache__` and `.pytest_cache` after any such copy.

### When it goes wrong

| Message | Fix |
|---|---|
| `Preflight failed` | Each failing line prints its own fix |
| `cannot reach Ollama` | `ollama serve` is not running |
| `model ... missing` | Run the `ollama pull` it names |
| `chunks.json / bm25.pkl missing` | Step 5 |
| `AlreadyLocked` | Something else holds the index — stop the server |
| `UnicodeEncodeError: 'charmap'` | Prefix with `PYTHONIOENCODING=utf-8` |
| `No matching distribution for onnxruntime` | Python 3.12+. Rebuild the venv on 3.10/3.11 |
| Every answer says "cut off" | Stale cached page. Hard-reload with Ctrl+Shift+R |

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

**`num_gpu: 0` is not a preference on machine A, it prevents a crash.**
Measured 2026-11-11: an `/api/embed` call to `bge-m3` *without* that option
kills its runner outright —

```
llama-server process has terminated: exit status 0xc0000409:
The system detected an overrun of a stack-based buffer
```

It is reproducible on every attempt and recovers the moment
`"options":{"num_gpu":0}` is added. Ollama tries to offload to the 2 GB
GT 1030 and dies. `src/index/embed.py` and `src/chat/llm.py` always send it,
which is why the application is unaffected — but **any hand-rolled `curl`
against Ollama on this box must include it too**, or you will spend an hour
debugging a model that is actually fine. This is plan risk R6 with a concrete
failure mode rather than the predicted "thrash".

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

**Two machines are recorded here.** Machine A is where the project was built
and where the numbers quoted in `README.md` and `eval/results/REPORT.md` come
from. Machine B is a faster box the project was moved to on 2026-11-11. The
code, the model tags and `src/config.py` are **identical** on both —
`plan.md` §12 is the full portability log.

- **A** — Ryzen 5 3400G (4c/8t, Zen+, AVX2, **no AVX-512**), 16 GB DDR4-2666,
  GT 1030 2 GB (**unusable**: a 4B Q4 needs ~2.5 GB).
- **B** — i7-11700 (8c/16t, Rocket Lake, **AVX-512 + VBMI + VNNI**), 32 GB,
  RX 6600 8 GB (**usable but deliberately unused**).

On both, `num_gpu: 0` is forced and Ollama never appears as a GPU process.
On B that is now a *choice* for demo parity, not a hardware limit.

| | A (built on) | B (2026-11-11) |
|---|---|---|
| Prefill, 4B, cold | **23.9 tok/s** (1,162 tok → 48.6 s) | **250.9 tok/s** (1,317 tok → 5.2 s) |
| Generation, 4B | **5.8 tok/s** | **12.7 tok/s** |
| Prefill, 1.7b, cold | 56.3 tok/s · generation 10.3 tok/s | — |
| Bare question, no documents | prefill **0.8 s** | — |
| Repeat question (prefix cached) | prefill **0.2 s**, total ~9 s | — |
| Rerank, per candidate | 0.43 s (20 → 8.6 s) | **0.074 s** (12 → 0.89 s) |
| BM25 / vector+embed | ~0 ms / 205 ms | ~0 ms / ~0.23 s per chunk |
| Best `num_thread` | **8** (48.6 s vs 60.5 s at 4) — SMT *helps* Zen+ | **8** (5.2 s vs 8.0 s at 16) — SMT *hurts*; 8 = physical cores |
| Index build | ~45 s (50 chunks) | **11.5 s** |

**`NUM_THREAD=8` is correct on both machines for opposite reasons.** Do not
"simplify" that comment in `config.py` — measure before changing it.

**Most of B's gain is AVX-512, not cores.** Ollama logs
`AVX512_VNNI = 1` on B; VNNI accelerates the quantised integer matmul that
dominates prefill, which is why prefill gained 10.5× while generation — still
bandwidth-bound — gained only 2.2×. The §5 model of the bottleneck held; only
the constant moved.

**Latency is RAG, not the model.** A bare question prefills in 0.8 s; five
retrieved documents add 15-25 s of reading before the first token. A 2.4×
smaller model is only ~2× faster. The real fix is a GPU with ≥4 GB VRAM —
prefill is the embarrassingly-parallel phase GPUs exist for.

The system prompt (~740 tok) is byte-identical every turn and llama.cpp
caches it. Documents (~390 tok) are new each question — that is what you pay
for. **Never interpolate anything into `SYSTEM_PROMPT`.**

### Eval runtimes

| Command | A | B |
|---|---|---|
| `pytest` | ~75 s | **27 s** |
| `pytest -m slow` | ~4 min (real LLM calls) | **36 s** |
| `eval_retrieval` | ~6-9 min (reranker dominates) | **~1 min** |
| `eval_answers` | ~30 min | **8.3 min** (mean 9.5 s/answer) |
| `eval_conversations` | ~8 min | **1.7 min** |

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
- **Progress reporting is additive and must stay that way.**
  `RetrievalResult.stages` carries `bm25_ms`, `vector_ms`, `fuse_ms`,
  `rerank_ms`, `fused_count` and `code_boosted` purely so the UI can show
  which sub-step was slow. Delete every one of them and the hits, their order
  and their scores are unchanged -- that is the property to preserve. They
  reach the browser inside the `searched` stage frame.
- **The UI never hard-codes a model name.** Each sub-step in the step
  inspector carries a tool chip, and the four model chips resolve from
  `/health` at run time (`{m:'llm'|'embed'|'rewrite'|'reranker'}`). All four
  are `.env`-overridable, so a literal there would eventually name a model the
  server is not running -- worse than naming nothing. Chips render as
  `local model` until `/health` answers, then correct themselves.
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
- **The example picker offers only measured questions.** Every entry in
  `EXAMPLES` in `index.html` is copied verbatim from
  `rag_chatbot_kit/eval/eval_questions.csv`, so the UI can never suggest a
  question the project has no evidence about. Q33 is deliberately omitted
  (section 9). The Bangla entry must keep the CSV's DECOMPOSED `ড়`
  (U+09A1 U+09BC), not precomposed U+09DC -- see the NFC trap in section 7.
- Filenames never reach the reader: the model cites `[employee_handbook_v3.md]`,
  the UI strips it and lists clean document titles instead.

---

## 9. Known limitations

- **Q33 is the one failing eval question.** Asked in Bangla what canteen lunch
  costs, it gives the employee price but omits the guest price. Both are in the
  retrieved chunk and the completeness rule names this exact shape of mistake.
  Left as-is rather than patched with a Q33-shaped instruction, which would be
  tuning to the test.
  **Verified still failing on machine B, where the judge wrongly passed it.**
  The judge printed `CORRECT` while its own stated reason ended "...though it
  omits the price for guests", producing a headline 40/40. The answer contains
  no `২০০`. **Always audit Q33's CSV row before quoting a 40/40.** This is the
  optimistic twin of the pessimistic misgrade in §7 — the judge errs in *both*
  directions.
- **C1 turn 3 ("And what about sick leave?") fails on machine B** and passed on
  A. Reproducible across four runs, not flaky. `rewrite_query` returns the
  question unchanged and the rewrite model echoes it verbatim **with no
  exception** — so this is *not* the silent-fallback bug in §6.4; `think:false`
  works. The model simply judges the question standalone, so "carry forward"
  never enters the query. Cause is almost certainly AVX-512 kernels shifting a
  borderline decision at `temperature 0.1`. Not fixed: tuning the rewrite
  prompt to recover one assertion would be tuning to the test.
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
