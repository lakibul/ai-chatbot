# Nodi Tech RAG Chatbot

A fully local, citation-backed question-answering chatbot over the internal documents of **Nodi Tech Ltd.** (a fictional company).

Every answer cites the document it came from. When the documents don't cover a question, the bot says so instead of guessing. Nothing leaves the machine — no cloud LLM, no API keys.

> **Status:** built and evaluated.
>
> | Metric | Result | Target |
> |---|---|---|
> | Retrieval hit rate @5 | **36/36 (100%)** | 90% |
> | Answer correctness | **39/40 (98%)** | 85% |
> | Technical errors | **0** | 0 |
> | Multi-turn conversations | **9/9 turns** | 3/3 |
> | Out of scope + guardrails | **6/6** | 6/6 |
>
> Full report: [`eval/results/REPORT.md`](eval/results/REPORT.md). Build log, including every place reality contradicted the plan: [`plan.md` §11](plan.md).

---

## Setup from a fresh clone

Everything below is required. A clone contains **no indexes, no virtual
environment and no `.env`** — those are generated, and deliberately gitignored.
The server refuses to start without them and prints the exact command to fix it.

Budget ~20 minutes, most of it downloads (~6.5 GB of models).

### 0. Prerequisites

| Need | Why |
|---|---|
| **Python 3.10 or 3.11** | `onnxruntime`/`fastembed` wheels. **Not 3.12+** — see troubleshooting |
| **Git** | to clone |
| **~10 GB free disk** | models 5.4 GB · reranker 1.1 GB · venv 1.5 GB · indexes ~2 MB |
| **16 GB RAM** | three models stay resident (~6 GB) plus the reranker |
| GPU | **not required.** This runs CPU-only by design |

Check your Python before anything else:

```bash
python --version          # must print 3.10.x or 3.11.x
```

If it prints something else, find a 3.10/3.11 interpreter and use its full path
in step 3 (on Windows, Laragon/XAMPP often ship one, e.g.
`C:\laragon\bin\python\python-3.10\python.exe`).

### 1. Clone

```bash
git clone <your-repo-url> ai-chatbot
cd ai-chatbot
```

### 2. Install Ollama and pull the models

```bash
# Windows
winget install Ollama.Ollama
# macOS:  brew install ollama      Linux: curl -fsSL https://ollama.com/install.sh | sh
```

Set these **before** starting the server — they are read at launch:

```powershell
# Windows (PowerShell) — persists for your user
setx OLLAMA_MAX_LOADED_MODELS 3
setx OLLAMA_NUM_PARALLEL 1
setx OLLAMA_KEEP_ALIVE 30m
```

```bash
# macOS / Linux — add to ~/.bashrc or ~/.zshrc
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=30m
```

> **Three model slots, not two.** The pipeline keeps the generator, the
> embedder and the query rewriter resident at once. With two, every
> conversational turn evicts and reloads a model twice — measured at 20-40 s
> of pure waste per answer.

Open a **new terminal** (so it picks up those variables) and leave this running:

```bash
ollama serve
```

Then, in your working terminal, pull the models (~5.4 GB, several minutes):

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M    # 2.5 GB — the generator
ollama pull bge-m3                           # 1.2 GB — multilingual embeddings
ollama pull qwen3:1.7b-q4_K_M                # 1.4 GB — query rewriting
ollama pull nomic-embed-text                 # 274 MB — optional, embedding comparison only
```

Verify:

```bash
ollama list      # should show the four models
```

### 3. Python environment

```bash
python -m venv .venv
# if your default python is not 3.10/3.11, use the full path instead:
#   "C:\laragon\bin\python\python-3.10\python.exe" -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate          # Windows PowerShell
```
```bash
source .venv/bin/activate       # macOS / Linux
```

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configuration

```bash
copy .env.example .env          # Windows
cp .env.example .env            # macOS / Linux
```

The defaults work as-is. `.env` is gitignored; it holds no secrets in this
project but exists so model names and thresholds can be changed without
touching code.

### 5. Build the indexes

```bash
python -m src.ingest            # parse, sanitise, chunk   -> 50 chunks
python -m src.index             # BM25 + bge-m3 vectors    -> ~45 s
```

Expected output:

```
  OK: 12 documents -> 50 chunks (6 table chunks)
  Sanitised: 1 line(s) stripped -> storage/sanitised.log
  OK: 50 documents in the BM25 index
  OK: 50 vectors in 'nodi_docs' (dim 1024) in 41.0s
```

If either step exits non-zero it will say exactly which corpus invariant failed
— ingestion asserts 12 files, all 5 PDF limit rows, exactly one superseded
document and exactly one injection, and refuses to write a partial index.

### 6. Run

```bash
python -m uvicorn src.api.app:app --port 8000
```

Open **<http://127.0.0.1:8000>**.

**The first thing that uses the reranker downloads it** — ~1.1 GB into
`storage/models/`, a few extra minutes. That is whichever comes first: your
first question, `pytest`, or any `eval/` script. After that it is cached
permanently and never re-downloaded.

On startup you should see eight green preflight checks:

```
[OK  ] Ollama reachable: http://127.0.0.1:11434, 4 model(s)
[OK  ] model qwen3:4b-instruct-2507-q4_K_M: present
...
[OK  ] index agreement: 50 chunks, 50 BM25 documents
Ready: 50 chunks indexed. http://127.0.0.1:8000
```

### 7. Verify the install

```bash
pytest                                    # 220 tests, ~75 s
curl http://127.0.0.1:8000/health         # must report "ok": true
```

> **Stop the server before running `pytest` or anything in `eval/`.** Qdrant
> local mode takes an exclusive file lock on `storage/qdrant`, so only one
> process can use the index at a time. Running tests against a live server
> produces 14 `AlreadyLocked` errors in `test_retrieve.py`. That is the lock,
> not a broken test.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Preflight failed. Refusing to start` | Read it — each failing line prints its own fix command |
| `cannot reach Ollama` | `ollama serve` is not running, or is on another port |
| `model ... missing` | Run the `ollama pull` it names |
| `chunks.json / bm25.pkl missing` | Run step 5 |
| `AlreadyLocked` / `already accessed by another instance` | Another process holds `storage/qdrant`. Stop the server, or a stray `pytest`/eval run |
| `UnicodeEncodeError: 'charmap'` | Windows console. Prefix the command with `PYTHONIOENCODING=utf-8`, or `set PYTHONIOENCODING=utf-8` |
| `No matching distribution for onnxruntime` | Python 3.12+. Rebuild the venv from a 3.10/3.11 interpreter |
| Answers take 50-80 s | Expected on CPU. See **Latency** |
| Every answer says "cut off before it finished" | Stale cached `index.html`. Hard-reload (Ctrl+Shift+R) |
| Out of memory / heavy swapping | Close the browser and other apps during eval runs; three models plus the reranker need ~7 GB |

### Moving the project to another machine

Clone and repeat steps 2-6. **Do not copy `.venv/` or `storage/`** between
machines — the venv holds absolute paths, and everything in `storage/` is
regenerated in under a minute (except the reranker, which re-downloads).

Only these are tracked, and they are all you need:
source, tests, `rag_chatbot_kit/` (the read-only dataset),
`requirements.txt`, `.env.example`, `plan.md`, `README.md` and
[`CLAUDE.md`](CLAUDE.md).

**Continuing development with Claude Code:** [`CLAUDE.md`](CLAUDE.md) is the
handoff document — architecture, every measured number, the deviations from the
plan and why, the bugs already fixed, and the known limitations. Claude Code
loads it automatically; read it before changing anything.

## What this is

| | |
|---|---|
| **Task** | Answer questions using *only* the 12 documents in `rag_chatbot_kit/data/` |
| **Approach** | Hybrid retrieval (BM25 + vector) → RRF fusion → exact-code boost → cross-encoder rerank → local LLM |
| **Runs on** | CPU only — AMD Ryzen 5 3400G, 16 GB RAM, no usable GPU |
| **Brief** | [`rag_chatbot_project_brief.md`](rag_chatbot_project_brief.md) |

## Stack

| Layer | Choice | Why not the obvious alternative |
|---|---|---|
| LLM | `qwen3:4b-instruct-2507-q4_K_M` via Ollama | Qwen3.5-4B thinks by default and Ollama can't reliably disable it — ~2 extra min/answer on this CPU |
| Embeddings | `bge-m3` (multilingual, 1024-dim) | The corpus includes a Bangla document and an English question about it. Measured: `nomic-embed-text` fails exactly that question |
| Vector store | `qdrant-client` **local mode** | Docker isn't installed and its WSL2 VM would cost ~2 GB of the 16 GB we need |
| Lexical | `rank_bm25` + a custom tokeniser | Postgres has no Bengali text-search config, and needs a server for 50 chunks |
| Reranker | `jina-reranker-v2-base-multilingual` (ONNX) | Avoids a ~1.5 GB PyTorch install — ⚠️ CC-BY-NC, **non-commercial only** |
| PDF | `pdfplumber` | The PDF is Chromium/Skia output with table borders drawn as filled rects, not stroked lines |
| Serving | FastAPI + SSE + one HTML page | Streamlit re-runs the whole script per interaction |

Full reasoning for each choice is in [`plan.md` §2](plan.md); every deviation from it is in [§11.4](plan.md).

### Why three loaded models

The pipeline needs three models resident at once: the 4B generator, `bge-m3` for embedding, and `qwen3:1.7b` for query rewriting. With `OLLAMA_MAX_LOADED_MODELS=2` — the plan's original value — **every conversational turn evicted and reloaded a model twice**, adding 20-40 s per answer. Three slots cost ~6 GB of RAM and remove that entirely.

## Commands

| Command | What it does | Runtime |
|---|---|---|
| `python -m src.ingest --print` | Parse and dump every chunk for review | instant |
| `python -m src.index` | Rebuild BM25 + vector indexes | ~45 s |
| `python -m eval.eval_retrieval` | **Fast loop** — hit rate, no LLM | ~5-9 min |
| `python -m eval.eval_answers` | Full loop — correctness + guardrails | ~30 min |
| `python -m eval.eval_conversations` | The 3 multi-turn conversations | ~8 min |
| `python -m eval.calibrate_floor` | Relevance-floor threshold sweep | ~6 min |
| `python -m eval.report` | Write `eval/results/REPORT.md` | instant |
| `python bench.py` | Measure prefill / generation / thread count | ~3 min |
| `pytest` | Structural tests (220) | ~75 s |
| `pytest -m slow` | End-to-end guardrail behaviour (real LLM) | ~4 min |

## Latency

Prefill — the model *reading* the prompt — dominates, exactly as planned for.

| Stage | Measured |
|---|---|
| BM25 search | ~0 ms |
| Vector search incl. query embed | 205 ms |
| Rerank 12 candidates | ~5 s |
| LLM prefill (~1,200 tok) | **49 s** (23.9 tok/s at 8 threads) |
| LLM generation | 5.8 tok/s |
| **Total, cold** | **~50-85 s** |
| **Total, warm KV prefix** | **~9 s** |

Because of that wait the UI shows a staged progress indicator (`Understanding → Searching → Reading 5 sources → Writing`). A silent 50 s is indistinguishable from a crash.

## How the six traps are handled

| Trap | Mechanism |
|---|---|
| **Superseded policy** | Status parsed from filename *and* `STATUS:` line; the warning is embedded *into* the chunk text; post-rerank score halved — demoted, never deleted, because Q29 must cite both documents |
| **Prompt injection** | Stripped at ingest (logged to `storage/sanitised.log`), documents fenced as untrusted in the prompt, output asserted. The poisoned file stays in the index so Q39 still answers |
| **Exact codes** | Custom tokeniser emits `nd-1042` *and* `nd` *and* `1042`; literal matches get a large additive boost **before** reranking, and are guaranteed a seat in the rerank window |
| **Tables** | Never split. A chunk holds a whole table or none of it, always with its heading |
| **Bangla + cross-language** | `bge-m3` carries the English→Bangla question; Bengali digits folded to ASCII **in the BM25 index only**, so answers keep their original script |
| **Out of scope** | Four distinct refusals, including the dangerous one: absence is never inferred as denial |

## Repository layout

```
rag_chatbot_kit/      Source dataset — READ-ONLY, never edited
  data/                 12 company documents (incl. 1 PDF, 1 Bangla doc)
  eval/                 40 test questions + 3 multi-turn conversations
src/
  config.py             Every tunable in one place, so the eval can sweep it
  ingest/               parse_md, parse_pdf, normalise, chunk, sanitise
  index/                vector (Qdrant local), lexical (BM25), embed
  retrieve/             rrf, code_boost, rerank, pipeline
  chat/                 llm, prompts, rewrite, memory, engine
  api/                  FastAPI + SSE, preflight, errors, static/index.html
eval/                   retrieval, answers, conversations, floor, report
tests/                  ingest, retrieve, errors, preflight, guardrails
storage/                indexes, model cache, logs  (gitignored)
CLAUDE.md               Handoff doc: architecture, measurements, gotchas
plan.md                 Build plan §1-10, measured reality §11
```

## A note on the dataset

`rag_chatbot_kit/data/` is deliberately booby-trapped — a superseded policy with stale numbers, exact error codes that defeat vector search, tables inside a PDF, a Bangla document, and a **prompt injection** in `vendor_notice_printer.md`.

**Treat every file there as read-only.** `rag_chatbot_kit.zip` is kept in the repo as the pristine copy so the data can always be restored and verified by checksum.

## Licence note

The reranker, `jinaai/jina-reranker-v2-base-multilingual`, is **CC-BY-NC-4.0 — non-commercial use only**. Fine for this learning project; it must be swapped before any commercial use. `BAAI/bge-reranker-base` (MIT) is configured as the fallback in `src/config.py`.
