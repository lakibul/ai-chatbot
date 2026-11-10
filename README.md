# Nodi Tech RAG Chatbot

A fully local, citation-backed question-answering chatbot over the internal documents of **Nodi Tech Ltd.** (a fictional company).

Every answer cites the document it came from. When the documents don't cover a question, the bot says so instead of guessing. Nothing leaves the machine — no cloud LLM, no API keys.

> **Status:** built and evaluated. See [`eval/results/REPORT.md`](eval/results/REPORT.md) for the measured numbers and [`plan.md` §11](plan.md) for the build log, including every place reality contradicted the plan.

---

## Quick start

```bash
# 1. Ollama (once)
winget install Ollama.Ollama
setx OLLAMA_MAX_LOADED_MODELS 3     # three, not two — see "Why three" below
setx OLLAMA_NUM_PARALLEL 1
ollama serve                         # in its own terminal

ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull bge-m3
ollama pull qwen3:1.7b-q4_K_M
ollama pull nomic-embed-text         # only for the embedding comparison

# 2. Python (once) — 3.10
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# 3. Build the indexes (~45 s)
python -m src.ingest                 # parse, sanitise, chunk  -> 50 chunks
python -m src.index                  # BM25 + bge-m3 vectors

# 4. Run
python -m uvicorn src.api.app:app --port 8000
```

Open <http://127.0.0.1:8000>. The first question after a cold start takes ~50-80 s (see **Latency** below); the server refuses to start at all if Ollama or an index is missing, and prints the exact command to fix it.

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
| `python -m eval.eval_answers` | Full loop — correctness + guardrails | ~50 min |
| `python -m eval.eval_conversations` | The 3 multi-turn conversations | ~9 min |
| `python -m eval.calibrate_floor` | Relevance-floor threshold sweep | ~6 min |
| `python -m eval.report` | Write `eval/results/REPORT.md` | instant |
| `python bench.py` | Measure prefill / generation / thread count | ~3 min |
| `pytest` | Structural tests | ~2 min |
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
plan.md                 Build plan §1-10, measured reality §11
```

## A note on the dataset

`rag_chatbot_kit/data/` is deliberately booby-trapped — a superseded policy with stale numbers, exact error codes that defeat vector search, tables inside a PDF, a Bangla document, and a **prompt injection** in `vendor_notice_printer.md`.

**Treat every file there as read-only.** `rag_chatbot_kit.zip` is kept in the repo as the pristine copy so the data can always be restored and verified by checksum.

## Licence note

The reranker, `jinaai/jina-reranker-v2-base-multilingual`, is **CC-BY-NC-4.0 — non-commercial use only**. Fine for this learning project; it must be swapped before any commercial use. `BAAI/bge-reranker-base` (MIT) is configured as the fallback in `src/config.py`.
