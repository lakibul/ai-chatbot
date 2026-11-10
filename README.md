# Nodi Tech RAG Chatbot

A fully local, citation-backed question-answering chatbot over the internal documents of **Nodi Tech Ltd.** (a fictional company).

Every answer cites the document it came from. When the documents don't cover a question, the bot says so instead of guessing. Nothing leaves the machine — no cloud LLM, no API keys.

> **Status:** planning complete, implementation not started.
> The full build plan lives in **[`plan.md`](plan.md)**. Start there.

---

## What this is

| | |
|---|---|
| **Task** | Answer questions using *only* the 12 documents in `rag_chatbot_kit/data/` |
| **Approach** | Hybrid retrieval (BM25 + vector) → RRF fusion → cross-encoder rerank → local LLM |
| **Runs on** | CPU only — AMD Ryzen 5 3400G, 16 GB RAM, no usable GPU |
| **Brief** | [`rag_chatbot_project_brief.md`](rag_chatbot_project_brief.md) |

## Stack

| Layer | Choice | Why not the obvious alternative |
|---|---|---|
| LLM | `qwen3:4b-instruct-2507-q4_K_M` via Ollama | Qwen3.5-4B thinks by default and Ollama can't reliably disable it — ~2 extra min/answer on this CPU |
| Embeddings | `bge-m3` (multilingual) | The corpus includes a Bangla document and an English question about it |
| Vector store | `qdrant-client` **local mode** | Docker isn't installed and its WSL2 VM would cost ~2 GB of the 16 GB we need |
| Lexical | `rank_bm25` | Postgres has no Bengali text-search config, and needs a server for ~90 chunks |
| Reranker | `jina-reranker-v2-base-multilingual` (ONNX) | Avoids a ~1.5 GB PyTorch install — ⚠️ CC-BY-NC, non-commercial only |
| PDF | `pdfplumber` | The PDF is Chromium/Skia output emitting text glyph-by-glyph; `pypdf` returns word-salad |
| Serving | FastAPI + SSE + one HTML page | Streamlit re-runs the whole script per interaction |

Full reasoning for each choice is in [`plan.md` §2](plan.md).

## Targets

| Metric | Target |
|---|---|
| Retrieval hit rate @5 | ≥ 90% (≥ 33 of 36 in-scope questions) |
| Answer correctness | ≥ 85% (≥ 34 of 40) |
| Out-of-scope + guardrails | 6 / 6 |
| Version conflict | Annual leave is **18 days**, never 15 |

## Milestones

`M0` setup → `M1` ingestion → `M2` vector search → `M3` hybrid + rerank → `M4` chat → `M4.5` error handling → `M5` guardrails → `M6` evaluation

## Repository layout

```
rag_chatbot_kit/      Source dataset — READ-ONLY, never edited
  data/                 12 company documents (incl. 1 PDF, 1 Bangla doc)
  eval/                 40 test questions + 3 multi-turn conversations
plan.md               Full build plan: decisions, milestones, risks
rag_chatbot_project_brief.md
```

`src/`, `eval/`, `tests/` and `storage/` arrive with M0–M1. Generated state (`storage/`, `eval/results/`, `.venv/`, `.env`) is gitignored.

## A note on the dataset

`rag_chatbot_kit/data/` is deliberately booby-trapped — a superseded policy with stale numbers, exact error codes that defeat vector search, tables inside a PDF, a Bangla document, and a **prompt injection** in `vendor_notice_printer.md`.

**Treat every file there as read-only.** `rag_chatbot_kit.zip` is kept in the repo as the pristine copy so the data can always be restored and verified by checksum.
