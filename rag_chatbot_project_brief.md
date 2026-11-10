# Build a RAG Chatbot: Team Project

Build a chatbot that answers questions about **Nodi Tech Ltd.**, a fictional company, using only the documents in `data/`. Every answer must cite its source, and the bot must say "I don't know" when the documents don't cover the question.

## What's in this kit

| Folder | Contents |
|---|---|
| `data/` | 12 company documents: HR handbook, IT security, expenses (PDF), NodiDesk product docs, a Bangla document, an outdated policy, and a vendor notice containing a prompt injection |
| `eval/eval_questions.csv` | 40 test questions with expected answers and source documents |
| `eval/conversation_tests.json` | 3 multi-turn conversations to test memory and follow-up questions |

The data deliberately contains traps:

- **Exact codes** (ND-1042, SEC-EXC). Vector search often misses these, while lexical search finds them.
- **Tables,** in the markdown files and in the PDF.
- **A superseded policy** with outdated numbers. The bot must prefer the current handbook.
- **A Bangla document,** plus English questions about it.
- **A hidden prompt injection** in `vendor_notice_printer.md`.
- **Out-of-scope questions** in the evaluation set.

## Architecture

```
INDEXING
data/ → parse (md + PDF) → chunk → add context line → embed → vector DB
                                                    └→ BM25 index

CHAT
user message → rewrite follow-up into a standalone question
  → lexical search + vector search → fuse (RRF) → rerank → top 5
  → prompt: system rules + chat history + fenced chunks
  → LLM answer with [source] citations → stream to UI
```

## Suggested stack

You can swap any part, but say why.

- **LLM:** Qwen via Ollama.
- **Embeddings:** bge-m3 (multilingual, handles Bangla). Compare it against nomic-embed-text.
- **Vector DB:** Qdrant. **Lexical:** Postgres full-text search or `rank_bm25`.
- **Reranker:** bge-reranker-v2-m3.
- **Backend:** Python + FastAPI. **UI:** a simple HTML page or Streamlit.

## Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Ingestion | All 12 files, including the PDF table, are parsed and chunked. Print the chunks and check them by eye. |
| M2 | Vector search | Embed the chunks into Qdrant, and a question returns its top 5 chunks. |
| M3 | Hybrid + rerank | BM25 is added, results are fused with RRF, and the reranker runs. ND-1042 questions find the error codes file. |
| M4 | Chat | Answers stream with citations, follow-ups work (conversation_tests.json), and the last 6 turns are kept. |
| M5 | Guardrails | Out-of-scope questions get "I don't know", the injection is ignored, and the bot never reveals its system prompt. |
| M6 | Evaluation | A script runs all 40 questions and reports retrieval hit rate and answer correctness. |

## Targets

- **Retrieval:** the correct source document appears in the top 5 for at least 90% of in-scope questions.
- **Answers:** at least 85% correct, judged against `expected_answer`.
- **Guardrails:** all 3 out-of-scope questions and all 3 guardrail questions handled correctly.
- **Version conflict:** the annual leave answer is 18 days, never 15.

## Rules for building with AI coding tools

1. Write the spec for each milestone before you prompt.
2. Do one milestone per session, run it, and commit.
3. Read every diff. If you can't explain the code, don't merge it.
4. Have the AI write tests along with the code.
5. Keep secrets out of the code, in a `.env` file that isn't committed.
6. After every change to chunking, embeddings or prompts, rerun the evaluation. The score decides, not your impression.

## Stretch goals

- Add contextual retrieval (an LLM-written context line per chunk), and measure the change in hit rate.
- Show retrieved chunks and scores in a debug panel.
- Add a metadata filter: questions about HR search only HR documents.
- Add thumbs up/down feedback and log it.
- Add a tool call: `calculate_expense_approval(amount)` returns who must approve.
