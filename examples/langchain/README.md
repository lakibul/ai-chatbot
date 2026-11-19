# LangChain examples (learn by running)

Every file here was run, and its output is quoted in **PROJECT-GUIDE-BN.md, part 6**.

## Set up (one time, in a SEPARATE environment)

Do not install these into the project's `.venv`. That one holds only the few packages the
app needs, pinned in `requirements.txt`.

```bash
python -m venv lcenv
lcenv\Scripts\activate
pip install langchain langgraph langchain-text-splitters langchain-ollama langchain-classic rank_bm25 grandalf
```

Versions these were verified with (Python 3.10, November 2026): `langchain-core 1.7.7`,
`langchain-ollama 1.1.0`, `ollama 0.6.3`, `langchain 1.4.10`, `langgraph 1.2.23`,
`langchain-text-splitters 1.1.3`, `langchain-classic 1.0.8`. LangChain changes fast. If a file
breaks on newer versions, read the warnings (`python -W always file.py`).

## The files

| File | Guide chapter | Needs Ollama? | Shows |
|---|---|---|---|
| `c01_basics.py` | 6.3 – 6.6 | no | messages, models, prompts, the `{ }` trap, parsers, trimming |
| `c02_lcel.py` | 6.7 | no | the `\|` pipe, parallel, branch, retry, fallback, streaming, events, graph |
| `c03_rag.py` | 6.8 – 6.10 | no | documents, splitting, vector store, custom retriever, hybrid, RAG chain |
| `c04_memory_tools_callbacks.py` | 6.11 – 6.16 | no | memory (and the deprecated way), tools, callbacks, cache, rate limit |
| `c05_graph_agent.py` | 6.13, 6.14 | no | LangGraph: routing, loops, memory, human approval, an agent loop |
| `c06_real_ollama.py` | 6.20 | **yes** | real models: structured output, tool calling, embeddings, RAG over this project's data |
| `c07_structured_output_reliability.py` | 6.20 | **yes** | repeats the structured-output test, because one run said 3/3 and the next 0/3 |

`c01`–`c05` use fake models, so they are instant and give the same result every time.
Run them from the project folder: `python examples/langchain/c01_basics.py`.

`c06` is slow on a CPU (about 5 minutes). It needs Ollama running with
`qwen3:1.7b-q4_K_M`, `qwen3:4b-instruct-2507-q4_K_M` and `bge-m3`, and it reads
`storage/chunks.json`, so build the index first (`python -m src.ingest`).

## Two things worth knowing before you run them

- `c02` proves "runs at the same time" with a barrier, not a stopwatch. The stopwatch version gave
  0.31 s on one run and 1.04 s on the next, because the machine was busy.
- `c06` must send `num_gpu=0` to the embedder. Without it `bge-m3` crashes on this machine
  (see `CLAUDE.md`, section 2).
