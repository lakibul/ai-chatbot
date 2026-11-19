# Code map — how this project is built

A plain-English tour of the **code**: which file does what, and how they fit
together.

This is the "where is everything" guide. Two companions:

| Guide | Answers |
|---|---|
| **This file** | *Which file does what?* |
| [`PROJECT-GUIDE-EN.md`](PROJECT-GUIDE-EN.md) | *What is a context window? What is BM25?* (the ideas) |
| [`CLAUDE.md`](CLAUDE.md) | *What did we measure? What breaks if I touch this?* (handoff notes) |

---

## 1. The big picture: three programs, not one

Most people assume a chatbot is one program. This is **three**, and they run at
different times.

```
   ┌─────────────────────────────────────────────────────────────┐
   │  PROGRAM 1 — INGEST                      run once, when     │
   │  Read the 12 documents, clean them, cut them into 50 pieces │
   │  python -m src.ingest                                       │
   └──────────────────────────┬──────────────────────────────────┘
                              │  writes storage/chunks.json
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  PROGRAM 2 — INDEX              the documents change        │
   │  Build two search indexes from those 50 pieces              │
   │  python -m src.index                                        │
   └──────────────────────────┬──────────────────────────────────┘
                              │  writes storage/bm25.pkl + storage/qdrant/
                              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  PROGRAM 3 — SERVER            runs all day, answers people │
   │  python -m uvicorn src.api.app:app --port 8000              │
   └─────────────────────────────────────────────────────────────┘
```

Programs 1 and 2 together take about 45 seconds (measured on the slower of
the two machines this was built on).

**Why this matters:** they run only when the documents change. Program 3 is the only one that runs while someone is waiting. That is
why all the speed work went into program 3.

---

## 2. The folder tree

```
ai-chatbot/
│
├── rag_chatbot_kit/        THE GIVEN MATERIAL — never edit this
│   ├── data/               the 12 company documents (11 .md + 1 .pdf)
│   └── eval/               the 40 test questions + 3 test conversations
│
├── src/                    THE APPLICATION
│   ├── config.py           every setting, in one place
│   ├── ingest/             program 1: read and cut up the documents
│   ├── index/              program 2: build the two search indexes
│   ├── retrieve/           finding the right passages
│   ├── chat/               turning passages into an answer
│   └── api/                the web server and the web page
│
├── eval/                   MEASURING — how good is it?
├── tests/                  281 automatic checks
├── storage/                GENERATED — safe to delete and rebuild
│
└── *.md, start-lan.bat     docs and the launcher
```

Three rules worth knowing:

1. **`rag_chatbot_kit/` is read-only.** It is the exam paper. Editing it would
   be cheating.
2. **`storage/` is disposable.** Delete it and run the two build commands; you
   get it back. It is not in git for that reason.
3. **`src/config.py` holds every number.** No setting is written anywhere else.

---

## 3. What each file does

### `src/config.py` — the control panel (206 lines)

Every tunable number lives here: which models to use, how many passages to
fetch, the size of the context window, time limits. Nothing else in the code
hard-codes a number.

**Why it matters:** the evaluation can change one number and re-measure. If
settings were scattered across twenty files, that would be impossible.

### `src/ingest/` — reading the documents (program 1)

| File | What it does |
|---|---|
| `parse_md.py` | Reads a Markdown file and understands its headings |
| `parse_pdf.py` | Reads the one PDF. Its table has no lines, only filled rectangles, so this rebuilds the rows |
| `normalise.py` | Fixes letters and digits so Bangla text matches reliably |
| `sanitise.py` | **Removes the hidden attack** planted in one document, and logs what it removed |
| `chunk.py` | Cuts documents into passages. A table is never cut in half |
| `models.py` | The shape of one passage (its text, document name, section, status) |
| `build.py` | Runs all of the above in order, and **refuses to finish if something is wrong** |

**The important one is `build.py`.** It checks that all 12 files are present,
that the PDF table has its 5 rows, that exactly one document is marked
out-of-date, and that exactly one attack was found. If any check fails it stops
with an error instead of writing a half-built index.

### `src/index/` — building the two search indexes (program 2)

| File | What it does |
|---|---|
| `embed.py` | Sends text to the `bge-m3` model, gets back 1,024 numbers |
| `vector.py` | Stores those numbers in Qdrant (search by **meaning**) |
| `lexical.py` | Builds the BM25 word index (search by **exact words**) |
| `types.py` | The shape of one search result, shared by both searches |
| `build.py` | Runs both, and prints how many passages went in |

**The clever bit is in `lexical.py`.** Normal word-splitters break `ND-1042`
into `nd` and `1042`, so searching for the whole code finds nothing. This one
keeps `nd-1042` **and** `nd` **and** `1042`, so either way of asking works.

### `src/retrieve/` — finding the right passages

| File | What it does |
|---|---|
| `rrf.py` | Merges the two search results into one list |
| `code_boost.py` | Lifts any passage that literally contains a code like `ND-1042` |
| `rerank.py` | Reads the question and each passage *together* and scores the fit |
| `pipeline.py` | **Runs all of the above in the right order** |

**Two orderings in `pipeline.py` are load-bearing**, meaning the system breaks
if you swap them:

- The **code boost comes before** the re-ranker. The re-ranker has no special
  respect for `ND-1042`; every row of an error table looks equally related to
  it. Exactness must be applied while it still counts.
- The **relevance floor comes after** the re-ranker, because only the
  re-ranker's score is calibrated enough to decide "nothing here is relevant".

### `src/chat/` — producing the answer

| File | What it does |
|---|---|
| `smalltalk.py` | Answers "hello" instantly, with no search and no model |
| `rewrite.py` | Turns "and for sick leave?" into a question that stands alone |
| `prompts.py` | **The exact text sent to the model.** Every sentence here was added because of a real failure |
| `llm.py` | Talks to Ollama and streams the answer back word by word |
| `backend.py` | **The switch.** Sends each model call to `llm.py` or to `lc.py`, depending on `USE_LANGCHAIN` in `.env` |
| `lc.py` | The same two model calls written with **LangChain** (`ChatPromptTemplate \| ChatOllama`). Not loaded unless the switch is on |
| `memory.py` | Remembers the last 6 turns, and saves conversations to disk |
| `engine.py` | **The conductor.** Calls everything above in order |

### LangChain: what it is used for here, and what it is not

Set `USE_LANGCHAIN=1` in `.env` and restart. `GET /health` then reports
`"backend": "langchain"`.

| Part | With the switch on |
|---|---|
| The two model calls (answer, rewrite) | **LangChain** — `ChatPromptTemplate \| ChatOllama` |
| Prompt text, history, fencing | unchanged (`prompts.py`) |
| Search, merge, re-rank, floor, guardrails | unchanged — still this project's own code |
| Events the browser receives, the Pipeline page | identical |
| Cloud tracing (LangSmith) | **forced off** — nothing leaves the machine |

Retrieval is deliberately not LangChain. A LangChain retriever returns only a
list of documents. Per-passage numbers (a score, a rank) would fit in each
document's `metadata`, but the Pipeline page also shows things that belong to no
single passage — the query tokens, the shortlist rules, the relevance-floor
verdict, the time of every step — and a retriever call has no place to report
them. Keeping them would mean routing them around the retriever.

**`engine.py` is the heart of the project** (585 lines, the largest Python
file). One question goes in, and a stream of small messages comes out:

| Message | Meaning |
|---|---|
| `stage` | "I am now searching" — so a 30-second wait does not look like a crash |
| `trace` | the raw data of one step, for the Pipeline view |
| `token` | one piece of the answer |
| `context` | how full the context window was |
| `done` | **finished successfully** |
| `error` | something broke |

**The rule that governs this file:** a stream that ends without `done` is a
**failure**, never an answer. A broken pipeline that quietly says "I could not
find that" would be the worst possible bug for a tool whose whole promise is
that you can trust it.

### `src/api/` — the web server and the page

| File | Lines | What it does |
|---|---|---|
| `app.py` | 244 | The web server. Receives questions, streams answers back |
| `preflight.py` | 117 | **8 start-up checks.** Refuses to start if a model is missing, and prints the exact fix |
| `errors.py` | 76 | Formats the streamed messages and writes the logs |
| `static/index.html` | 2,115 | The whole web page: layout, styling and chat logic |
| `static/pipeline.js` | 783 | The **Pipeline** view — every step, with its raw data |
| `static/context.js` | 125 | The **Context** view — how full the model's window was |
| `static/architecture.js` | 364 | The **Architecture** page |
| `static/accuracy.js` | 207 | The **Accuracy** page |
| `static/accuracy-data.js` | 932 | **Generated.** The measured results shown on that page |

**Note on `accuracy-data.js`:** it is written by a script, never by hand. That
is deliberate — the Accuracy page can only ever show real measurements.

### `eval/` — measuring how good it is

| File | What it does | Time |
|---|---|---|
| `eval_retrieval.py` | Does the search find the right document? **No model used** | ~6 min |
| `eval_answers.py` | All 40 questions, answered and graded | ~30 min |
| `eval_conversations.py` | The 3 multi-turn conversations | ~8 min |
| `calibrate_floor.py` | Works out the "nothing is relevant" threshold | — |
| `report.py` | Turns the results into a readable report | — |
| `build_accuracy_data.py` | Builds the data for the Accuracy page | — |
| `run_on_copy.py` | Runs any of the above **while the server is running** | — |

**`run_on_copy.py` exists because of a real limitation.** The meaning index
allows only one program at a time. Without this, you must stop the server to
measure anything. It copies the index first, so both can run.

**`eval_answers.py` scores three outcomes, not two:** correct, incorrect, and
**error**. If any question hit a technical error, the whole run is thrown away.
Otherwise a crashed server would look like a wrong answer, and you would spend
hours "fixing" a prompt on a broken machine.

### `tests/` — 281 automatic checks

| File | Checks | What it protects |
|---|---|---|
| `test_smalltalk.py` | 85 | "hello" is a greeting; "how do I claim?" is not |
| `test_guardrails.py` | 33 | The attack is removed; refusals stay refusals |
| `test_memory.py` | 28 | 6 turns means 6 questions **and** their answers |
| `test_retrieve.py` | 26 | `ND-1042` ranks first; Bangla documents are found |
| `test_ingest.py` | 25 | Tables are never cut in half |
| `test_trace.py` | 23 | The Pipeline view shows what really happened |
| `test_errors.py` | 17 | **A crash is never reported as "I don't know"** |
| `test_context.py` | 12 | The context numbers add up to the real total |
| `test_sse_format.py` | 9 | The browser can actually read the stream |
| `test_preflight.py` | 9 | Start-up checks catch a missing model |
| `test_accuracy_data.py` | 8 | The Accuracy page cannot show stale numbers |
| `test_mount.py` | 6 | The LAN address works |

---

## 4. How one question travels through the code

Follow "How many days of annual leave do I get?" through the files:

```
BROWSER   index.html              you press Enter
   │                              sends the question to the server
   ▼
SERVER    api/app.py              receives it, opens a stream back
   │
   ▼
          chat/engine.py          ─── the conductor from here on ───
   │
   ├─ 1 ─ chat/smalltalk.py       a greeting? no → carry on
   ├─ 2 ─ ingest/sanitise.py      asking for hidden instructions? no → carry on
   ├─ 3 ─ chat/memory.py          fetch the last few turns
   ├─ 4 ─ chat/rewrite.py         does it stand alone? yes → leave it
   │
   ├─ 5 ─ retrieve/pipeline.py    ─── searching from here on ───
   │        ├── index/lexical.py    search by words     → top 20
   │        ├── index/vector.py     search by meaning   → top 20
   │        ├── retrieve/rrf.py     merge the two lists
   │        ├── retrieve/code_boost.py  protect exact codes
   │        └── retrieve/rerank.py  re-read each one    → best 5
   │
   ├─ 6 ─ chat/engine.py          is the best one relevant enough?
   ├─ 7 ─ chat/prompts.py         build the exact text for the model
   ├─ 8 ─ chat/backend.py         send it to Ollama (llm.py, or lc.py if USE_LANGCHAIN=1)
   ├─ 9 ─ chat/engine.py          which documents did the answer cite?
   └─10 ─ chat/memory.py          save the conversation
   │
   ▼
BROWSER   index.html              words appear as they arrive
          pipeline.js             the Pipeline view fills in live
          context.js              the Context button appears at the end
```

**Where the time goes.** Steps 1–7 take under a second in total. Step 8 takes
almost the whole wait, because the model must *read* roughly 1,400 tokens
before it can write the first word. On a computer with no graphics card, that
reading is the slow part.

---

## 5. The generated files in `storage/`

| File | Size | What it holds | Made by |
|---|---|---|---|
| `chunks.json` | 56 KB | all 50 passages, as text | `python -m src.ingest` |
| `bm25.pkl` | 64 KB | the word index | `python -m src.index` |
| `qdrant/` | 1.1 MB | the meaning index (1,024 numbers per passage) | `python -m src.index` |
| `models/` | ~1.1 GB | the downloaded re-ranker | first use, automatic |
| `conversations.json` | grows | saved chats | the server, as you use it |
| `sanitised.log` | tiny | what the attack-stripper removed | `python -m src.ingest` |
| `requests.log` | grows | one line per question asked | the server |

**The passage text is stored three times** — in `chunks.json`, inside
`bm25.pkl`, and inside `qdrant/`. That is on purpose: each search can then
return a complete passage by itself, with no second lookup. At 1.2 MB total the
duplication costs nothing. The real risk is the three copies drifting apart,
which is why the two build commands are always run together.

---

## 6. "I want to change X" — which file?

| You want to… | Edit this | Then run |
|---|---|---|
| Change a setting (passages, timeouts, window size) | `src/config.py` or `.env` | restart the server |
| Change what the assistant is told to do | `src/chat/prompts.py` | restart, then re-run the full eval |
| Add a greeting it should recognise | `src/chat/smalltalk.py` | restart |
| Change how documents are cut up | `src/ingest/chunk.py` | **both** build commands |
| Change the look of the page | `src/api/static/index.html` | just reload the browser |
| Change the Pipeline view | `src/api/static/pipeline.js` | just reload the browser |
| Add or change a document | `rag_chatbot_kit/data/` *(normally off-limits)* | **both** build commands |

**One rule above all the others:** if you change anything that can affect an
answer — the prompts, the chunking, the models, the retrieval settings — the
measured score **no longer applies** until you re-run `eval_answers.py` and
look at the result. The Accuracy page knows this and warns you on screen when
the code has changed since the last measurement.

---

## 7. Why the code is shaped this way

Three ideas explain most of the structure:

**1. Each piece does one job.** Searching does not know about models. The model
code does not know about searching. `engine.py` is the only file that knows the
order of events. That means you can change the re-ranker without touching
anything else.

**2. Failures are loud when building, quiet when answering.** Building an index
stops dead on a problem, because a silently broken index poisons every later
answer. Answering a question does the opposite: if the re-ranker fails, it
falls back to the merged order and tells you on screen. Someone is waiting, so
a slightly worse answer beats an error page.

**3. Measured, not guessed.** Every number in the documentation came from
running something. Where a number is an estimate, the text says so. That
discipline is why the project can state 40/40 and also state exactly what that
claim does and does not cover.

---

*Next: [`PROJECT-GUIDE-EN.md`](PROJECT-GUIDE-EN.md) for the ideas behind the
code, or [`CLAUDE.md`](CLAUDE.md) for the measurements and the traps.*
