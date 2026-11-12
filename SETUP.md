# SETUP.md — commands only

Copy and paste. Windows PowerShell. For the *why* behind any step, see
[`README.md`](README.md) and [`CLAUDE.md`](CLAUDE.md).

| Part | When you need it |
|---|---|
| [1. First-time setup](#1-first-time-setup-on-a-new-device) | A device that has never run this project |
| [2. Starting and stopping](#2-starting-and-stopping) | Every day |
| [3. After `git pull`](#3-after-git-pull) | Every time you pull new code |
| [4. If something breaks](#4-if-something-breaks) | When you see an error |

---

## 1. First-time setup on a new device

About 20 minutes, mostly downloading ~6.5 GB. You need **Python 3.10 or 3.11**
(not 3.12+), **Git**, ~10 GB free disk and 16 GB RAM.

### 1.1 Check Python
```powershell
python --version
```
Must print `3.10.x` or `3.11.x`. If not, install Python 3.11 first.

### 1.2 Get the code
```powershell
git clone https://github.com/lakibul/ai-chatbot.git
cd ai-chatbot
```

### 1.3 Install Ollama and set three model slots
```powershell
winget install Ollama.Ollama
setx OLLAMA_MAX_LOADED_MODELS 3
setx OLLAMA_NUM_PARALLEL 1
setx OLLAMA_KEEP_ALIVE 30m
```
**Close PowerShell and open a new one.** `setx` only affects new windows. If
Ollama is in the system tray, right-click it and choose **Quit**.

Use two windows from here on:

| Window | What it runs |
|---|---|
| **A** | `ollama serve` — leave it open |
| **B** | everything else below |

Window A:
```powershell
ollama serve
```

### 1.4 Download the models (Window B, ~5.4 GB)
```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull bge-m3
ollama pull qwen3:1.7b-q4_K_M
ollama pull nomic-embed-text
ollama list
```
`ollama list` must show all four.

### 1.5 Python environment (Window B, inside the `ai-chatbot` folder)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```
If your default `python` is not 3.10/3.11, use that version's full path in the
first line, e.g. `"C:\laragon\bin\python\python-3.10\python.exe" -m venv .venv`.

### 1.6 Build the search index (~45 s)
```powershell
python -m src.ingest
python -m src.index
```
Expect `12 documents -> 50 chunks`, then `50 vectors ... (dim 1024)`.

### 1.7 Start it
```powershell
python -m uvicorn src.api.app:app --port 8000
```
Open <http://127.0.0.1:8000>. Eight lines starting with `[OK]` must print.

The **first question is slow**: it downloads a 1.1 GB helper model, once.

### 1.8 Check it
Stop the server (`Ctrl+C`) first, then:
```powershell
pytest
```
All tests should pass (about 230). First run 3–4 min, later runs ~75 s.

---

## 2. Starting and stopping

**Every time, in this order:**

| # | Do | Command |
|---|---|---|
| 1 | Ollama running? | `ollama list` (if it errors, run `ollama serve` in Window A) |
| 2 | Go to the project | `cd C:\ai-chatbot` |
| 3 | Activate Python | `.venv\Scripts\activate` |
| 4 | Start the server | see below |
| 5 | Stop it | `Ctrl+C` in the server window |

**This PC only:**
```powershell
python -m uvicorn src.api.app:app --port 8000
```
→ <http://127.0.0.1:8000>

**Other devices on your network** — double-click `start-lan.bat`, or:
```powershell
$env:ROOT_PATH="ai-chatbot"
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
→ `http://THIS-PC-ADDRESS:8000/ai-chatbot` (the `.bat` window prints the address)

Allow the port once, in PowerShell **as Administrator**:
```powershell
New-NetFirewallRule -DisplayName 'Nodi 8000' -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow -Profile Any
```

> The app has **no login**. Anyone who can reach the address can read every
> document. Keep it on a trusted network.

---

## 3. After `git pull`

Only **some** pulls need extra work. Follow these steps in order.

### 3.1 Before you pull
Stop the server (`Ctrl+C`). The index is locked while it runs.

### 3.2 Pull, then see what changed
```powershell
cd C:\ai-chatbot
git pull
git log --oneline ORIG_HEAD..HEAD
git diff --name-only ORIG_HEAD HEAD
```
The first list is the new commits. The second is the files that changed.
`ORIG_HEAD` means "where the project was before this pull".

### 3.3 Let a script tell you what to do
Paste this right after the pull. It reads the list of changed files and prints
only what applies to you:
```powershell
$changed = git diff --name-only ORIG_HEAD HEAD
if ($changed -match 'requirements.txt')                           { 'DO -> pip install -r requirements.txt' }
if ($changed -match 'rag_chatbot_kit/data|src/ingest/|src/index/') { 'DO -> python -m src.ingest ; python -m src.index   (rebuild the index)' }
if ($changed -match 'src/config.py')                              { 'CHECK -> a model name may have changed: run  ollama list  and compare with src/config.py' }
if ($changed -match '\.env\.example')                             { 'CHECK -> new settings: compare .env with .env.example, add the new lines' }
if ($changed -match 'src/api/static/')                            { 'DO -> after restarting, hard-refresh the browser (Ctrl+Shift+R)' }
if ($changed -match '^src/')                                      { 'DO -> restart the server' }
'(nothing more is printed = nothing more to do)'
```

### 3.4 What each line means

| What changed in the pull | What to run | Why |
|---|---|---|
| `requirements.txt` | `.venv\Scripts\activate` then `pip install -r requirements.txt` | New or updated Python packages |
| `rag_chatbot_kit/data/`, `src/ingest/`, or `src/index/` | `python -m src.ingest` then `python -m src.index` | The documents, the chunking or the embedding changed, so the old index is stale |
| `src/config.py` with a new model name | `ollama pull <the new tag>` | The server refuses to start if a model is missing and prints the exact command |
| `.env.example` | Add the new lines to your `.env` by hand | `git pull` never touches `.env`. Do **not** overwrite it |
| `src/api/static/` | Restart, then `Ctrl+Shift+R` in the browser | The browser keeps the old page, so answers can look "cut off" |
| any other `src/` file | Restart the server | Python only reads code at start-up |
| `tests/`, `*.md`, `*.bat` only | Nothing | |

### 3.5 Restart and check
```powershell
.venv\Scripts\activate
python -m uvicorn src.api.app:app --port 8000
```
Eight `[OK]` lines must print. For a full check, stop the server and run
`pytest` (~75 s).

### 3.6 If `git pull` refuses
It says "local changes would be overwritten". That is your call — commit your
changes or set them aside first. Check with:
```powershell
git status
```
Never delete `storage/` or `.venv/` to "fix" a pull. They are not in git and a
pull never touches them.

---

## 4. If something breaks

| You see | Do this |
|---|---|
| `Preflight failed. Refusing to start` | Read it. Each failing line prints its own fix |
| `cannot reach Ollama` | Start `ollama serve` in Window A |
| `model ... missing` | Run the `ollama pull` it shows |
| `chunks.json / bm25.pkl missing`, or `index agreement` fails | `python -m src.ingest` then `python -m src.index` |
| `AlreadyLocked` | Stop the server (or any other `pytest` / eval) and retry |
| `UnicodeEncodeError: 'charmap'` | `$env:PYTHONIOENCODING="utf-8"` then retry |
| `No matching distribution for onnxruntime` | Python is 3.12+. Rebuild `.venv` with 3.10 or 3.11 |
| Every answer says "cut off" | Browser cached the old page: `Ctrl+Shift+R` |
| Answers 20–40 s slower than usual | `OLLAMA_MAX_LOADED_MODELS` is not 3. Redo step 1.3 and open a **new** window |
| LAN address does not open | Run the firewall command in Part 2 as Administrator |

---

## 5. Never do these

- **Never copy `.venv/` or `storage/`** from another PC. They hold absolute paths.
  Part 1.5 and 1.6 rebuild them in minutes.
- **Never edit `rag_chatbot_kit/data/`.** It is the read-only test dataset.
- **Never run `pytest` or an `eval` script while the server is running.**
- **Never put anything that changes per question into `SYSTEM_PROMPT`.** It
  breaks the prompt cache and costs roughly 28 s on the next answer
  (676 tokens ÷ 23.9 tokens/s — a calculation, not a separate measurement).
