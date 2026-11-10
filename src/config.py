"""Single source of truth for every tunable in the pipeline.

Plan section 7: "Config, not constants: every threshold (chunk size, top_k,
RRF k, relevance floor, code-boost weight, superseded multiplier) lives in
config.py so the eval can sweep it."

Anything a human might want to change, or the eval might want to sweep,
belongs here and nowhere else.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "rag_chatbot_kit" / "data"
EVAL_DIR = ROOT / "rag_chatbot_kit" / "eval"
STORAGE_DIR = ROOT / "storage"

QDRANT_PATH = STORAGE_DIR / "qdrant"
BM25_PATH = STORAGE_DIR / "bm25.pkl"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"
SANITISED_LOG = STORAGE_DIR / "sanitised.log"
REQUESTS_LOG = STORAGE_DIR / "requests.log"
ERRORS_LOG = STORAGE_DIR / "errors.log"

EVAL_QUESTIONS = EVAL_DIR / "eval_questions.csv"
CONVERSATION_TESTS = EVAL_DIR / "conversation_tests.json"
EVAL_RESULTS_DIR = ROOT / "eval" / "results"

# The corpus is fixed and known. Ingestion asserts this exactly (plan M4.5:
# "a run that indexes 11 of 12 files exits non-zero").
EXPECTED_SOURCE_FILES = 12

# ------------------------------------------------------------- ollama ------
OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434")

# Native endpoints only. Plan 2.1: "/api/chat, never /v1/chat/completions" --
# the OpenAI-compatible path drops Ollama-specific controls.
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_EMBED_URL = f"{OLLAMA_HOST}/api/embed"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"

LLM_MODEL = _env("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
DEV_LLM_MODEL = _env("DEV_LLM_MODEL", "qwen3:1.7b-q4_K_M")
REWRITE_MODEL = _env("REWRITE_MODEL", DEV_LLM_MODEL)
EMBED_MODEL = _env("EMBED_MODEL", "bge-m3")
ALT_EMBED_MODEL = _env("ALT_EMBED_MODEL", "nomic-embed-text")

REQUIRED_MODELS = (LLM_MODEL, DEV_LLM_MODEL, EMBED_MODEL)

# Mandatory generation options (plan 2.1). These are correctness requirements
# on this hardware, not tuning preferences.
NUM_GPU = _env_int("NUM_GPU", 0)          # force CPU: a 2 GB GT 1030 cannot hold a 4B Q4
NUM_CTX = _env_int("NUM_CTX", 4096)       # the tag advertises 256K; that KV cache would OOM 16 GB
NUM_THREAD = _env_int("NUM_THREAD", 4)    # 4 physical Zen+ cores; M0 benchmarks 4 vs 8
KEEP_ALIVE = _env("KEEP_ALIVE", "30m")    # avoid a 10-20 s reload per request
TEMPERATURE = _env_float("TEMPERATURE", 0.1)   # extractive QA, not creativity

# ------------------------------------------------------------ chunking -----
# Tables are never split (plan 4.4), so these are targets for prose only.
CHUNK_TARGET_CHARS = _env_int("CHUNK_TARGET_CHARS", 1200)
CHUNK_MAX_CHARS = _env_int("CHUNK_MAX_CHARS", 2000)
CHUNK_OVERLAP_CHARS = _env_int("CHUNK_OVERLAP_CHARS", 150)

# ----------------------------------------------------------- retrieval -----
BM25_TOP_K = _env_int("BM25_TOP_K", 20)
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", 20)
RRF_K = _env_int("RRF_K", 60)
RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 20)
FINAL_TOP_K = _env_int("FINAL_TOP_K", 5)

# Plan 4.3: an exact code match outranks anything a cross-encoder believes.
# Applied BEFORE reranking.
CODE_BOOST = _env_float("CODE_BOOST", 1000.0)

# Plan 4.1: superseded chunks are demoted, never deleted -- Q29 must cite both.
SUPERSEDED_MULTIPLIER = _env_float("SUPERSEDED_MULTIPLIER", 0.5)

# Plan 4.6 / M5: calibrated by sweep against the eval set, never guessed.
# This placeholder is replaced by a measured value in M5.
RELEVANCE_FLOOR = _env_float("RELEVANCE_FLOOR", 0.0)

RERANKER_MODEL = _env("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
RERANKER_FALLBACK = _env("RERANKER_FALLBACK", "Xenova/ms-marco-MiniLM-L-6-v2")

COLLECTION_NAME = _env("COLLECTION_NAME", "nodi_docs")

# ---------------------------------------------------------------- chat -----
HISTORY_TURNS = _env_int("HISTORY_TURNS", 6)
MAX_INPUT_CHARS = _env_int("MAX_INPUT_CHARS", 2000)
REWRITE_MAX_TOKENS = _env_int("REWRITE_MAX_TOKENS", 40)
ANSWER_MAX_TOKENS = _env_int("ANSWER_MAX_TOKENS", 512)

# ------------------------------------------------------------ timeouts -----
# Sized to measured reality (plan M4.5): 60-90 s per answer is NORMAL here,
# so a tight generation timeout would kill healthy requests.
CONNECT_TIMEOUT = _env_float("CONNECT_TIMEOUT", 5.0)
EMBED_TIMEOUT = _env_float("EMBED_TIMEOUT", 30.0)
REWRITE_TIMEOUT = _env_float("REWRITE_TIMEOUT", 30.0)
GENERATE_TIMEOUT = _env_float("GENERATE_TIMEOUT", 300.0)
SSE_HEARTBEAT_SECONDS = _env_float("SSE_HEARTBEAT_SECONDS", 10.0)

# One brain, one question (plan M4.5).
MAX_CONCURRENT_REQUESTS = _env_int("MAX_CONCURRENT_REQUESTS", 1)
QUEUE_WAIT_SECONDS = _env_float("QUEUE_WAIT_SECONDS", 2.0)


def ollama_options(num_predict: int | None = None) -> dict:
    """The option block every /api/chat call must carry."""
    options = {
        "num_gpu": NUM_GPU,
        "num_ctx": NUM_CTX,
        "num_thread": NUM_THREAD,
        "temperature": TEMPERATURE,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict
    return options


def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
