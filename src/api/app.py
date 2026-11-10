"""FastAPI server with SSE streaming.

Plan 2.7: FastAPI plus one vanilla HTML page. No build step, no Node
toolchain, no Streamlit (which re-runs the whole script per interaction).

Plan M4.5, concurrency -- "one brain, one question". OLLAMA_NUM_PARALLEL=1 and
4 cores mean two simultaneous questions do not go twice as fast, they go half
as fast each. A single slot, a short wait, then an explicit busy message.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from .. import config
from ..chat.engine import ChatEngine
from ..chat.llm import LLMUnavailable, list_models
from ..chat.memory import ConversationStore
from ..retrieve.pipeline import Retriever
from .errors import error_frame, log_exception, log_request, sse
from .preflight import assert_ready, run_preflight

STATIC_DIR = Path(__file__).resolve().parent / "static"

# One slot. Qdrant local mode also takes an exclusive lock, so a single
# Retriever per process is a hard constraint, not just a tuning choice.
_slot = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

state: dict[str, Any] = {"retriever": None, "engine": None,
                         "store": ConversationStore()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checks = run_preflight()
    for check in checks:
        print(check.render())
    assert_ready(checks)                 # refuse to start, loudly

    retriever = Retriever()
    state["retriever"] = retriever
    state["engine"] = ChatEngine(retriever)
    print(f"Ready: {len(retriever.bm25)} chunks indexed. "
          f"http://127.0.0.1:8000")
    try:
        yield
    finally:
        retriever.close()


app = FastAPI(title="Nodi Tech RAG Chatbot", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(default="", max_length=20000)
    conversation_id: str = Field(default="default", max_length=128)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    """Plan M4.5: Ollama reachability, loaded models, chunk count, freshness."""
    retriever: Retriever | None = state.get("retriever")
    try:
        models = list_models()
        ollama_ok = True
    except LLMUnavailable as exc:
        models, ollama_ok = [str(exc)], False

    body = {
        "ok": ollama_ok and retriever is not None,
        "ollama": {"reachable": ollama_ok, "host": config.OLLAMA_HOST,
                   "models": models},
        "index": {
            "chunks": len(retriever.bm25) if retriever else 0,
            "collection": config.COLLECTION_NAME,
            "qdrant_path": str(config.QDRANT_PATH),
            "built_at": (config.CHUNKS_PATH.stat().st_mtime
                         if config.CHUNKS_PATH.exists() else None),
        },
        "models": {"llm": config.LLM_MODEL, "embed": config.EMBED_MODEL,
                   "rewrite": config.REWRITE_MODEL,
                   "reranker": config.RERANKER_MODEL},
        "conversations": len(state["store"]),
    }
    return JSONResponse(body, status_code=200 if body["ok"] else 503)


@app.post("/reset")
async def reset(payload: AskRequest) -> JSONResponse:
    state["store"].reset(payload.conversation_id)
    return JSONResponse({"ok": True})


@app.post("/ask")
async def ask(payload: AskRequest, request: Request) -> EventSourceResponse:
    request_id = uuid.uuid4().hex[:12]
    question = (payload.question or "").strip()

    async def event_stream():
        # --- the single slot ------------------------------------------
        try:
            await asyncio.wait_for(_slot.acquire(),
                                   timeout=config.QUEUE_WAIT_SECONDS)
        except asyncio.TimeoutError:
            log_request(request_id, question, "busy")
            yield error_frame("busy")
            return

        engine: ChatEngine = state["engine"]
        conversation = state["store"].get(payload.conversation_id)
        outcome = "incomplete"
        timings: dict[str, Any] = {}
        saw_done = False

        generator = engine.ask(question, conversation)
        try:
            async for kind, data in iterate_in_threadpool(generator):
                # If the browser went away, stop pulling. Closing the
                # generator unwinds httpx's stream context manager, which
                # aborts the upstream Ollama request -- otherwise an orphaned
                # generation blocks the next real question (plan R13).
                if await request.is_disconnected():
                    outcome = "client_disconnected"
                    break

                if kind == "error":
                    outcome = f"error:{data.get('code', 'unknown')}"
                    yield error_frame(data.get("code", "unexpected"),
                                      data.get("message", ""))
                    continue
                if kind == "done":
                    saw_done = True
                    outcome = "ok"
                    timings = data.get("timings", {})
                yield sse(kind, data)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:                  # noqa: BLE001
            log_exception(request_id, exc)
            outcome = "error:unexpected"
            yield error_frame("unexpected", str(exc))
        finally:
            # Close the generator explicitly rather than waiting for GC.
            # GeneratorExit unwinds httpx's `with stream(...)` block, which
            # closes the upstream connection and makes Ollama abandon the
            # generation. Without this, an orphaned generation keeps a core
            # busy and blocks the next real question for ~a minute (R13).
            generator.close()
            _slot.release()
            log_request(request_id, question, outcome, timings,
                        {"done_frame": saw_done,
                         "conversation": payload.conversation_id})

    return EventSourceResponse(
        event_stream(),
        ping=int(config.SSE_HEARTBEAT_SECONDS),   # keep proxies from killing it
    )
