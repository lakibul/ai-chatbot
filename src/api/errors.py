"""SSE frame helpers and the request/error logs.

Plan M4.5, "the streaming trap": once SSE has sent HTTP 200 and the first
byte, it is too late to send an HTTP error code. A mid-stream failure cannot
be a 500, so the protocol carries its own frames:

    event: stage   data: {"stage":"reranking"}
    event: token   data: {"text":"18"}
    event: error   data: {"code":"llm_timeout","message":"..."}
    event: done    data: {"citations":[...]}
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime
from typing import Any

from .. import config

# Human-readable text for each failure code. The UI renders these in an
# inline error bubble -- never as an answer, and never as "I don't know".
ERROR_MESSAGES = {
    "llm_unavailable": "The local model is not running. Start it with `ollama serve`.",
    "llm_timeout": "The local model did not finish in time.",
    "llm_bad_response": "The local model returned an unusable response.",
    "retrieval_failed": "Could not search the documents.",
    "empty_question": "Please type a question.",
    "empty_answer": "The model returned an empty answer.",
    "incomplete_stream": "The answer ended unexpectedly and may be incomplete.",
    "busy": "One question at a time on this hardware. Please wait for the current answer.",
    "unexpected": "Something went wrong.",
}


def sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    """A frame for sse_starlette's EventSourceResponse."""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def error_frame(code: str, message: str = "", **extra: Any) -> dict[str, str]:
    return sse("error", {"code": code,
                         "message": message or ERROR_MESSAGES.get(code, code),
                         **extra})


def log_request(request_id: str, question: str, outcome: str,
                timings: dict[str, Any] | None = None,
                extra: dict[str, Any] | None = None) -> None:
    """One structured line per request.

    Plan M4.5 observability: the per-stage timings are what turn section 5's
    estimates into facts, and what tell us whether a slow answer was prefill
    or a stalled reranker.
    """
    config.ensure_storage()
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "request_id": request_id,
        "question": question[:300],
        "outcome": outcome,
        **(timings or {}),
        **(extra or {}),
    }
    with config.REQUESTS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_exception(request_id: str, exc: BaseException) -> None:
    config.ensure_storage()
    with config.ERRORS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"--- {datetime.now().isoformat(timespec='seconds')} "
                     f"request_id={request_id}\n")
        handle.write("".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)))
        handle.write("\n")
