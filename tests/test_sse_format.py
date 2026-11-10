"""The SSE wire contract between the server and the browser.

This file exists because of a bug that every other test missed.

`sse_starlette` terminates lines with CRLF, so frames are separated by
"\\r\\n\\r\\n". The browser client split the stream on "\\n\\n", which matches
NOTHING in that output -- so zero frames were ever parsed, no tokens rendered,
no `done` frame arrived, and every single answer displayed as "the answer was
cut off before it finished".

The server was correct throughout. curl showed the frames, and the Python
tests used httpx's `iter_lines()`, which handles CRLF transparently. Nothing
exercised the one parser that actually mattered.

So these tests pin the wire format itself, and run the browser's exact
splitting regexes against real sse_starlette bytes.
"""
from __future__ import annotations

import json
import re

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sse_starlette.sse import EventSourceResponse

from src.api.errors import error_frame, sse

# These two patterns are copied verbatim from src/api/static/index.html.
# If you change them in one place, change them in the other.
FRAME_SPLIT = re.compile(r"\r\n\r\n|\n\n|\r\r")
LINE_SPLIT = re.compile(r"\r\n|\n|\r")


def parse_like_the_browser(raw: str) -> list[tuple[str, dict]]:
    """A faithful port of parseFrame() in index.html."""
    events: list[tuple[str, dict]] = []
    for frame in FRAME_SPLIT.split(raw):
        event, data = "message", ""
        for line in LINE_SPLIT.split(frame):
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].lstrip(" ")
        if data:
            try:
                events.append((event, json.loads(data)))
            except json.JSONDecodeError:
                pass
    return events


def _render(frames: list[dict]) -> str:
    """Serve `frames` once and return the raw bytes the browser would see.

    sse_starlette keeps a module-level asyncio Event on AppStatus, bound to
    whichever event loop first touched it. A second TestClient therefore dies
    with "bound to a different event loop", so it is reset per render. This is
    a quirk of the library, not of the application.
    """
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None

    app = FastAPI()

    @app.get("/s")
    async def stream():
        async def gen():
            for frame in frames:
                yield frame
        return EventSourceResponse(gen())

    with TestClient(app) as client:
        return client.get("/s").text


FRAMES = [
    sse("stage", {"stage": "searching"}),
    sse("stage", {"stage": "reading", "sources": 5}),
    sse("token", {"text": "18"}),
    sse("token", {"text": " days."}),
    sse("done", {"citations": [{"doc_id": "employee_handbook_v3.md"}],
                 "timings": {"total_s": 42.0}}),
]


def test_sse_frames_are_crlf_separated():
    """Pin the actual wire format, so a library change is caught here."""
    raw = _render(FRAMES)
    assert "\r\n\r\n" in raw, "sse_starlette is expected to emit CRLF frames"


def test_splitting_on_lf_only_would_find_nothing():
    """The exact bug, encoded. If this ever starts passing with a naive
    split, the wire format changed and index.html should be revisited."""
    raw = _render(FRAMES)
    naive = [f for f in raw.split("\n\n") if "data:" in f]
    assert len(naive) < len(FRAMES), (
        "a naive \\n\\n split must NOT recover every frame -- that is the "
        "assumption that broke the browser client")


def test_browser_parser_recovers_every_frame():
    raw = _render(FRAMES)
    events = parse_like_the_browser(raw)
    kinds = [kind for kind, _ in events]
    assert kinds == ["stage", "stage", "token", "token", "done"], kinds


def test_browser_parser_reassembles_the_answer():
    raw = _render(FRAMES)
    events = parse_like_the_browser(raw)
    answer = "".join(p["text"] for k, p in events if k == "token")
    assert answer == "18 days."


def test_browser_parser_sees_the_done_frame_and_its_citations():
    """Without this, the UI treats a perfectly good answer as truncated."""
    raw = _render(FRAMES)
    done = [p for k, p in parse_like_the_browser(raw) if k == "done"]
    assert len(done) == 1
    assert done[0]["citations"][0]["doc_id"] == "employee_handbook_v3.md"
    assert done[0]["timings"]["total_s"] == 42.0


def test_browser_parser_ignores_ping_comments():
    """sse_starlette sends ': ping - <timestamp>' keep-alives during the
    40-70s prefill silence. They must not be mistaken for data."""
    raw = ": ping - 2026-11-10 10:58:47\r\n\r\n" + _render(FRAMES)
    events = parse_like_the_browser(raw)
    assert [k for k, _ in events] == ["stage", "stage", "token", "token", "done"]


def test_error_frames_survive_the_same_path():
    frames = [sse("token", {"text": "partial"}),
              error_frame("llm_timeout", "took too long")]
    raw = _render(frames)
    events = parse_like_the_browser(raw)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "llm_timeout"
    assert events[-1][1]["message"] == "took too long"


def test_unicode_survives_the_wire():
    """Bangla answers must arrive intact (Q32, Q33)."""
    frames = [sse("token", {"text": "সকাল ৮:১৫"}),
              sse("done", {"citations": [], "timings": {}})]
    raw = _render(frames)
    events = parse_like_the_browser(raw)
    assert events[0][1]["text"] == "সকাল ৮:১৫"


def test_error_frame_defaults_to_a_human_message():
    frame = error_frame("llm_unavailable")
    payload = json.loads(frame["data"])
    assert "ollama serve" in payload["message"]
