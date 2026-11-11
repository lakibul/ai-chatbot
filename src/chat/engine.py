"""The chat engine: one question in, a stream of events out.

Event protocol (plan M4.5, "the streaming trap"):

    ("stage", {...})     progress, so a wait never reads as a crash
    ("token", {...})     a piece of the answer
    ("error", {...})     a technical failure, mid-stream if need be
    ("done",  {...})     citations and timings -- the ONLY success signal

A stream that ends without a `done` event is a FAILURE, never a complete
answer. Otherwise a dropped connection looks like a short, confident,
truncated answer -- the worst possible failure mode for a citation bot (R12).

The governing rule of M4.5 is enforced here: a technical failure must never be
reported as "I don't know". Refusals travel as ordinary tokens; failures
travel as `error` events. They cannot be confused because they are different
event types.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from .. import config
from ..index.types import Hit
from ..retrieve.pipeline import RetrievalResult, Retriever
from .llm import LLMError, chat_stream
from .memory import Conversation
from ..ingest.sanitise import find_injections
from .prompts import (NO_ANSWER_TEXT, PROMPT_EXTRACTION_REFUSAL,
                      build_messages)
from .rewrite import rewrite_query
from .smalltalk import smalltalk_reply

Event = tuple[str, dict[str, Any]]


@dataclass
class Timings:
    rewrite: float = 0.0
    retrieve: float = 0.0
    prefill: float = 0.0
    generate: float = 0.0
    total: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"rewrite_s": round(self.rewrite, 2),
                "retrieve_s": round(self.retrieve, 2),
                "prefill_s": round(self.prefill, 2),
                "generate_s": round(self.generate, 2),
                "total_s": round(self.total, 2), **self.extra}


def display_title(raw_title: str, doc_id: str) -> str:
    """A document name a person would recognise, not a filename.

    The UI shows these instead of "employee_handbook_v3.md": a filename is an
    implementation detail, and a wall of them makes a cited answer look like
    debug output rather than a reference.
    """
    title = (raw_title or "").strip()
    for prefix in ("Nodi Tech Ltd. - ", "Nodi Tech Ltd. — ",
                   "NodiDesk - ", "NodiDesk — "):
        if title.startswith(prefix):
            title = title[len(prefix):]
            break
    return title or doc_id


_CITE_MARKER = re.compile(r"\[([^\[\]\n]+?\.(?:md|pdf))\]", re.I)


def used_citations(answer: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Narrow the source list to the documents the answer actually cited.

    Retrieval hands the model five chunks; a one-line answer typically uses
    one. Listing all five makes the citation look indiscriminate, which is
    worse than useless for a tool whose whole claim is that it tells you
    where a number came from.

    If the model cited nothing recognisable, every retrieved document is
    listed rather than none: showing what it read is honest, showing nothing
    would be a silent loss of provenance.
    """
    named = {m.group(1).strip().lower() for m in _CITE_MARKER.finditer(answer or "")}
    if not named:
        return citations
    used = [c for c in citations if c["doc_id"].lower() in named]
    return used or citations


def citations_from(hits: list[Hit]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for hit in hits:
        entry = seen.setdefault(hit.doc_id, {
            "doc_id": hit.doc_id,
            "title": display_title(hit.payload.get("doc_title", ""), hit.doc_id),
            "sections": [],
            "status": hit.status,
            "score": round(hit.score, 4),
        })
        if hit.section and hit.section not in entry["sections"]:
            entry["sections"].append(hit.section)
    return list(seen.values())


class ChatEngine:
    def __init__(self, retriever: Retriever, model: str | None = None):
        self.retriever = retriever
        self.model = model or config.LLM_MODEL

    # ------------------------------------------------------------ helpers --
    def _below_floor(self, result: RetrievalResult) -> bool:
        """Plan 4.6. The floor is only meaningful on a calibrated reranker
        score, so it is skipped when the reranker was unavailable -- a
        degraded run must not start refusing in-scope questions."""
        if config.RELEVANCE_FLOOR <= 0:
            return False
        if result.stages.get("ranking") != "rerank":
            return False
        return result.top_score < config.RELEVANCE_FLOOR

    # --------------------------------------------------------------- ask ---
    def ask(self, question: str, conversation: Conversation | None = None,
            model: str | None = None) -> Iterator[Event]:
        started = time.time()
        timings = Timings()
        history = conversation.window() if conversation else []

        question = (question or "").strip()
        if not question:
            yield "error", {"code": "empty_question",
                            "message": "Please type a question."}
            return
        if len(question) > config.MAX_INPUT_CHARS:
            # An unbounded paste would blow num_ctx and push the real
            # documents out of the prompt (plan M4.5, input validation).
            question = question[:config.MAX_INPUT_CHARS]
            yield "stage", {"stage": "truncated",
                            "message": f"Question truncated to "
                                       f"{config.MAX_INPUT_CHARS} characters."}

        # --- conversational fast path -------------------------------------
        # Measured: "hello" cost 27.6s, of which 21.3s was the model reading
        # five retrieved policy chunks it had no use for. Answered here, it
        # costs nothing. Only fires when the WHOLE message is a pleasantry.
        canned = smalltalk_reply(question)
        if canned is not None:
            yield "token", {"text": canned}
            if conversation is not None:
                conversation.add_user(question, ts=started)
                conversation.add_assistant(canned)
            timings.total = time.time() - started
            yield "done", {"citations": [], "timings": timings.as_dict(),
                           "reason": "smalltalk"}
            return

        # --- prompt-extraction guard (plan M5) ----------------------------
        # Checked on the QUESTION, using the same patterns that strip
        # injections from documents at ingest. Answered before retrieval and
        # before the LLM: the model cannot leak what it is never asked, and
        # the refusal costs no prefill. This is a refusal, not a failure, so
        # it travels as tokens and ends with a proper done frame.
        if find_injections(question):
            yield "token", {"text": PROMPT_EXTRACTION_REFUSAL}
            yield "done", {"citations": [], "timings": timings.as_dict(),
                           "reason": "prompt_extraction_refused"}
            return

        # --- rewrite ------------------------------------------------------
        yield "stage", {"stage": "rewriting"}
        mark = time.time()
        search_query, was_rewritten = rewrite_query(question, history)
        timings.rewrite = time.time() - mark
        if was_rewritten:
            yield "stage", {"stage": "rewritten", "query": search_query}

        # --- retrieve -----------------------------------------------------
        yield "stage", {"stage": "searching"}
        mark = time.time()
        try:
            result = self.retriever.retrieve(search_query)
        except Exception as exc:                  # noqa: BLE001
            yield "error", {"code": "retrieval_failed",
                            "message": f"Search failed: {exc}"}
            return
        timings.retrieve = time.time() - mark

        # What the search actually did, for the step inspector. Counts only --
        # no filenames, which belong to the sources list, not to progress.
        yield "stage", {
            "stage": "searched",
            "lexical": result.stages.get("bm25_count", 0),
            "vector": result.stages.get("vector_count", 0),
            "fused": result.stages.get("fused_count", 0),
            "boosted": result.stages.get("code_boosted", 0),
            "candidates": result.stages.get("candidates", 0),
            "ranking": result.stages.get("ranking", ""),
            # Per-sub-step milliseconds, so the inspector can show which part
            # of the search cost the time instead of one opaque total.
            "ms": {
                "bm25": result.stages.get("bm25_ms"),
                "vector": result.stages.get("vector_ms"),
                "fuse": result.stages.get("fuse_ms"),
                "rerank": result.stages.get("rerank_ms"),
            },
        }

        for note in result.degraded:
            yield "stage", {"stage": "degraded", "message": note}

        if not result.hits:
            # Nothing found at all. A DISTINCT string from the refusal, so the
            # eval can tell "retrieval returned nothing" from "the documents
            # do not cover this" (plan M4.5).
            yield "token", {"text": "I could not find anything in the Nodi Tech "
                                    "documents for that question."}
            yield "done", {"citations": [], "timings": timings.as_dict(),
                           "reason": "no_results"}
            return

        if self._below_floor(result):
            yield "token", {"text": NO_ANSWER_TEXT}
            yield "done", {"citations": [], "timings": timings.as_dict(),
                           "reason": "below_relevance_floor",
                           "top_score": round(result.top_score, 4)}
            return

        # The count, plus the DOCUMENT NAMES being read -- not filenames.
        # These are only visible inside the collapsed step inspector, where
        # knowing what is being consulted during a 40s wait is the point.
        reading_titles: list[str] = []
        for hit in result.hits:
            title = display_title(hit.payload.get("doc_title", ""), hit.doc_id)
            if title not in reading_titles:
                reading_titles.append(title)
        yield "stage", {"stage": "reading", "sources": len(result.hits),
                        "titles": reading_titles}

        # --- generate -----------------------------------------------------
        messages = build_messages(question, result.hits, history)
        yield "stage", {"stage": "writing", "model": model or self.model}

        mark = time.time()
        answer_parts: list[str] = []
        metrics: dict[str, Any] = {}
        upstream_completed = False
        try:
            for kind, text, payload in chat_stream(
                    messages, model=model or self.model,
                    num_predict=config.ANSWER_MAX_TOKENS):
                if kind == "token":
                    answer_parts.append(text)
                    yield "token", {"text": text}
                elif kind == "done":
                    metrics = payload
                    upstream_completed = True
        except LLMError as exc:
            # Mid-stream failure. It is far too late for an HTTP 500, so the
            # protocol carries its own error frame (plan M4.5).
            yield "error", {"code": getattr(exc, "code", "llm_error"),
                            "message": str(exc)}
            return
        except Exception as exc:                  # noqa: BLE001
            yield "error", {"code": "unexpected", "message": str(exc)}
            return

        if not upstream_completed:
            # Ollama's stream ended without its own done marker. Emitting our
            # `done` here would dress a truncated answer up as a complete one
            # -- the worst failure mode for a citation bot (plan R12).
            yield "error", {
                "code": "incomplete_stream",
                "message": "The model stopped before finishing its answer.",
                "partial_characters": len("".join(answer_parts)),
            }
            return

        stream_seconds = time.time() - mark
        timings.prefill = float(metrics.get("prefill_seconds", 0.0) or 0.0)
        # Report generation as time spent AFTER the prompt was read, so the
        # two numbers add up instead of overlapping.
        timings.generate = max(stream_seconds - timings.prefill, 0.0)
        timings.total = time.time() - started
        timings.extra = {
            "prompt_tokens": metrics.get("prompt_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "ranking": result.stages.get("ranking", ""),
        }

        answer = "".join(answer_parts).strip()
        if not answer:
            yield "error", {"code": "empty_answer",
                            "message": "The model returned an empty answer."}
            return

        citations = used_citations(answer, citations_from(result.hits))
        if conversation is not None:
            conversation.add_user(question, ts=started)
            conversation.add_assistant(answer, citations)

        yield "done", {"citations": citations,
                       "timings": timings.as_dict(),
                       "degraded": result.degraded}

    # ------------------------------------------------------ convenience ---
    def ask_sync(self, question: str, conversation: Conversation | None = None,
                 model: str | None = None) -> dict[str, Any]:
        """Collect the whole stream. Used by the eval harness and tests.

        Returns {"answer", "citations", "error", "timings"}. `error` is None
        on success, and a dict on failure -- the eval scores that as ERROR,
        never as an incorrect answer (plan M6, three outcomes not two).
        """
        parts: list[str] = []
        citations: list[dict[str, Any]] = []
        timings: dict[str, Any] = {}
        error: dict[str, Any] | None = None
        stages: list[dict[str, Any]] = []
        done = False

        for kind, payload in self.ask(question, conversation, model=model):
            if kind == "token":
                parts.append(payload["text"])
            elif kind == "stage":
                stages.append(payload)
            elif kind == "error":
                error = payload
            elif kind == "done":
                done = True
                citations = payload.get("citations", [])
                timings = payload.get("timings", {})

        if error is None and not done:
            # No done event: treat as failed, never as a complete answer (R12).
            error = {"code": "incomplete_stream",
                     "message": "The answer stream ended without completing."}

        return {"answer": "".join(parts).strip(), "citations": citations,
                "timings": timings, "error": error, "stages": stages}
