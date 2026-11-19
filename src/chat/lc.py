"""LangChain backend for the two model calls (answer and rewrite).

Same contract as llm.py -- ``chat`` returns a ChatResult, ``chat_stream``
yields ("token", text, {}) then ("done", "", metrics) -- so the engine cannot
tell which one it is using. It is chosen by ``config.USE_LANGCHAIN`` in
backend.py; nothing here is imported when the flag is off.

What LangChain is used for, and what it is NOT used for:

  USED      ChatPromptTemplate | ChatOllama as one LCEL chain; message objects;
            streaming through the chain.
  NOT USED  retrieval. A LangChain retriever returns only a list of Documents.
            Per-passage numbers (scores, ranks) would fit in each Document's
            metadata, but the Pipeline page also shows things that belong to
            no single passage -- the query tokens, the shortlist rules, the
            relevance-floor verdict, the milliseconds of every step -- and a
            retriever call has no place to report them. Keeping them would
            mean routing them AROUND the retriever, so retrieval stays the
            project's own code. (Checked: tests/test_langchain.py and the
            LangChain chapter of PROJECT-GUIDE-BN.md.)

LangSmith (LangChain's cloud tracing) is forced OFF below. It would send every
question and every retrieved passage to a third party, and this project's
promise is that nothing leaves the machine.
"""
from __future__ import annotations

import os

# Forced, not setdefault: a stray variable in the user's shell must not be able
# to switch cloud tracing on. Set BEFORE any langchain import reads the env.
for _name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
    os.environ[_name] = "false"

from collections.abc import Iterator          # noqa: E402
from functools import lru_cache               # noqa: E402
from typing import Any                        # noqa: E402

import httpx                                  # noqa: E402
import ollama                                 # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: E402
from langchain_ollama import ChatOllama       # noqa: E402

from .. import config                         # noqa: E402
from .llm import (ChatResult, LLMBadResponse, LLMTimeout,  # noqa: E402
                  LLMUnavailable, _result_from)

# The prompt is already fully assembled by prompts.build_messages (system
# prompt, history, fenced documents, question, reminder). The template only
# carries it through the chain; it must not re-parse the text, which contains
# braces and angle brackets. A MessagesPlaceholder passes message objects
# through untouched.
_PROMPT = ChatPromptTemplate.from_messages([MessagesPlaceholder("messages")])

_ROLES = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}


def to_messages(messages: list[dict[str, str]]) -> list:
    """{"role", "content"} dicts -> LangChain message objects, order kept."""
    try:
        return [_ROLES[m["role"]](content=m["content"]) for m in messages]
    except KeyError as exc:
        raise LLMBadResponse(f"unknown message role: {exc}") from exc


_OPTION_KEYS = ("num_gpu", "num_ctx", "num_thread", "temperature", "num_predict")


@lru_cache(maxsize=16)
def _model(model: str, timeout: float, options: tuple) -> ChatOllama:
    """One ChatOllama per (model, timeout, options), reused across requests.

    Every option mirrors config.ollama_options + llm._payload. They are not
    cosmetic: ``num_gpu=0`` stops Ollama trying the unusable 2 GB GPU, and
    ``reasoning=False`` is Ollama's ``think: false`` -- without it qwen3:1.7b
    spends the whole rewrite budget "thinking" and returns nothing.
    """
    o = dict(options)
    return ChatOllama(
        model=model,
        base_url=config.OLLAMA_HOST,
        num_gpu=o["num_gpu"],
        num_ctx=o["num_ctx"],
        num_thread=o["num_thread"],
        temperature=o["temperature"],
        num_predict=o.get("num_predict"),
        keep_alive=config.KEEP_ALIVE,
        reasoning=False,
        client_kwargs={"timeout": httpx.Timeout(timeout, connect=config.CONNECT_TIMEOUT)},
    )


def _chain(model: str, num_predict: int | None, timeout: float | None,
           options: dict | None = None):
    merged = config.ollama_options(num_predict)
    if options:
        unknown = set(options) - set(_OPTION_KEYS)
        if unknown:                       # never drop a setting silently
            raise ValueError(f"unsupported Ollama option(s) for LangChain: {sorted(unknown)}")
        merged.update(options)
    return _PROMPT | _model(model, timeout or config.GENERATE_TIMEOUT,
                            tuple(sorted(merged.items())))


def _translate(exc: Exception, model: str, timeout: float | None) -> Exception:
    """Map the library's exceptions onto the project's, so the engine's
    ``except LLMError`` and its error codes keep working unchanged."""
    if isinstance(exc, (httpx.TimeoutException,)):
        return LLMTimeout(f"{model}: no response within "
                          f"{timeout or config.GENERATE_TIMEOUT:.0f}s")
    if isinstance(exc, ollama.ResponseError):
        return LLMBadResponse(f"{model}: Ollama returned {exc.status_code} -- "
                              f"{str(exc)[:200]}")
    if isinstance(exc, (ConnectionError, httpx.RequestError)):
        return LLMUnavailable(f"cannot reach Ollama at {config.OLLAMA_HOST} ({exc}). "
                              f"Start it with: ollama serve")
    return exc


def _metadata(message: Any) -> dict:
    return dict(getattr(message, "response_metadata", None) or {})


def chat(messages: list[dict[str, str]], model: str | None = None,
         num_predict: int | None = None, timeout: float | None = None,
         options: dict | None = None) -> ChatResult:
    """One non-streaming turn (rewriter, token counting)."""
    model = model or config.LLM_MODEL
    chain = _chain(model, num_predict, timeout, options)
    payload = {"messages": to_messages(messages)}
    last: Exception | None = None
    for attempt in range(2):
        try:
            reply = chain.invoke(payload)
            return _result_from({"message": {"content": reply.content},
                                 **_metadata(reply)}, model)
        except Exception as exc:                          # noqa: BLE001
            mapped = _translate(exc, model, timeout)
            # Same rule as llm.chat: never retry a timeout, retry a refused
            # connection once (Ollama is usually just swapping models).
            if isinstance(mapped, LLMUnavailable) and attempt == 0:
                last = mapped
                continue
            if mapped is exc:
                raise
            raise mapped from exc
    raise last  # pragma: no cover  (both attempts unavailable)


def chat_stream(messages: list[dict[str, str]], model: str | None = None,
                num_predict: int | None = None, timeout: float | None = None,
                options: dict | None = None) -> Iterator[tuple[str, str, dict]]:
    """Yield ("token", text, {}) per chunk, then ("done", "", metrics).

    A generator on purpose: when the browser disconnects the caller closes it,
    which unwinds LangChain's stream and the HTTP request underneath, so an
    orphaned generation cannot block the next question (plan R13).

    A stream that ends without Ollama's ``done`` flag yields no "done" at all,
    exactly like llm.chat_stream -- the engine treats that as a FAILURE, never
    as a finished answer.
    """
    model = model or config.LLM_MODEL
    chain = _chain(model, num_predict, timeout, options)
    try:
        for chunk in chain.stream({"messages": to_messages(messages)}):
            if chunk.content:
                yield "token", chunk.content, {}
            meta = _metadata(chunk)
            if meta.get("done"):
                yield "done", "", _result_from(
                    {"message": {"content": chunk.content}, **meta}, model).__dict__
                return
    except GeneratorExit:
        raise
    except Exception as exc:                              # noqa: BLE001
        mapped = _translate(exc, model, timeout)
        if mapped is exc:
            raise
        raise mapped from exc
