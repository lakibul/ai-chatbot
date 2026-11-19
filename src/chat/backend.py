"""The one switch between the hand-written Ollama client and LangChain.

The engine and the rewriter import ``chat`` and ``chat_stream`` from HERE, not
from llm.py. Each call looks at ``config.USE_LANGCHAIN`` at call time, so the
choice is one line in .env and a restart, and tests can flip it.

Both backends honour the same contract (see llm.py), which is what makes the
switch safe: the events the browser receives are identical either way.

lc.py is imported lazily. With the flag off, LangChain is never loaded, so the
default path is exactly the code that produced the audited 40/40.
"""
from __future__ import annotations

from collections.abc import Iterator

from .. import config
from . import llm
from .llm import ChatResult


def name() -> str:
    """Which backend is active -- shown by /health so it is never a guess."""
    return "langchain" if config.USE_LANGCHAIN else "native"


def chat(messages: list[dict[str, str]], model: str | None = None,
         num_predict: int | None = None, timeout: float | None = None,
         options: dict | None = None) -> ChatResult:
    if config.USE_LANGCHAIN:
        from . import lc
        return lc.chat(messages, model=model, num_predict=num_predict,
                       timeout=timeout, options=options)
    return llm.chat(messages, model=model, num_predict=num_predict,
                    timeout=timeout, options=options)


def chat_stream(messages: list[dict[str, str]], model: str | None = None,
                num_predict: int | None = None, timeout: float | None = None,
                options: dict | None = None) -> Iterator[tuple[str, str, dict]]:
    if config.USE_LANGCHAIN:
        from . import lc
        yield from lc.chat_stream(messages, model=model, num_predict=num_predict,
                                  timeout=timeout, options=options)
        return
    yield from llm.chat_stream(messages, model=model, num_predict=num_predict,
                               timeout=timeout, options=options)
