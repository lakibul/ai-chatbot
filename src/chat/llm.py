"""Ollama chat client.

Plan 2.1: the NATIVE /api/chat endpoint, never /v1/chat/completions. The
OpenAI-compatible path drops Ollama-specific controls (num_gpu, num_ctx,
num_thread, keep_alive) and carries the thinking-mode bugs that ruled out
qwen3.5 in the first place.

Plan M4.5 governs the error handling: a technical failure must NEVER be
reported as "I don't know". Every failure here raises an LLMError subclass,
and nothing in this module can produce a refusal string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from .. import config


class LLMError(RuntimeError):
    """Base: a technical failure, never a refusal."""
    code = "llm_error"


class LLMUnavailable(LLMError):
    """Ollama is not reachable."""
    code = "llm_unavailable"


class LLMTimeout(LLMError):
    """The model did not finish in time."""
    code = "llm_timeout"


class LLMBadResponse(LLMError):
    """Ollama answered, but not with something usable."""
    code = "llm_bad_response"


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    prefill_seconds: float = 0.0
    generation_seconds: float = 0.0
    load_seconds: float = 0.0
    total_seconds: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def prefill_tokens_per_second(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def output_tokens_per_second(self) -> float:
        return self.output_tokens / self.generation_seconds if self.generation_seconds else 0.0


def _timeouts(generate_timeout: float | None) -> httpx.Timeout:
    return httpx.Timeout(generate_timeout or config.GENERATE_TIMEOUT,
                         connect=config.CONNECT_TIMEOUT)


def _payload(messages: list[dict[str, str]], model: str, stream: bool,
             num_predict: int | None, options_override: dict | None) -> dict:
    options = config.ollama_options(num_predict)
    if options_override:
        options.update(options_override)
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": options,
        "keep_alive": config.KEEP_ALIVE,
    }


def _result_from(data: dict, model: str) -> ChatResult:
    ns = 1_000_000_000
    return ChatResult(
        text=(data.get("message") or {}).get("content", ""),
        model=model,
        prompt_tokens=data.get("prompt_eval_count", 0) or 0,
        output_tokens=data.get("eval_count", 0) or 0,
        prefill_seconds=(data.get("prompt_eval_duration", 0) or 0) / ns,
        generation_seconds=(data.get("eval_duration", 0) or 0) / ns,
        load_seconds=(data.get("load_duration", 0) or 0) / ns,
        total_seconds=(data.get("total_duration", 0) or 0) / ns,
        raw=data,
    )


def chat(messages: list[dict[str, str]], model: str | None = None,
         num_predict: int | None = None, timeout: float | None = None,
         options: dict | None = None) -> ChatResult:
    """One non-streaming turn. Used by the query rewriter, the judge and bench."""
    model = model or config.LLM_MODEL
    body = _payload(messages, model, False, num_predict, options)

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = httpx.post(config.OLLAMA_CHAT_URL, json=body,
                                  timeout=_timeouts(timeout))
            response.raise_for_status()
            return _result_from(response.json(), model)
        except httpx.TimeoutException as exc:
            # Never retry a timeout: at 60-90 s per call a blind retry doubles
            # an already-painful wait for something that will time out again.
            raise LLMTimeout(
                f"{model}: no response within "
                f"{timeout or config.GENERATE_TIMEOUT:.0f}s") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMBadResponse(
                f"{model}: Ollama returned {exc.response.status_code} "
                f"-- {exc.response.text[:200]}") from exc
        except httpx.RequestError as exc:
            last_error = exc
            if attempt == 0:
                continue            # retry once: usually Ollama swapping models
    raise LLMUnavailable(
        f"cannot reach Ollama at {config.OLLAMA_HOST} ({last_error}). "
        f"Start it with: ollama serve")


def chat_stream(messages: list[dict[str, str]], model: str | None = None,
                num_predict: int | None = None, timeout: float | None = None,
                options: dict | None = None) -> Iterator[tuple[str, str, dict]]:
    """Yield ("token", text, {}) per chunk, then ("done", "", metrics).

    A generator rather than a callback so the caller can abort it -- plan
    M4.5 requires aborting the upstream request when the browser disconnects,
    or an orphaned generation blocks the next real question for a minute.
    """
    model = model or config.LLM_MODEL
    body = _payload(messages, model, True, num_predict, options)

    try:
        with httpx.stream("POST", config.OLLAMA_CHAT_URL, json=body,
                          timeout=_timeouts(timeout)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LLMBadResponse(f"{model}: unparseable stream line") from exc
                if data.get("error"):
                    raise LLMBadResponse(f"{model}: {data['error']}")
                piece = (data.get("message") or {}).get("content", "")
                if piece:
                    yield "token", piece, {}
                if data.get("done"):
                    yield "done", "", _result_from(data, model).__dict__
                    return
    except httpx.TimeoutException as exc:
        raise LLMTimeout(
            f"{model}: stream stalled past "
            f"{timeout or config.GENERATE_TIMEOUT:.0f}s") from exc
    except httpx.HTTPStatusError as exc:
        raise LLMBadResponse(f"{model}: Ollama returned {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise LLMUnavailable(
            f"cannot reach Ollama at {config.OLLAMA_HOST} ({exc}). "
            f"Start it with: ollama serve") from exc


def list_models() -> list[str]:
    """Model tags Ollama currently has. Used by preflight (M4.5)."""
    try:
        response = httpx.get(config.OLLAMA_TAGS_URL,
                             timeout=httpx.Timeout(10.0, connect=config.CONNECT_TIMEOUT))
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMUnavailable(
            f"cannot reach Ollama at {config.OLLAMA_HOST} ({exc})") from exc
    return [m.get("name", "") for m in response.json().get("models", [])]
