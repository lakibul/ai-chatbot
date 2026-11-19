# -*- coding: utf-8 -*-
"""How reliable is structured output on a SMALL local model? (guide chapter 6.6 / 6.20)

c06 ran this once and got 3/3 correct on one run and 0/3 on the next, with the same code
(the wrong answer was price_bdt=2500000000000000, a runaway number). One run proves nothing,
so this script repeats it and also tries the other two ways LangChain can ask for JSON.

    python examples/langchain/c07_structured_output_reliability.py
Needs Ollama with qwen3:4b-instruct-2507-q4_K_M. About 5 minutes on a CPU.
"""
import os
import time

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

WRITER = "qwen3:4b-instruct-2507-q4_K_M"
OPTS = dict(base_url="http://127.0.0.1:11434", num_gpu=0, num_ctx=4096, num_thread=8, keep_alive="30m")
TEXT = "The Growth plan costs BDT 2,500 per agent per month."
TRIES = 6


class Plan(BaseModel):
    name: str = Field(description="plan name")
    price_bdt: int = Field(description="monthly price in BDT")


def run(label, method, temperature):
    llm = ChatOllama(model=WRITER, reasoning=False, num_predict=120, temperature=temperature, **OPTS)
    chain = llm.with_structured_output(Plan, method=method)
    seen, ok = [], 0
    t = time.time()
    for _ in range(TRIES):
        try:
            res = chain.invoke(f"Extract the plan from: {TEXT}")
            good = isinstance(res, Plan) and res.price_bdt == 2500
            ok += good
            seen.append(res.price_bdt if isinstance(res, Plan) else repr(res)[:20])
        except Exception as e:                                   # noqa: BLE001
            seen.append(f"ERR:{type(e).__name__}")
    print(f"{label:46} {ok}/{TRIES} correct  values={seen}  ({time.time() - t:.0f}s)", flush=True)


print(f"model: {WRITER}   text: {TEXT!r}\n")
run("json_schema (the default), temperature 0.1", "json_schema", 0.1)
run("json_schema, temperature 0", "json_schema", 0.0)
run("function_calling, temperature 0.1", "function_calling", 0.1)
run("function_calling, temperature 0", "function_calling", 0.0)
run("json_mode, temperature 0.1", "json_mode", 0.1)
print("DONE")
