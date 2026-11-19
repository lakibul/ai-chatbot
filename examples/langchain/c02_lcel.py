# -*- coding: utf-8 -*-
"""Chapter 4: LCEL (the pipe), the Runnable interface."""
import asyncio
import time
from operator import itemgetter

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnableBranch, RunnableConfig, RunnableLambda,
                                      RunnableParallel, RunnablePassthrough)

print("=== 1. the pipe builds a chain ===")
prompt = ChatPromptTemplate.from_messages([("human", "Say hi to {name}")])
model = FakeListChatModel(responses=["Hi Rahim!"])
chain = prompt | model | StrOutputParser()
print(type(chain).__name__, "->", chain.invoke({"name": "Rahim"}))
print("steps:", [type(s).__name__ for s in chain.steps])

print("\n=== 2. every Runnable has the same 6 methods ===")
sq = RunnableLambda(lambda x: x * x)
print("invoke :", sq.invoke(4))
print("batch  :", sq.batch([1, 2, 3]))
print("stream :", list(sq.stream(5)))
print("ainvoke:", asyncio.run(sq.ainvoke(6)))
print("a plain function becomes a Runnable by the | operator:", (sq | (lambda x: x + 1)).invoke(3))

print("\n=== 3. RunnableParallel: run branches at the same time ===")
import threading
gate = threading.Barrier(3, timeout=3)      # opens ONLY when 3 threads are waiting at once
def meet(tag):
    def run(x):
        gate.wait()                         # a one-after-another run would wait here forever
        return f"{tag}({x})"
    return RunnableLambda(run)
print(RunnableParallel(a=meet("A"), b=meet("B"), c=meet("C")).invoke("q"),
      "<- all 3 reached the barrier together: they truly ran at the same time")
try:
    gate.reset()
    (meet("A") | meet("B") | meet("C")).invoke("q")
except threading.BrokenBarrierError:
    print("one after another -> BrokenBarrierError: nobody else arrives (so a chain is sequential)")
print("dict literal in a chain is the same thing:", ({"x": RunnablePassthrough(), "y": sq} | RunnableLambda(lambda d: d)).invoke(3))

print("\n=== 4. passthrough + assign (carry data forward) ===")
c = RunnablePassthrough.assign(double=lambda d: d["n"] * 2).assign(text=lambda d: f"n={d['n']}, double={d['double']}")
print(c.invoke({"n": 7}))
print("itemgetter picks a key:", (itemgetter("n") | sq).invoke({"n": 9}))

print("\n=== 5. branching (routing) ===")
route = RunnableBranch(
    (lambda d: "refund" in d["q"].lower(), RunnableLambda(lambda d: "-> billing team")),
    (lambda d: "error" in d["q"].lower(), RunnableLambda(lambda d: "-> support engineer")),
    RunnableLambda(lambda d: "-> general assistant"))
for q in ("I want a refund", "got error ND-1042", "hello"):
    print(f"{q!r:24}", route.invoke({"q": q}))

print("\n=== 6. retry and fallback ===")
calls = {"n": 0}
def flaky(x):
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("Ollama busy")
    return f"ok after {calls['n']} tries"
print("retry   :", RunnableLambda(flaky).with_retry(stop_after_attempt=4, wait_exponential_jitter=False).invoke(1))
broken = RunnableLambda(lambda x: 1 / 0)
backup = RunnableLambda(lambda x: "fallback answer")
print("fallback:", broken.with_fallbacks([backup]).invoke(1))

print("\n=== 7. batch with a concurrency limit ===")
lock, state = threading.Lock(), {"now": 0, "peak": 0}
def tracked(x):
    with lock:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
    time.sleep(0.05)
    with lock:
        state["now"] -= 1
    return x
for limit in (1, 3):
    state.update(now=0, peak=0)
    RunnableLambda(tracked).batch(list(range(8)), config=RunnableConfig(max_concurrency=limit))
    print(f"8 jobs, max_concurrency={limit} -> at most {state['peak']} ran at once")

print("\n=== 8. streaming flows THROUGH the chain ===")
streamer = ChatPromptTemplate.from_messages([("human", "{q}")]) | FakeListChatModel(responses=["Hello world"]) | StrOutputParser()
print([c for c in streamer.stream({"q": "x"})])

print("\n=== 9. async + astream_events (what a UI listens to) ===")
async def events():
    seen = []
    async for ev in streamer.astream_events({"q": "x"}, version="v2"):
        seen.append(ev["event"] + ":" + ev["name"])
    return seen
ev = asyncio.run(events())
print(len(ev), "events, e.g.:", [e for e in ev if "chat_model" in e or "prompt" in e.lower()][:4])

print("\n=== 10. look at the graph (draw the chain) ===")
chain.get_graph().print_ascii()

print("=== 11. bind / with_config / tags ===")
tagged = chain.with_config(run_name="greeter", tags=["demo"])
print("run_name set ->", tagged.config.get("run_name"), tagged.config.get("tags"))
print("DONE")
