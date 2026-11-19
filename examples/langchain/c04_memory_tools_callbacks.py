# -*- coding: utf-8 -*-
"""Chapters 6-9: memory, tools, callbacks, caching, rate limits."""
import time

from langchain_core.caches import InMemoryCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool

print("=== 1. chat memory: RunnableWithMessageHistory ===")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."), MessagesPlaceholder("history"), ("human", "{input}")])
seen = []
def spy(p):                       # show what the model REALLY receives
    seen.append([f"{m.type}:{m.content}" for m in p.to_messages()]); return p
from langchain_core.runnables import RunnableLambda
chain = prompt | RunnableLambda(spy) | FakeListChatModel(responses=["18 days.", "Up to 5 days."]) | StrOutputParser()
stores = {}
def get_history(session_id):
    return stores.setdefault(session_id, InMemoryChatMessageHistory())
memo = RunnableWithMessageHistory(chain, get_history, input_messages_key="input", history_messages_key="history")
cfg = {"configurable": {"session_id": "rahim"}}
print("turn 1 ->", memo.invoke({"input": "How many leave days?"}, cfg))
print("turn 2 ->", memo.invoke({"input": "Can I carry some forward?"}, cfg))
print("the model saw on turn 2:", seen[-1])
print("another user (new session) has an empty memory:", len(get_history("karim").messages), "messages")
print("stored for rahim:", [m.content for m in get_history("rahim").messages])

print("\n=== 2. tools: a function the model may ASK you to run ===")
@tool
def add_days(start_day: int, days: int) -> int:
    """Add a number of days to a start day-of-month."""
    return start_day + days
print("name       :", add_days.name)
print("description:", add_days.description)
print("arguments  :", add_days.args)
print("run it     :", add_days.invoke({"start_day": 10, "days": 5}))

print("\n=== 3. the tool-calling loop, by hand (this is all an 'agent' is) ===")
ask = AIMessage(content="", tool_calls=[{"name": "add_days", "args": {"start_day": 10, "days": 5}, "id": "call_1"}])
print("1) model replies with a REQUEST, not text:", ask.tool_calls[0]["name"], ask.tool_calls[0]["args"])
result = add_days.invoke(ask.tool_calls[0])             # invoking with the call gives a ToolMessage
print("2) we run it and wrap the result        :", type(result).__name__, repr(result.content))
final = FakeListChatModel(responses=["The date is the 15th."]).invoke([HumanMessage("10th + 5 days?"), ask, result])
print("3) the model sees the result, answers   :", final.content)

print("\n=== 4. callbacks: watch every step (local, no cloud) ===")
class Timeline(BaseCallbackHandler):
    def __init__(self): self.rows = []
    def on_chain_start(self, serialized, inputs, **kw): self.rows.append(("chain_start", (serialized or {}).get("name") or kw.get("name")))
    def on_chat_model_start(self, serialized, messages, **kw): self.rows.append(("model_start", f"{len(messages[0])} messages"))
    def on_llm_new_token(self, token, **kw): self.rows.append(("token", token))
    def on_llm_end(self, response, **kw): self.rows.append(("model_end", ""))
    def on_chain_end(self, outputs, **kw): self.rows.append(("chain_end", ""))
tl = Timeline()
simple = ChatPromptTemplate.from_messages([("human", "{q}")]) | FakeListChatModel(responses=["Hi"]) | StrOutputParser()
list(simple.stream({"q": "x"}, config={"callbacks": [tl]}))
print([r[0] + (":" + r[1] if r[1] else "") for r in tl.rows])

print("\n=== 5. caching identical calls ===")
set_llm_cache(InMemoryCache())
m = FakeListChatModel(responses=["first", "second", "third"])
print("same question twice:", m.invoke("q1").content, "/", m.invoke("q1").content, " <- 2nd came from the cache")
print("a different question:", m.invoke("q2").content)
set_llm_cache(None)

print("\n=== 6. rate limiting ===")
rl = InMemoryRateLimiter(requests_per_second=5, check_every_n_seconds=0.01, max_bucket_size=1)
limited = FakeListChatModel(responses=["x"] * 4, rate_limiter=rl)
t = time.time()
for _ in range(4):
    limited.invoke("q")
dt = time.time() - t
print(f"4 calls at 5/second took at least 0.45s? {dt >= 0.45}   (lower bound only: a busy machine only makes it slower)")
print("DONE")
