# -*- coding: utf-8 -*-
"""Chapter 6.20: the same ideas with REAL local models (slow: this is a CPU).

Run from the project folder so storage/chunks.json is found:
    python examples/langchain/c06_real_ollama.py
Needs Ollama running with qwen3:1.7b-q4_K_M, qwen3:4b-instruct-2507-q4_K_M and bge-m3.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"          # nothing leaves this machine
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field

HOST = "http://127.0.0.1:11434"
SMALL = "qwen3:1.7b-q4_K_M"
WRITER = "qwen3:4b-instruct-2507-q4_K_M"
# the same options this project always sends (num_gpu=0 is not optional on this machine)
OPTS = dict(base_url=HOST, num_gpu=0, num_ctx=4096, num_thread=8, temperature=0.1, keep_alive="30m")


def stamp(title):
    print(f"\n=== {title} ===", flush=True)


stamp("1. ChatOllama: invoke, and what comes back besides the text")
small = ChatOllama(model=SMALL, reasoning=False, num_predict=40, **OPTS)
t = time.time()
r = small.invoke("Reply with exactly: ok")
print("text            :", repr(r.content), f"({time.time() - t:.1f}s)")
print("type            :", type(r).__name__)
print("usage_metadata  :", r.usage_metadata)
print("metadata keys   :", sorted(k for k in r.response_metadata if k in ("model", "done", "done_reason", "total_duration", "load_duration", "prompt_eval_count", "eval_count")))

stamp("2. stream")
chunks = [c.content for c in small.stream("Name two colours, short.") if c.content]
print(len(chunks), "pieces:", chunks[:6], "...")

stamp("3. init_chat_model: one line, any provider (here: ollama)")
m = init_chat_model(SMALL, model_provider="ollama", reasoning=False, num_predict=20, **OPTS)
print(type(m).__name__, "->", repr(m.invoke("Say hi in one word.").content))

stamp("4. structured output on a SMALL model: how reliable is it? (3 tries each)")
class Plan(BaseModel):
    name: str = Field(description="plan name")
    price_bdt: int = Field(description="monthly price in BDT")
TEXT = "The Growth plan costs BDT 2,500 per agent per month."
for name in (SMALL, WRITER):
    llm = ChatOllama(model=name, reasoning=False, num_predict=120, **OPTS).with_structured_output(Plan)
    ok = 0
    for i in range(3):
        try:
            res = llm.invoke(f"Extract the plan from: {TEXT}")
            good = isinstance(res, Plan) and res.price_bdt == 2500
            ok += good
            print(f"  {name[:22]:22} try {i + 1}: {res!r}  {'OK' if good else 'WRONG'}", flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f"  {name[:22]:22} try {i + 1}: FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
    print(f"  -> {name}: {ok}/3 correct")

stamp("5. tool calling on the 4B model")
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
tooled = ChatOllama(model=WRITER, reasoning=False, num_predict=120, **OPTS).bind_tools([add])
resp = tooled.invoke("What is 17 plus 25? Use the add tool.")
print("model asked for tools:", [(c["name"], c["args"]) for c in resp.tool_calls] or "NONE (answered in text: " + repr(resp.content[:60]) + ")")
if resp.tool_calls:
    print("we run it            :", add.invoke(resp.tool_calls[0]["args"]))

stamp("6. real embeddings (bge-m3): meaning, not spelling")
# num_gpu=0 prevents a bge-m3 crash on this machine (CLAUDE.md section 2).
# NOTE: OllamaEmbeddings wants keep_alive in SECONDS (an int); ChatOllama accepts "30m".
emb = OllamaEmbeddings(model="bge-m3", base_url=HOST, num_gpu=0, keep_alive=1800)
def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
v = emb.embed_documents(["How many days of annual leave do I get?",
                         "What is the yearly vacation allowance?",
                         "What does error ND-1042 mean?"])
print("vector length                          :", len(v[0]))
print("leave vs vacation (same meaning)       :", round(cos(v[0], v[1]), 3))
print("leave vs error code (different meaning):", round(cos(v[0], v[2]), 3))

stamp("7. a real RAG in ~20 lines, over this project's own 50 passages")
root = Path(__file__).resolve().parent.parent.parent if "examples" in str(Path(__file__)) else Path(r"C:\ai-chatbot")
chunks_path = root / "storage" / "chunks.json"
rows = json.loads(chunks_path.read_text(encoding="utf-8"))
docs = [Document(page_content=r["embed_text"], metadata={"doc": r["doc_id"], "status": r["status"]}) for r in rows]
t = time.time()
store = InMemoryVectorStore(embedding=emb)
store.add_documents(docs)
print(f"embedded {len(docs)} passages in {time.time() - t:.0f}s")
retriever = store.as_retriever(search_kwargs={"k": 5})
fmt = lambda ds: "\n\n".join(f'<document id="{d.metadata["doc"]}">\n{d.page_content}\n</document>' for d in ds)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the documents. End with the document name in square brackets."),
    ("human", "{context}\n\nQuestion: {question}")])
writer = ChatOllama(model=WRITER, reasoning=False, num_predict=200, **OPTS)
rag = {"context": retriever | fmt, "question": lambda x: x} | prompt | writer | StrOutputParser()
for q in ("How many days of annual leave do I get?",
          "Is the old policy of 15 annual leave days still valid?"):
    t = time.time()
    docs_used = [d.metadata["doc"] for d in retriever.invoke(q)]
    print(f"\nQ: {q}\n   retrieved: {docs_used}")
    print(f"A: {rag.invoke(q)}   ({time.time() - t:.0f}s)", flush=True)
print("\nDONE")
