# -*- coding: utf-8 -*-
"""Chapters 10-11: LangGraph (state machines), modern memory, human approval, agents."""
import operator
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

print("=== 1. a graph: state + nodes + edges ===")
class S(TypedDict):
    question: str
    route: str
    answer: str

def classify(s: S):
    return {"route": "billing" if "refund" in s["question"].lower() else "general"}
def billing(s: S):
    return {"answer": "Billing team: refunds within 7 days."}
def general(s: S):
    return {"answer": "General assistant answer."}

g = StateGraph(S)
g.add_node("classify", classify); g.add_node("billing", billing); g.add_node("general", general)
g.add_edge(START, "classify")
g.add_conditional_edges("classify", lambda s: s["route"], {"billing": "billing", "general": "general"})
g.add_edge("billing", END); g.add_edge("general", END)
app = g.compile()
print(app.invoke({"question": "How do I get a refund?"}))
print("stream_mode='updates' (one line per node that ran):")
for step in app.stream({"question": "hello"}, stream_mode="updates"):
    print("  ", step)

print("\n=== 2. the graph, as a diagram you can paste into any Mermaid viewer ===")
print(app.get_graph().draw_mermaid())

print("=== 3. a LOOP: nodes can repeat until a rule says stop ===")
class Loop(TypedDict):
    tries: Annotated[list, operator.add]       # reducer: new values are APPENDED, not replaced
def attempt(s: Loop):
    return {"tries": [f"try{len(s['tries']) + 1}"]}
lg = StateGraph(Loop)
lg.add_node("attempt", attempt)
lg.add_edge(START, "attempt")
lg.add_conditional_edges("attempt", lambda s: END if len(s["tries"]) >= 3 else "attempt")
print(lg.compile().invoke({"tries": []}))

print("\n=== 4. MODERN memory: a checkpointer + thread_id (replaces RunnableWithMessageHistory) ===")
model = FakeListChatModel(responses=["18 days.", "Up to 5 days can be carried forward."])
def chatbot(state: MessagesState):
    return {"messages": [model.invoke(state["messages"])]}
mg = StateGraph(MessagesState)
mg.add_node("chatbot", chatbot); mg.add_edge(START, "chatbot"); mg.add_edge("chatbot", END)
memo = mg.compile(checkpointer=InMemorySaver())
rahim = {"configurable": {"thread_id": "rahim"}}
memo.invoke({"messages": [("user", "How many leave days?")]}, rahim)
r2 = memo.invoke({"messages": [("user", "Can I carry some forward?")]}, rahim)
print("rahim's thread now holds:", [f"{m.type}:{m.content}" for m in r2["messages"]])
print("a different thread is empty:", memo.get_state({"configurable": {"thread_id": "karim"}}).values)

print("\n=== 5. human-in-the-loop: pause, ask, resume ===")
class H(TypedDict):
    amount: int
    result: str
def approve(s: H):
    ok = interrupt(f"Approve BDT {s['amount']}?")        # execution STOPS here and is saved
    return {"result": "approved" if ok == "yes" else "rejected"}
hg = StateGraph(H); hg.add_node("approve", approve); hg.add_edge(START, "approve"); hg.add_edge("approve", END)
hap = hg.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}
first = hap.invoke({"amount": 30000}, cfg)
print("paused, asking:", first["__interrupt__"][0].value)
print("after human says yes:", hap.invoke(Command(resume="yes"), cfg))

print("\n=== 6. an AGENT = a loop of  model -> tool -> model  (scripted model, no real LLM) ===")
class Scripted(BaseChatModel):
    script: list
    i: int = 0
    @property
    def _llm_type(self):
        return "scripted"
    def bind_tools(self, tools, **kwargs):
        return self
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[self.i]; self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

@tool
def lookup_limit(expense_type: str) -> str:
    """Look up the reimbursement limit for an expense type."""
    return {"hotel outside dhaka": "BDT 6,000 per night"}.get(expense_type.lower(), "unknown")

llm = Scripted(script=[
    AIMessage(content="", tool_calls=[{"name": "lookup_limit", "args": {"expense_type": "hotel outside Dhaka"}, "id": "c1"}]),
    AIMessage(content="The hotel limit outside Dhaka is BDT 6,000 per night."),
])
agent = create_agent(model=llm, tools=[lookup_limit], system_prompt="Answer using the tools.")
res = agent.invoke({"messages": [("user", "What is the hotel limit outside Dhaka?")]})
for m in res["messages"]:
    extra = f"  tool_calls={[c['name'] for c in m.tool_calls]}" if getattr(m, "tool_calls", None) else ""
    print(f"  {m.type:5s} {str(m.content)[:60]!r}{extra}")
print("the agent graph itself:", [n for n in agent.get_graph().nodes])
print("DONE")
