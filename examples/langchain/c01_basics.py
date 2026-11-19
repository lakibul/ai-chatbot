# -*- coding: utf-8 -*-
"""Chapter 1-3: messages, chat models, prompts, parsers. No real LLM: fake models."""
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage, ToolMessage,
                                     trim_messages)
from langchain_core.output_parsers import (CommaSeparatedListOutputParser, JsonOutputParser,
                                           PydanticOutputParser, StrOutputParser)
from langchain_core.prompts import (ChatPromptTemplate, FewShotChatMessagePromptTemplate,
                                    MessagesPlaceholder, PromptTemplate)
from pydantic import BaseModel, Field

print("=== 1. messages ===")
msgs = [SystemMessage("You are helpful."), HumanMessage("Hi"), AIMessage("Hello!")]
for m in msgs:
    print(type(m).__name__, "| role/type =", m.type, "| content =", repr(m.content))

print("\n=== 2. a chat model: invoke / stream / batch ===")
model = FakeListChatModel(responses=["Dhaka is the capital.", "Paris is the capital."])
reply = model.invoke("Capital of Bangladesh?")
print("invoke ->", type(reply).__name__, repr(reply.content))
for chunk in FakeListChatModel(responses=["abc"]).stream("x"):
    print("stream chunk:", repr(chunk.content), end="  ")
print()
print("batch  ->", [r.content for r in FakeListChatModel(responses=["one", "two"]).batch(["q1", "q2"])])

print("\n=== 3. PromptTemplate vs ChatPromptTemplate ===")
pt = PromptTemplate.from_template("Tell me about {topic}.")
print(repr(pt.invoke({"topic": "pricing"}).to_string()))
cpt = ChatPromptTemplate.from_messages([
    ("system", "You answer about {company}."),
    MessagesPlaceholder("history", optional=True),
    ("human", "{question}"),
])
out = cpt.invoke({"company": "Nodi Tech", "question": "Hours?",
                  "history": [HumanMessage("hi"), AIMessage("hello")]})
for m in out.to_messages():
    print(" ", m.type.ljust(6), m.content)
print("input variables:", sorted(cpt.input_variables))

print("\n=== 4. THE BRACE TRAP (this project hit it) ===")
try:
    ChatPromptTemplate.from_messages([("human", 'Return JSON like {"a": 1}')]).invoke({})
except Exception as e:
    print("template text with { } ->", type(e).__name__)
ok = ChatPromptTemplate.from_messages([("human", 'Return JSON like {{"a": 1}}')]).invoke({})
print("escaped with {{ }}      ->", ok.to_messages()[0].content)
safe = ChatPromptTemplate.from_messages([("human", "{doc}")]).invoke({"doc": 'raw {text} <document>'})
print("braces in the VALUE are safe ->", safe.to_messages()[0].content)

print("\n=== 5. partial variables ===")
p = ChatPromptTemplate.from_messages([("system", "Date is {today}."), ("human", "{q}")]).partial(today="2026-11-19")
print(p.invoke({"q": "?"}).to_messages()[0].content)

print("\n=== 6. few-shot ===")
examples = [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}]
shots = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=ChatPromptTemplate.from_messages([("human", "{q}"), ("ai", "{a}")]))
full = ChatPromptTemplate.from_messages([("system", "Maths."), shots, ("human", "{q}")])
print([f"{m.type}:{m.content}" for m in full.invoke({"q": "5+5"}).to_messages()])

print("\n=== 7. output parsers ===")
print("str   ->", repr(StrOutputParser().invoke(AIMessage("plain text"))))
print("list  ->", CommaSeparatedListOutputParser().invoke(AIMessage("a, b, c")))
print("json  ->", JsonOutputParser().invoke(AIMessage('{"price": 2500, "plan": "Growth"}')))


class Plan(BaseModel):
    name: str = Field(description="plan name")
    price: int


pp = PydanticOutputParser(pydantic_object=Plan)
obj = pp.invoke(AIMessage('{"name": "Growth", "price": 2500}'))
print("pydantic ->", obj, "| type:", type(obj).__name__)
print("format instructions start:", pp.get_format_instructions()[:60].replace("\n", " "), "...")

print("\n=== 8. trim messages (keep the last N tokens/messages) ===")
hist = [HumanMessage(f"q{i}") if i % 2 == 0 else AIMessage(f"a{i}") for i in range(8)]
kept = trim_messages(hist, max_tokens=3, token_counter=len, strategy="last", start_on="human")
print("kept:", [m.content for m in kept])
print("tool message:", ToolMessage(content="42", tool_call_id="call_1").type)
