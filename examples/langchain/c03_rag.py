# -*- coding: utf-8 -*-
"""Chapter 5: RAG with LangChain, piece by piece. Fake model + fake embeddings."""
import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

TEXT = """# Leave
Annual leave is 18 days. Apply at least 3 working days in advance.

# Remote work
After probation you may work remotely up to 2 days per week.

# Notice
Confirmed employees must give 60 days notice."""

print("=== 1. Document = text + metadata ===")
d = Document(page_content="Annual leave is 18 days.", metadata={"source": "handbook.md", "section": "Leave"})
print(d.page_content, "|", d.metadata)

print("\n=== 2. splitting ===")
md = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "section")]).split_text(TEXT)
print("by heading:", [(x.metadata["section"], len(x.page_content)) for x in md])
rc = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10).split_text(TEXT)
print(f"by size (60 chars, overlap 10): {len(rc)} chunks; first two:", [c.replace(chr(10), ' / ') for c in rc[:2]])

print("\n=== 3. embeddings + vector store ===")
emb = DeterministicFakeEmbedding(size=16)         # real life: OllamaEmbeddings(model="bge-m3")
vec = emb.embed_query("hello")
print("one text ->", len(vec), "numbers, e.g.", [round(x, 3) for x in vec[:3]])
store = InMemoryVectorStore(embedding=emb)
store.add_documents(md)
hits = store.similarity_search_with_score("Annual leave is 18 days. Apply at least 3 working days in advance.", k=2)
print("nearest:", [(h.metadata["section"], round(s, 3)) for h, s in hits], " (fake vectors: only an EXACT text matches strongly)")
retr = store.as_retriever(search_kwargs={"k": 2})
print("as_retriever -> a Runnable:", type(retr).__name__, "| .invoke gives", len(retr.invoke("x")), "Documents")

print("\n=== 4. a CUSTOM retriever (how to plug your own search into LangChain) ===")
class KeywordRetriever(BaseRetriever):
    docs: list[Document]
    k: int = 2
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        tok = lambda s: re.findall(r"\w+", s.lower())
        bm = BM25Okapi([tok(x.page_content) for x in self.docs])
        scores = bm.get_scores(tok(query))
        order = sorted(range(len(self.docs)), key=lambda i: -scores[i])[: self.k]
        # per-passage numbers CAN travel in metadata:
        return [Document(page_content=self.docs[i].page_content,
                         metadata={**self.docs[i].metadata, "bm25_score": round(float(scores[i]), 3), "rank": r + 1})
                for r, i in enumerate(order)]
kw = KeywordRetriever(docs=md)
for h in kw.invoke("how many days notice"):
    print(" ", h.metadata, "|", h.page_content[:40])
print("what a retriever's return value CANNOT hold is anything NOT about one passage:")
print("  e.g. the query tokens, the shortlist decisions, the relevance-floor verdict, per-step milliseconds")

print("\n=== 5. hybrid search: EnsembleRetriever (weighted reciprocal-rank fusion) ===")
sem = store.as_retriever(search_kwargs={"k": 3})
hybrid = EnsembleRetriever(retrievers=[kw, sem], weights=[0.5, 0.5])
print("hybrid returned:", [h.metadata.get("section") for h in hybrid.invoke("how many days notice")])

print("\n=== 6. the classic RAG chain ===")
def fmt(docs):
    return "\n\n".join(f"<document section='{d.metadata.get('section')}'>\n{d.page_content}\n</document>" for d in docs)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the documents."),
    ("human", "{context}\n\nQuestion: {question}")])
fake = FakeListChatModel(responses=["Confirmed employees give 60 days notice."])
rag = {"context": kw | fmt, "question": RunnablePassthrough()} | prompt | fake | StrOutputParser()
print(rag.invoke("how many days notice"))

print("\n=== 7. RAG that ALSO returns its sources (so you can show citations) ===")
rag2 = (RunnablePassthrough.assign(docs=lambda x: kw.invoke(x["question"]))
        .assign(answer=lambda x: (prompt | fake | StrOutputParser()).invoke({"context": fmt(x["docs"]), "question": x["question"]})))
out = rag2.invoke({"question": "how many days notice"})
print("answer :", out["answer"])
print("sources:", [d.metadata["section"] for d in out["docs"]])

print("\n=== 8. history-aware rewrite (what this project's rewrite.py does by hand) ===")
rewriter = (ChatPromptTemplate.from_messages([
    ("system", "Rewrite the follow-up as one standalone question."),
    ("human", "Conversation:\n{history}\nFollow-up: {question}\nRewritten:")])
    | FakeListChatModel(responses=["How many annual leave days can be carried forward?"]) | StrOutputParser())
print(rewriter.invoke({"history": "User: How many days of annual leave?\nAssistant: 18.", "question": "Can I carry some forward?"}))
print("DONE")
