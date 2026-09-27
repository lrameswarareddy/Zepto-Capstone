import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from fastapi import FastAPI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

MOCK_LLM = os.getenv("MOCK_LLM", "1")
APP_ROOT = Path(__file__).resolve().parent
DOCS_DIR = APP_ROOT / "docs"

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class State(TypedDict, total=False):
    query: str
    intent: str
    relevant_docs: list[str]
    answer: str
    sources: list[str]
    confidence: float


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

PROMPT_TEMPLATE = """
Role: You are Zepto's support policy assistant.
Context: Use only the retrieved Zepto policy excerpts provided below. If a question cannot be answered from the context, say so clearly.
Task: Answer the user's question using the provided policy context and keep the answer grounded, brief, and factual.
Format: Return JSON with fields: answer, sources, confidence.
Length: Keep the answer concise but operationally useful.
Negative constraint: Do not answer using information not present in the provided context.
Few-shot example:
Question: What is the refund timeline?
Context: Approved refunds are credited within 3-5 business days or instantly to the Zepto wallet if the customer opts for wallet credit.
Answer: Refunds are credited within 3-5 business days to the original payment method, or instantly to the Zepto wallet if wallet credit is selected.
""".strip()


def classify_intent(question: str) -> Literal["policy_question", "general_question"]:
    q = question.lower()
    if any(keyword in q for keyword in POLICY_KEYWORDS):
        return "policy_question"
    return "general_question"


client = chromadb.Client()
collection_name = "zepto_policy_docs"
collection = client.get_or_create_collection(collection_name)
encoder = None


def get_encoder():
    global encoder
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return encoder


def ingest_documents():
    if collection.count() > 0:
        return
    ids = []
    docs = []
    metas = []
    for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = file_path.read_text(encoding="utf-8")
        ids.append(file_path.stem)
        docs.append(text)
        metas.append({"source": file_path.name})
    if docs:
        embeddings = get_encoder().encode(docs).tolist()
        collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)


def retrieve_top_chunks(query: str, top_k: int = 3):
    embedding = get_encoder().encode(query).tolist()
    result = collection.query(query_embeddings=[embedding], n_results=top_k)
    return result


def mock_mode() -> bool:
    return str(MOCK_LLM).lower() in {"", "1", "true"}


def generate_real_answer(query: str, context: list[str], sources: list[str]) -> AskResponse:
    """Placeholder extension point with bounded structured-output retries."""
    for attempt in range(3):
        try:
            if not context:
                raise ValueError("No grounded context available")
            raise NotImplementedError("Configure a real LLM provider for MOCK_LLM=0")
        except (ValueError, NotImplementedError) as error:
            if attempt == 2:
                return AskResponse(answer=f"Real LLM response unavailable: {error}", sources=sources, confidence=0.0)
    return AskResponse(answer="Real LLM response unavailable.", sources=sources, confidence=0.0)


def classify_intent_node(state: State) -> State:
    state["intent"] = classify_intent(state["query"])
    return state


def retrieve_and_answer_node(state: State) -> State:
    ingest_documents()
    hits = retrieve_top_chunks(state["query"])
    sources = hits.get("ids", [[]])[0]
    snippets = hits.get("documents", [[]])[0]
    state["relevant_docs"] = snippets
    if mock_mode():
        top_snippet = snippets[0][:200] if snippets else "No supporting context found."
        response = AskResponse(answer=f"Based on the retrieved context: {top_snippet}", sources=sources, confidence=1.0)
    else:
        response = generate_real_answer(state["query"], snippets, sources)
    state["answer"] = response.answer
    state["sources"] = response.sources
    state["confidence"] = response.confidence
    return state


def direct_answer_node(state: State) -> State:
    if mock_mode():
        response = AskResponse(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0)
    else:
        response = generate_real_answer(state["query"], [], [])
    state["answer"] = response.answer
    state["sources"] = response.sources
    state["confidence"] = response.confidence
    return state


def route_intent(state: State) -> Literal["retrieve_and_answer", "direct_answer"]:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


graph_builder = StateGraph(State)
graph_builder.add_node("classify_intent", classify_intent_node)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer_node)
graph_builder.add_node("direct_answer", direct_answer_node)
graph_builder.add_edge(START, "classify_intent")
graph_builder.add_conditional_edges("classify_intent", route_intent)
graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)
support_graph = graph_builder.compile()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    result = support_graph.invoke({"query": payload.query})
    return AskResponse(answer=result["answer"], sources=result["sources"], confidence=result["confidence"])
