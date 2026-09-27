import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

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


class State(TypedDict):
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
encoder = SentenceTransformer("all-MiniLM-L6-v2")


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
        embeddings = encoder.encode(docs).tolist()
        collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)


def retrieve_top_chunks(query: str, top_k: int = 3):
    embedding = encoder.encode(query).tolist()
    result = collection.query(query_embeddings=[embedding], n_results=top_k)
    return result


@app.on_event("startup")
def startup_event():
    ingest_documents()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    intent = classify_intent(payload.query)
    if intent == "policy_question":
        hits = retrieve_top_chunks(payload.query)
        doc_ids = hits.get("ids", [[]])[0]
        snippets = hits.get("documents", [[]])[0]
        top_snippet = snippets[0][:200] if snippets else "No supporting context found."
        if str(MOCK_LLM).lower() in {"", "1", "true"}:
            answer = f"Based on the retrieved context: {top_snippet}"
            return AskResponse(answer=answer, sources=doc_ids, confidence=1.0)
        answer = "I can answer policy questions using all retrieved context when the real LLM mode is enabled."
        return AskResponse(answer=answer, sources=doc_ids, confidence=1.0)

    if str(MOCK_LLM).lower() in {"", "1", "true"}:
        return AskResponse(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0)
    return AskResponse(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0)
