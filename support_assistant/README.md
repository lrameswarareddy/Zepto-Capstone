# Support Assistant

Run locally from this directory:

```powershell
$env:MOCK_LLM = "1"
python -m uvicorn main:app --reload
```

The default path is deterministic and offline for LLM generation. `sentence-transformers` creates local `all-MiniLM-L6-v2` embeddings and ChromaDB stores the eight policy documents from `docs/`.

Embedding-model import and document ingestion are lazy: `/health` and general policy-independent questions respond without loading the model; the first policy question loads the local model and indexes the documents.

## Architecture

```text
8 policy files -> ingest_documents() -> MiniLM embeddings -> ChromaDB collection
                                                               |
query -> classify_intent node -> conditional route -> retrieve_and_answer node
                                      \-------------> direct_answer node
                                             |
                                  AskResponse(answer, sources, confidence)
```

`ingest_documents()` reads and embeds one chunk per policy document. `retrieve_and_answer_node()` embeds a policy query and retrieves the top three chunks from the `zepto_policy_docs` collection. In mock mode it returns the first 200 characters of the highest-ranked chunk. `direct_answer_node()` handles general questions without retrieval. The graph in `main.py` is a LangGraph `StateGraph` with `classify_intent`, `retrieve_and_answer`, and `direct_answer` nodes plus a conditional edge.

Only final answer generation branches on `MOCK_LLM`: the default value (`1`) uses the deterministic response; `MOCK_LLM=0` enters the bounded three-attempt real-provider extension point. Retrieval and embeddings run in both modes.

## Example Responses

Policy query:

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect...",
  "sources": ["doc_02", "doc_06", "doc_01"],
  "confidence": 1.0
}
```

General query:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Docker

```powershell
docker build -t zepto-support-assistant .
 docker run --rm -p 8000:8000 zepto-support-assistant
```

The container runs `uvicorn main:app` on port 8000 and uses the same mock baseline.
