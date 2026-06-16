# Tutorial: Build a RAG Workflow

This tutorial shows an end-to-end retrieval-augmented generation workflow:
chunk documents, embed chunks, store vectors and metadata, retrieve relevant
context, optionally rerank, and build a prompt for an LLM.

The example uses a tiny deterministic embedding function so it runs without an
external model. In a real application, replace `embed_text()` with your
embedding provider or local model.

## 1. Setup

```python
import hashlib
import re
from typing import Iterable

import numpy as np
import lynse

DIM = 16
```

## 2. A toy embedding function

Replace this with a real embedding model in production:

```python
def embed_text(text: str, dim: int = DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        bucket = int.from_bytes(digest, "little") % dim
        vector[bucket] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector
```

This creates normalized bag-of-words style vectors. It is not a semantic model,
but it is enough to demonstrate the LynseDB workflow.

## 3. Source documents

```python
documents = [
    {
        "doc_id": "install",
        "title": "Install LynseDB",
        "tenant": "acme",
        "lang": "en",
        "text": "Install LynseDB with pip install lynsedb. Use Python 3.9 or newer.",
    },
    {
        "doc_id": "local-remote",
        "title": "Local and remote mode",
        "tenant": "acme",
        "lang": "en",
        "text": "Local mode embeds the Rust backend. Remote mode uses lynse serve and HTTP.",
    },
    {
        "doc_id": "filters",
        "title": "Metadata filters",
        "tenant": "acme",
        "lang": "en",
        "text": "Use where filters for tenant, language, tags, booleans, and date ranges.",
    },
    {
        "doc_id": "ops",
        "title": "Operations",
        "tenant": "globex",
        "lang": "en",
        "text": "Use checkpoint before snapshots. Monitor healthz, readyz, and metrics.",
    },
]
```

## 4. Chunk documents

This example uses short documents. For longer documents, split by section,
paragraph, sentence windows, or token windows.

```python
def chunk_document(doc: dict, max_words: int = 40) -> Iterable[dict]:
    words = doc["text"].split()
    for chunk_index, start in enumerate(range(0, len(words), max_words)):
        chunk_text = " ".join(words[start:start + max_words])
        yield {
            "doc_id": doc["doc_id"],
            "chunk_index": chunk_index,
            "title": doc["title"],
            "tenant": doc["tenant"],
            "lang": doc["lang"],
            "text": chunk_text,
            "source": f"{doc['doc_id']}#{chunk_index}",
        }
```

## 5. Create the collection

```python
client = lynse.VectorDBClient(uri="./rag-demo")
db = client.create_database("rag", drop_if_exists=True)
collection = db.require_collection(
    "chunks",
    dim=DIM,
    drop_if_exists=True,
    description="RAG chunks with toy embeddings",
)
```

Use the embedding dimension from your real model. A collection has one fixed
primary vector dimension.

## 6. Insert chunks

```python
ids = []
vectors = []
fields = []

for doc in documents:
    for chunk in chunk_document(doc):
        text_for_embedding = f"{chunk['title']} {chunk['text']}"
        ids.append(chunk["source"])
        vectors.append(embed_text(text_for_embedding))
        fields.append(chunk)

with collection.insert_session() as session:
    session.add(ids=ids, vectors=vectors, fields=fields, batch_size=1000)

collection.build_index("FLAT-COS")
collection.checkpoint()
```

For production ingestion:

- use stable IDs from your own document/chunk registry;
- store `doc_id`, `chunk_index`, tenant, language, source path, URL, or version
  in metadata;
- store the chunk text if you want LynseDB to return context directly;
- call `checkpoint()` before snapshots or controlled shutdowns.

## 7. Retrieve context

```python
question = "How do I run LynseDB as a server?"
query_vector = embed_text(question)

result = collection.search(
    query_vector,
    k=3,
    where="tenant = 'acme' AND lang = 'en'",
    return_fields=True,
)

for row in result.to_list():
    print(row["id"], row["distance"], row["title"], row["text"])
```

Tenant and language filters keep retrieval inside the correct application
boundary.

## 8. Hybrid retrieval

Hybrid search is often useful for RAG because users include exact product names,
commands, or identifiers:

```python
hybrid = collection.hybrid_search(
    vector=query_vector,
    text=question,
    text_fields=["title", "text"],
    where="tenant = 'acme' AND lang = 'en'",
    fusion="rrf",
    candidate_limit=20,
    k=3,
    return_fields=True,
)

for row in hybrid.to_list():
    print(row["id"], row["distance"], row["title"])
```

Use vector search for semantic recall and BM25 search for exact terms.
`fusion="rrf"` is a good default because vector and text scores use different
scales.

## 9. Rerank candidates

A reranker can be a cross-encoder, an LLM scoring function, or a business rule.
This simple example boosts chunks whose text contains a query term:

```python
def simple_rerank(payload):
    query_text = payload["query"].get("text") or ""
    query_terms = set(re.findall(r"[a-z0-9]+", query_text.lower()))
    scored = []

    for item in payload["items"]:
        field = item.get("field") or {}
        haystack = f"{field.get('title', '')} {field.get('text', '')}".lower()
        overlap = sum(1 for term in query_terms if term in haystack)
        scored.append((item["id"], float(overlap)))

    return scored

reranked = collection.hybrid_search(
    vector=query_vector,
    text=question,
    text_fields=["title", "text"],
    where="tenant = 'acme'",
    candidate_limit=20,
    k=10,
    reranker=simple_rerank,
    rerank_k=3,
    return_fields=True,
)

print(reranked.to_list())
```

Set `rerank_with_fields=True` when the reranker needs fields but the final
response does not need to return them.

## 10. Build an LLM prompt

```python
def build_prompt(question: str, rows: list[dict]) -> str:
    context_blocks = []
    for i, row in enumerate(rows, start=1):
        context_blocks.append(
            f"[{i}] {row.get('title', '')}\n"
            f"source: {row.get('source', '')}\n"
            f"{row.get('text', '')}"
        )

    context = "\n\n".join(context_blocks)
    return (
        "Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

prompt = build_prompt(question, reranked.to_list())
print(prompt)
```

Pass this prompt to your LLM client. Keep citations by carrying `source`,
`doc_id`, and `chunk_index` in metadata.

## 11. Update changed documents

When a source document changes, upsert its chunks. This example updates one
chunk by ID:

```python
updated_text = "Start the server with lynse serve --host 0.0.0.0 --port 7637."
updated_field = {
    "doc_id": "local-remote",
    "chunk_index": 0,
    "title": "Local and remote mode",
    "tenant": "acme",
    "lang": "en",
    "text": updated_text,
    "source": "local-remote#0",
}

collection.upsert(
    ids="local-remote#0",
    vectors=embed_text(f"{updated_field['title']} {updated_text}"),
    fields=updated_field,
)
collection.commit()
```

If the number of chunks changes, delete old chunk IDs that no longer exist:

```python
collection.delete([old_chunk_id])
collection.commit()
```

Run `compact()` later during maintenance if many rows have been tombstoned.

## 12. RAG checklist

- Pick an embedding model and record its dimension and metric.
- Normalize vectors if your metric strategy requires it.
- Use one stable public string or integer ID per chunk.
- Store source metadata needed for filtering and citations.
- Use `where` for tenant, permission, language, source, and freshness filters.
- Start with `FLAT-COS` or `FLAT-L2` as a baseline.
- Evaluate HNSW, IVF, DiskANN, or quantized indexes against known questions.
- Use hybrid search when exact terms matter.
- Add a reranker when final ordering quality matters.
- Snapshot or export before migrations and large re-indexing jobs.
