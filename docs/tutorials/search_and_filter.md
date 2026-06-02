# Tutorial: Search and Filter

This tutorial covers vector search, metadata filters, batch search, range search,
text search, hybrid search, and reranking.

## Setup

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./search-demo")
db = client.create_database("search_demo", drop_if_exists=True)
collection = db.require_collection("docs", dim=4, drop_if_exists=True)

items = [
    ([0.10, 0.20, 0.30, 0.40], 1, {"title": "vector database intro", "lang": "en", "rank": 1, "tags": ["vector", "database"]}),
    ([0.11, 0.19, 0.29, 0.39], 2, {"title": "python client guide", "lang": "en", "rank": 2, "tags": ["python"]}),
    ([0.80, 0.10, 0.20, 0.10], 3, {"title": "rust backend notes", "lang": "en", "rank": 3, "tags": ["rust"]}),
    ([0.70, 0.12, 0.19, 0.13], 4, {"title": "archive", "lang": "fr", "rank": 4, "tags": ["archive"]}),
]

with collection.insert_session() as session:
    session.bulk_add_items(items, enable_progress_bar=False)

collection.build_index("FLAT-L2")
```

## Vector search

```python
query = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)

result = collection.search(query, k=3, return_fields=True)
print(result.to_list())
```

The meaning of `distances` depends on the index metric:

| Metric | Result ordering | Range-search threshold |
| --- | --- | --- |
| IP | higher score is better | `score >= threshold` |
| Cosine | higher score is better | `score >= threshold` |
| L2 | lower squared distance is better | `distance <= threshold` |
| Hamming/Jaccard | lower distance is better | `distance <= threshold` |

## Metadata filters

Filters are SQL-like strings:

```python
result = collection.search(
    query,
    k=3,
    where="lang = 'en' AND rank <= 2",
    return_fields=True,
)
```

Use `CONTAINS` for array metadata:

```python
result = collection.query(where="tags CONTAINS 'vector'")
```

See [Field filters](../FieldExpression.md) for the supported syntax and
performance notes.

## Query fields without vector search

```python
rows = collection.query(where="rank >= 2 AND rank < 4")
print(rows.ids)
print(rows.fields)

ids_only = collection.query(where="lang = 'en'", return_ids_only=True)
print(ids_only.ids)
```

`query()` without `where` or `filter_ids` returns an empty result. This prevents
accidental full scans.

## Retrieve vectors

```python
data = collection.query_vectors(filter_ids=[1, 2, 3])
print(data.ids)
print(data.vectors.shape)
print(data.fields)
```

`query_vectors()` without `where` or `filter_ids` returns an empty `(0, dim)`
vector array.

## Batch search

```python
queries = np.random.rand(5, 4).astype(np.float32)
results = collection.batch_search(queries, k=2, return_fields=True)

for result in results:
    print(result.to_list())
```

## Range search

Range search returns all matches within a threshold, capped by `max_results`.

```python
nearby = collection.search_range(query, threshold=0.05, max_results=100)
print(nearby.ids, nearby.distances)
```

## Search profile

Use `search_profile()` to inspect filters, index path, candidate counts, and
timings.

```python
profile = collection.search_profile(
    query,
    k=3,
    where="lang = 'en'",
)

print(profile["items"]["ids"])
print(profile["profile"])
```

## Text search

Text search runs BM25 over stored metadata fields.

```python
result = collection.text_search(
    "vector database",
    k=3,
    text_fields=["title"],
    return_fields=True,
)
print(result.to_list())
```

`text_fields=None` searches all text-like metadata fields.

## Hybrid search

Hybrid search combines vector and text candidates. `fusion="rrf"` is a robust
default when vector and text scores use different scales.

```python
result = collection.hybrid_search(
    vector=query,
    text="vector database",
    text_fields=["title", "tags"],
    fusion="rrf",
    k=3,
    return_fields=True,
)
print(result.to_list())
```

Weighted fusion is useful when you have calibrated scores:

```python
result = collection.hybrid_search(
    vector=query,
    text="vector",
    fusion="weighted",
    vector_weight=0.7,
    text_weight=0.3,
    k=3,
)
```

## External rerank hook

`reranker` lets you apply a cross-encoder, LLM scorer, or custom business rule
after candidate retrieval.

The callback receives:

```python
{
    "query": {...},
    "items": [
        {"id": 1, "score": 0.91, "field": {...}},
    ],
}
```

It may return ordered IDs, `(id, score)` pairs, a score array aligned with input
items, or a `{id: score}` mapping.

```python
def rerank(payload):
    query_text = payload["query"].get("text", "")
    scores = []
    for item in payload["items"]:
        title = (item["field"] or {}).get("title", "")
        scores.append((item["id"], 1.0 if query_text in title else 0.0))
    return scores

result = collection.hybrid_search(
    vector=query,
    text="vector",
    text_fields=["title"],
    k=20,
    reranker=rerank,
    rerank_k=5,
    return_fields=True,
)
```

Set `rerank_with_fields=True` when the reranker needs fields but the final
result does not need to return them.
