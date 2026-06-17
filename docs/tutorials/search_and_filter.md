# Tutorial: Search and Filter

This tutorial covers vector search, document search, metadata filters, batch
search, range search, BM25 search, hybrid search, and reranking.

## Setup

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./search-demo")
db = client.create_database("search_demo", drop_if_exists=True)
collection = db.require_collection("docs", dim=4, drop_if_exists=True)

collection.add(
    ids=["intro", "python-guide", "rust-notes", "archive-fr"],
    vectors=[
        [0.10, 0.20, 0.30, 0.40],
        [0.11, 0.19, 0.29, 0.39],
        [0.80, 0.10, 0.20, 0.10],
        [0.70, 0.12, 0.19, 0.13],
    ],
    fields=[
        {"title": "vector database intro", "lang": "en", "rank": 1, "tags": ["vector", "database"]},
        {"title": "python client guide", "lang": "en", "rank": 2, "tags": ["python"]},
        {"title": "rust backend notes", "lang": "en", "rank": 3, "tags": ["rust"]},
        {"title": "archive", "lang": "fr", "rank": 4, "tags": ["archive"]},
    ],
)

collection.build_index("FLAT-L2")
```

## Vector search

```python
query = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)

result = collection.search(query, k=3, return_fields=True)
print(result.to_list())
```

`k` is the maximum number of returned matches. If filters or deleted rows remove
most candidates, fewer than `k` rows may be returned.

Use `return_fields=False` for the lowest payload size:

```python
result = collection.search(query, k=3)
print(result.ids)
print(result.distances)
```

Use `return_fields=True` when the application needs metadata in the same call:

```python
result = collection.search(query, k=3, return_fields=True)
for row in result.to_list():
    print(row["id"], row.get("title"))
```

The meaning of `distances` depends on the index metric:

| Metric | Result ordering | Range-search threshold |
| --- | --- | --- |
| IP | higher score is better | `score >= threshold` |
| Cosine | higher score is better | `score >= threshold` |
| L2 | lower squared distance is better | `distance <= threshold` |
| Hamming/Jaccard | lower distance is better | `distance <= threshold` |

## Document search

If you inserted documents with `collection.add(ids=..., documents=...)`, search
with text directly:

```python
result = collection.search(
    document="vector database guide",
    k=3,
    return_fields=True,
)
print(result.to_list())
```

LynseDB embeds the query text with the same default embedding path used for
document insertion.

## Metadata filters

Filters are standard SQL-style `where` strings:

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

For more examples by field type, see the
[Metadata filter cookbook](metadata_filter_cookbook.md).

Filter strings are applied before or during candidate selection depending on the
search path. Prefer selective filters such as tenant, language, category, and
date ranges when you can.

Quoting tips:

```python
collection.search(query, k=3, where="lang = 'en'")
collection.search(query, k=3, where='"document.lang" = \'en\'')
```

The second form quotes a field name that contains punctuation.

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

Use `filter_ids` when your application already knows the candidate IDs:

```python
rows = collection.query(filter_ids=[1, 3], return_ids_only=False)
print(rows.to_list())
```

## Retrieve vectors

```python
data = collection.query_vectors(filter_ids=[1, 2, 3])
print(data.ids)
print(data.vectors.shape)
print(data.fields)
```

`query_vectors()` without `where` or `filter_ids` returns an empty `(0, dim)`
vector array.

Use `query_vectors()` sparingly in online request paths. Returning raw vectors
is useful for evaluation, debugging, migration, and downstream numerical work,
but it increases response size.

## Batch search

```python
queries = np.random.rand(5, 4).astype(np.float32)
results = collection.batch_search(queries, k=2, return_fields=True)

for result in results:
    print(result.to_list())
```

Batch search returns one `ResultView` per query vector. A shared `where` filter
is applied to every query:

```python
results = collection.batch_search(
    queries,
    k=2,
    where="lang = 'en'",
    return_fields=True,
)
```

## Range search

Range search returns all matches within a threshold, capped by `max_results`.

```python
nearby = collection.search_range(query, threshold=0.05, max_results=100)
print(nearby.ids, nearby.distances)
```

Threshold meaning depends on the metric:

- L2, Hamming, and Jaccard return rows with distance `<= threshold`;
- inner product and cosine return rows with score `>= threshold`.

Use `max_results` as a safety cap for broad thresholds.

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

Use profiles during tuning, not as the primary application response. They are
intended to explain candidate counts and timing details.

## BM25 search

BM25 search runs over stored metadata fields.

```python
result = collection.bm25_search(
    "vector database",
    k=3,
    text_fields=["title"],
    return_fields=True,
)
print(result.to_list())
```

`text_fields=None` searches all text-like metadata fields.

BM25 search works best when text fields contain normal words, titles, tags, or
short chunks. Keep binary payloads and very large documents out of metadata
unless you actually want them searched.

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

`candidate_limit` controls how many candidates each retrieval signal may
contribute before fusion:

```python
result = collection.hybrid_search(
    vector=query,
    text="vector",
    text_fields=["title"],
    candidate_limit=50,
    k=5,
    return_fields=True,
)
```

Use a larger `candidate_limit` when recall matters and a smaller value when
latency matters.

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

## Search named vector fields

Named vector fields are searched with the same `search()` method and the
`vector_field` parameter:

```python
collection.create_vector_field("title_embedding", dim=2, metric="ip")
collection.add_named_vectors(
    "title_embedding",
    np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]], dtype=np.float32),
    ids=[1, 2, 3],
)
collection.build_index("FLAT-IP", field_name="title_embedding")
collection.commit()

title_result = collection.search(
    [0.75, 0.25],
    k=2,
    vector_field="title_embedding",
    return_fields=True,
)
print(title_result.to_list())
```

The `where` filter still uses metadata fields from the same IDs.

## Search checklist

- Build a flat index first and verify known queries.
- Choose the metric that matches your embeddings.
- Add `where` filters for tenant, language, visibility, category, and date.
- Use `return_fields=False` in hot paths unless fields are needed immediately.
- Use `search_profile()` to debug slow or surprising searches.
- Use text or hybrid search when exact terms matter.
- Use a reranker when final ordering requires a stronger model or business
  rules.
