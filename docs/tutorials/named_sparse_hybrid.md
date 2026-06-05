# Tutorial: Named, Sparse, and Hybrid Search

This tutorial covers retrieval patterns beyond one dense vector per record.

## Named vector fields

Use named vector fields when one logical record has multiple embeddings, such as
text, image, title, body, or multilingual embeddings.

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./multi-vector-demo")
db = client.create_database("multi_vector", drop_if_exists=True)
collection = db.require_collection("products", dim=4, drop_if_exists=True)

with collection.insert_session() as session:
    session.bulk_add_items(
        [
            ([0.10, 0.20, 0.30, 0.40], 1, {"sku": "A", "name": "red shirt"}),
            ([0.11, 0.21, 0.29, 0.39], 2, {"sku": "B", "name": "blue shirt"}),
            ([0.80, 0.10, 0.20, 0.10], 3, {"sku": "C", "name": "green hat"}),
        ],
        enable_progress_bar=False,
    )
```

Create a field for image embeddings:

```python
collection.create_vector_field("image", dim=3, metric="l2")

image_vectors = np.array(
    [
        [0.10, 0.20, 0.30],
        [0.12, 0.18, 0.31],
        [0.90, 0.20, 0.10],
    ],
    dtype=np.float32,
)

collection.add_named_vectors("image", image_vectors, ids=[1, 2, 3])
collection.build_index("HNSW-L2", field_name="image")
collection.commit()
```

Search that field:

```python
result = collection.search(
    [0.11, 0.20, 0.29],
    k=2,
    vector_field="image",
    return_fields=True,
)
print(result.to_list())
```

List fields:

```python
print(collection.list_vector_fields())
```

Rules:

- `default` is reserved for the primary collection vector.
- Named field names may contain ASCII letters, digits, `_`, and `-`.
- A named field has its own dimension, metric, and index.
- `add_named_vectors()` attaches vectors to existing IDs.

Use named fields instead of separate collections when:

- the same product, document, or media item has several embeddings;
- all embeddings should share one external ID and one metadata dict;
- filters such as tenant, language, category, and visibility should apply to
  every retrieval signal.

Use separate collections when dimensions, lifecycle, or permissions differ so
much that the records should be managed independently.

## Named field metrics and indexes

Choose the metric when creating the field:

```python
collection.create_vector_field("title", dim=384, metric="cos")
collection.create_vector_field("image", dim=512, metric="l2")
collection.create_vector_field("click_model", dim=128, metric="ip")
```

Build the field index with `field_name=...`:

```python
collection.build_index("HNSW-Cos", field_name="title")
collection.build_index("IVF-L2", field_name="image", n_clusters=256)
```

Search the field with `vector_field=...`:

```python
result = collection.search(
    title_query_vector,
    k=10,
    vector_field="title",
    where="sku = 'A'",
    return_fields=True,
)
```

Named vector fields attach to existing IDs. Insert the primary row first, then
add named vectors.

## Sparse vectors

Sparse vectors are useful for token, keyword, or feature-ID signals.

```python
collection.add_sparse_vectors(
    vectors=[
        {101: 1.0, 205: 0.5},
        {101: 0.8, 333: 0.7},
        {999: 1.0},
    ],
    ids=[1, 2, 3],
)
collection.commit()
```

Search sparse vectors with inner product:

```python
result = collection.search_sparse(
    {101: 1.0},
    k=3,
    return_fields=True,
)
print(result.to_list())
```

The sparse vector can be a dict or a list of `(feature_id, weight)` pairs.

Both forms are valid:

```python
collection.search_sparse({101: 1.0, 205: 0.5}, k=3)
collection.search_sparse([(101, 1.0), (205, 0.5)], k=3)
```

Sparse vector tips:

- feature IDs must be non-negative integers;
- weights are converted to floats;
- sparse search uses inner product;
- use sparse vectors for lexical features, categories, tags, learned sparse
  encoders, or business features;
- use metadata filters with `where=...` to restrict candidates.

```python
result = collection.search_sparse(
    {101: 1.0},
    k=10,
    where="sku = 'A'",
    return_fields=True,
)
```

## Text search

Text search uses BM25 over metadata fields. Text is indexed from stored fields
when you insert or update rows.

```python
result = collection.text_search(
    "shirt",
    k=3,
    text_fields=["name"],
    return_fields=True,
)
print(result.to_list())
```

Search one or more fields:

```python
result = collection.text_search(
    "red shirt",
    k=10,
    text_fields=["name", "sku"],
    where="sku IN ('A', 'B')",
    return_fields=True,
)
```

Pass `text_fields=None` to search all text-like metadata fields. Explicit field
lists are easier to reason about in production.

## Hybrid search

Hybrid search fuses dense vector and BM25 text results.

```python
query_vector = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)

result = collection.hybrid_search(
    vector=query_vector,
    text="shirt",
    text_fields=["name"],
    fusion="rrf",
    k=3,
    return_fields=True,
)
print(result.to_list())
```

Use weighted fusion when you know how much each signal should matter:

```python
result = collection.hybrid_search(
    vector=query_vector,
    text="shirt",
    fusion="weighted",
    vector_weight=0.8,
    text_weight=0.2,
    k=3,
)
```

Fusion modes:

| Mode | Use when |
| --- | --- |
| `rrf` | Vector and text scores are on different scales. This is the safest default. |
| `weighted` | Scores are calibrated enough that `vector_weight` and `text_weight` are meaningful. |

Control candidate breadth:

```python
result = collection.hybrid_search(
    vector=query_vector,
    text="shirt",
    text_fields=["name"],
    candidate_limit=100,
    k=10,
    return_fields=True,
)
```

Increase `candidate_limit` for recall. Decrease it for latency.

## Rerank candidates

Reranking is often the final stage in retrieval-augmented applications.

```python
def rerank(payload):
    rows = []
    for item in payload["items"]:
        field = item.get("field") or {}
        score = 1.0 if field.get("sku") == "A" else 0.1
        rows.append((item["id"], score))
    return rows

result = collection.hybrid_search(
    vector=query_vector,
    text="shirt",
    text_fields=["name"],
    k=20,
    reranker=rerank,
    rerank_k=5,
    return_fields=True,
)
```

When a reranker needs metadata, set `return_fields=True` or
`rerank_with_fields=True`.

The reranker callback receives a payload like:

```python
{
    "query": {
        "type": "hybrid_search",
        "text": "shirt",
    },
    "items": [
        {"id": 1, "score": 0.8, "field": {"sku": "A", "name": "red shirt"}},
    ],
}
```

It may return:

- ordered IDs, such as `[2, 1, 3]`;
- `(id, score)` pairs, such as `[(2, 0.99), (1, 0.75)]`;
- a score array aligned with input items;
- a mapping such as `{2: 0.99, 1: 0.75}`.

Use reranking for cross-encoders, LLM scoring, personalization, freshness,
availability, permissions, or business rules that should run after candidate
retrieval.

## End-to-end multimodal example

This example stores product text embeddings as the primary vector and image
embeddings as a named vector field:

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./multimodal-products")
db = client.create_database("store", drop_if_exists=True)
collection = db.require_collection("products", dim=4, drop_if_exists=True)

products = [
    ([0.10, 0.20, 0.30, 0.40], 1, {"sku": "A", "name": "red shirt", "category": "apparel"}),
    ([0.11, 0.21, 0.29, 0.39], 2, {"sku": "B", "name": "blue shirt", "category": "apparel"}),
    ([0.80, 0.10, 0.20, 0.10], 3, {"sku": "C", "name": "green hat", "category": "accessory"}),
]

with collection.insert_session() as session:
    session.bulk_add_items(products, enable_progress_bar=False)

collection.create_vector_field("image", dim=3, metric="l2")
collection.add_named_vectors(
    "image",
    np.array(
        [
            [0.10, 0.20, 0.30],
            [0.12, 0.18, 0.31],
            [0.90, 0.20, 0.10],
        ],
        dtype=np.float32,
    ),
    ids=[1, 2, 3],
)

collection.add_sparse_vectors(
    [
        {10: 1.0, 20: 0.5},
        {10: 0.9, 30: 0.8},
        {40: 1.0},
    ],
    ids=[1, 2, 3],
)

collection.build_index("FLAT-L2")
collection.build_index("HNSW-L2", field_name="image")
collection.commit()

text_like = collection.search(
    [0.10, 0.20, 0.30, 0.40],
    k=2,
    where="category = 'apparel'",
    return_fields=True,
)

image_like = collection.search(
    [0.11, 0.20, 0.30],
    k=2,
    vector_field="image",
    where="category = 'apparel'",
    return_fields=True,
)

keyword_like = collection.search_sparse(
    {10: 1.0},
    k=2,
    where="category = 'apparel'",
    return_fields=True,
)

print(text_like.to_list())
print(image_like.to_list())
print(keyword_like.to_list())
```

## Choosing the right retrieval signal

| Need | Use |
| --- | --- |
| Semantic similarity from one embedding model | Primary vector search. |
| Multiple embeddings per record | Named vector fields. |
| Feature-ID weights or learned sparse representations | Sparse vector search. |
| Exact or keyword-like text matching | `text_search()`. |
| Semantic plus lexical recall | `hybrid_search()`. |
| Final ordering with richer logic | `reranker`. |
