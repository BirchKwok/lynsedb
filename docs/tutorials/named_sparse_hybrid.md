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
