# Quickstart

This guide walks through the main LynseDB workflow: connect, create a
collection, insert vectors, build an index, search, filter metadata, inspect
results, and clean up.

## 1. Install and import

```shell
pip install LynseDB
```

Native Linux and macOS environments are supported. Native Windows environments
are not supported; on Windows, use WSL 2 (Windows Subsystem for Linux) or
Docker.

```python
import numpy as np
import lynse

print(lynse.__version__)
```

## 2. Connect

Use local mode for a single Python process:

```python
client = lynse.VectorDBClient(uri="./lynsedb-data")
```

Use remote mode when several processes or services need the same database:

```shell
lynse serve --host 127.0.0.1 --port 7637 --data-dir ./server-data
```

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637")
```

With server authentication:

```shell
lynse serve --host 127.0.0.1 --port 7637 --data-dir ./server-data --api-key your_key
```

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="your_key")
```

Health and operations endpoints:

```shell
curl http://127.0.0.1:7637/healthz
curl http://127.0.0.1:7637/readyz
curl http://127.0.0.1:7637/metrics
curl http://127.0.0.1:7637/openapi.json
```

## 3. Create a database and collection

`drop_if_exists=True` is destructive. Use it only for tests or when you really
want to truncate existing data.

```python
db = client.create_database("quickstart", drop_if_exists=True)

collection = db.require_collection(
    "documents",
    dim=4,
    drop_if_exists=True,
    description="quickstart collection",
)
```

Open an existing collection safely:

```python
collection = db.get_collection("documents")
```

## 4. Insert vectors

Each row has a public string or non-negative integer ID. Metadata fields are
optional JSON-like dicts and can be used later for filtering, BM25 search, or
result display.

```python
collection.add(
    ids=["intro", "guide", "note-fr", "note-archive"],
    vectors=[
        [0.10, 0.20, 0.30, 0.40],
        [0.11, 0.19, 0.29, 0.39],
        [0.80, 0.10, 0.20, 0.10],
        [0.75, 0.12, 0.18, 0.12],
    ],
    fields=[
        {"title": "LynseDB intro", "lang": "en", "rank": 1, "tags": ["vector", "rust"]},
        {"title": "Vector guide", "lang": "en", "rank": 2, "tags": ["vector"]},
        {"title": "French note", "lang": "fr", "rank": 3, "tags": ["note"]},
        {"title": "Another note", "lang": "fr", "rank": 4, "tags": ["note", "archive"]},
    ],
)
```

For large dense arrays, keep the same `add()` API and choose a larger batch
size:

```python
vectors = np.random.rand(10_000, 4).astype(np.float32)
added = collection.add(ids=[f"vec-{i}" for i in range(10_000)], vectors=vectors, batch_size=5000)
print(added)
```

## 5. Build an index

Flat search is the simplest and most recall-friendly default. Use HNSW or IVF as
data grows and latency matters.

```python
collection.build_index("FLAT-L2")
print(collection.index_mode)
```

IVF uses `n_clusters`; other index families allow the argument and ignore it:

```python
collection.build_index("IVF-L2", n_clusters=256)
```

## 6. Search

```python
query = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)

result = collection.search(query, k=2, return_fields=True)

print(result.ids)
print(result.distances)
print(result.fields)
print(result.to_list())
```

Filter by metadata during vector search:

```python
result = collection.search(
    query,
    k=3,
    where="lang = 'en' AND rank <= 2",
    return_fields=True,
)
print(result.to_list())
```

For IVF and HNSW, `nprobe` controls search breadth. Higher values generally
improve recall and increase latency.

```python
result = collection.search(query, k=3, nprobe=20)
```

Approximate flat distance rounding is available for IP, L2, and cosine metrics:

```python
result = collection.search(query, k=3, approx=True, eps=1e-4)
```

Flat, PQ, RaBitQ, PolarVec, and named vector-field searches ignore `nprobe`.
Hamming and Jaccard metrics ignore `approx` and `eps`.

## 7. Query metadata and vectors

Use `query()` when you need IDs and fields. Use `query_vectors()` when you need
stored vectors too.

```python
rows = collection.query(where="tags CONTAINS 'vector'")
print(rows.ids)
print(rows.fields)

vectors = collection.query_vectors(filter_ids=[1, 2])
print(vectors.ids)
print(vectors.vectors.shape)
```

Calling `query()` or `query_vectors()` without `where` or `filter_ids` returns an
empty `ResultView`; it does not perform a full scan.

## 8. Text and hybrid search

BM25 search uses stored metadata fields:

```python
text_result = collection.bm25_search(
    "vector guide",
    k=3,
    text_fields=["title"],
    return_fields=True,
)
print(text_result.to_list())
```

Hybrid search combines vector and text candidates:

```python
hybrid = collection.hybrid_search(
    vector=query,
    text="vector",
    text_fields=["title", "tags"],
    fusion="rrf",
    k=3,
    return_fields=True,
)
print(hybrid.to_list())
```

## 9. Named and sparse vectors

Named vector fields store additional embeddings for the same IDs. This is useful
for multimodal records, for example text and image embeddings on one item.

```python
collection.create_vector_field("image", dim=3, metric="l2")

image_vectors = np.array(
    [
        [0.10, 0.20, 0.30],
        [0.12, 0.19, 0.28],
        [0.90, 0.20, 0.10],
    ],
    dtype=np.float32,
)

collection.add_named_vectors("image", image_vectors, ids=[1, 2, 3])
collection.build_index("HNSW-L2", field_name="image")
collection.commit()

image_result = collection.search(
    [0.11, 0.20, 0.29],
    k=2,
    vector_field="image",
    return_fields=True,
)
print(image_result.to_list())
```

Sparse vectors store feature-ID weights and search with inner product:

```python
collection.add_sparse_vectors(
    vectors=[
        {10: 1.0, 42: 0.5},
        {11: 1.0, 42: 0.8},
    ],
    ids=["intro", "guide"],
)
collection.commit()

sparse_result = collection.search_sparse({42: 1.0}, k=2, return_fields=True)
print(sparse_result.to_list())
```

## 10. Update, delete, and compact

Use upsert when the same external ID should be replaced or inserted:

```python
collection.upsert(
    ids="intro",
    vectors=[0.12, 0.20, 0.31, 0.41],
    fields={"title": "updated intro", "lang": "en", "rank": 1},
)
collection.commit()
```

Deletes are soft deletes. Deleted IDs disappear from search and query results,
but their raw storage is kept until compaction:

```python
collection.delete(["note-archive"])
print(collection.list_deleted_ids())

collection.restore(["note-archive"])
print(collection.list_deleted_ids())

collection.delete(["note-archive"])
removed = collection.compact()
print(removed)
```

## 11. What you learned

This quickstart touched the whole everyday workflow:

- choose local or remote mode with `VectorDBClient`;
- create a database and collection;
- insert vectors with stable public IDs and metadata fields;
- commit writes through `insert_session()`;
- build and tune an index;
- search by vector, filter by metadata, query fields, and retrieve vectors;
- use document, BM25, hybrid, named vector, and sparse vector retrieval;
- update, soft-delete, restore, and compact rows.

For a complete curriculum, continue with the
[Learning path](tutorials/learning_path.md).

## 10. ResultView

Search, query, head, tail, and range APIs return `ResultView`.

```python
result = collection.search(query, k=2, return_fields=True)

ids, distances, fields = result

print(result.ids)          # numpy array
print(result.distances)    # numpy array
print(result.fields)       # list[dict]
print(result.to_dict())
print(result.to_list())
print(result.to_json())
```

Use `to_list()` for row iteration:

```python
for row in result.to_list():
    print(row["id"], row.get("title"))
```

Optional dataframe conversions are available when the dependency is installed:

```python
result.to_pandas()
result.to_polars()
result.to_arrow()
```

## 11. Delete, restore, and compact

Deletes are soft deletes. Deleted IDs are excluded from search and can be
restored until compaction.

```python
collection.delete(["note-fr"])
print(collection.list_deleted_ids())

collection.restore(["note-fr"])
print(collection.list_deleted_ids())

collection.delete(["note-archive"])
removed = collection.compact()
print(removed)
```

## 12. Back up and close

```python
collection.flush()
collection.checkpoint()

db.snapshot_collection("documents", "./documents.snapshot")
db.export_collection("documents", "./documents-export")

collection.close()
client.close()
```
