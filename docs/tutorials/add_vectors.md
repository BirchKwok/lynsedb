# Tutorial: Add Vectors and Documents

This tutorial covers the current write path in LynseDB: one public `add()`
entry point for dense vectors, documents that should be embedded automatically,
IDs, metadata fields, batching, and durability.

## Setup

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./lynsedb-ingest")
db = client.create_database("ingest_demo", drop_if_exists=True)
collection = db.require_collection("items", dim=128, drop_if_exists=True)
```

## IDs and Internal IDs

`add()` requires a public ID for every row. Public IDs can be non-negative
integers or non-empty strings:

```python
ids = ["manual-001#0", "manual-001#1", 42]
```

LynseDB stores these as external IDs. The Rust backend assigns a separate
monotonic internal integer ID for storage and clustering. This keeps the storage
layout simple while letting applications use natural IDs such as document
chunks, UUIDs, or existing numeric keys.

Good ID practice:

- keep public IDs unique inside one collection;
- use stable IDs from your source system when possible;
- use strings for composite IDs such as `"doc-123#chunk-4"`;
- do not depend on internal IDs for application logic.

## Vector Types

Dense vectors are stored as `float32` by default. Use `dtypes="float16"` when
you want half-precision vector storage:

```python
collection = db.require_collection("items_f16", dim=128, dtypes="float16")
```

Convert arrays before insertion when you control the embedding pipeline:

```python
vector = np.random.rand(128).astype(np.float32)
```

If your embedding model returns Python lists, `float64`, or `float16` arrays,
LynseDB accepts them through the Python API and computes distances in `float32`.
For `float16` collections, stored vectors are quantized to half precision on
write.

Check dimensions at the boundary of your embedding pipeline:

```python
if vector.shape != (128,):
    raise ValueError(f"expected a 128-dimensional vector, got {vector.shape}")
```

## Metadata Field Design

Fields are optional, but they are usually what make retrieval useful:

```python
field = {
    "doc_id": "manual-001",
    "chunk": 3,
    "tenant": "acme",
    "lang": "en",
    "title": "Install LynseDB",
    "text": "pip install lynsedb",
    "tags": ["install", "python"],
    "created_at": "2026-06-05T10:00:00Z",
}
```

Guidelines:

- store values you will filter on, display, or rerank with;
- keep field value types stable across rows;
- store dates as ISO-8601 strings;
- keep very large raw documents outside LynseDB when you only need short chunks;
- use arrays for tags and `CONTAINS` filters.

## Add One Row

Use `add()` for both single-row and multi-row insertion:

```python
inserted = collection.add(
    ids="manual-001#3",
    vectors=np.random.rand(128).astype(np.float32),
    fields={
        "category": "docs",
        "score": 0.91,
        "published": True,
        "tags": ["vector", "python"],
        "created_at": "2026-06-02",
    },
)

print(inserted)
```

Single inputs are normalized to one-row batches. The write is flushed for the
batch before `add()` returns.

## Add Many Rows

Pass lists or NumPy arrays when inserting many vectors:

```python
ids = [f"item-{i}" for i in range(1000)]
vectors = np.random.rand(1000, 128).astype(np.float32)
fields = [
    {"category": f"cat_{i % 3}", "score": float(i) / 100.0}
    for i in range(1000)
]

inserted = collection.add(
    ids=ids,
    vectors=vectors,
    fields=fields,
    batch_size=1000,
)

print(len(inserted))
```

`batch_size` controls how the client splits large calls. Use a batch size that
fits memory comfortably and keeps request payloads reasonable in HTTP mode.

## Add Documents With Automatic Embedding

When you pass `documents` without `vectors`, LynseDB embeds the text for you:

```python
collection.add(
    ids=["doc-pineapple", "doc-orange"],
    documents=[
        "This is a document about pineapple recipes.",
        "This is a document about oranges and citrus fruit.",
    ],
    fields=[
        {"category": "food", "lang": "en"},
        {"category": "food", "lang": "en"},
    ],
)
```

Install the optional local embedding adapter explicitly for repeatable
environments:

```shell
pip install "lynsedb[embeddings]"
```

If `fastembed` is missing, LynseDB can still try to install it lazily on first
document use. Set `LYNSE_AUTO_INSTALL_EMBEDDINGS=0` to disable that behavior.
The raw text is also stored in the row field as `document` so it can be
returned, filtered, or used for BM25 search.

Search with text directly:

```python
result = collection.search(
    document="pineapple recipes",
    k=5,
    return_fields=True,
)

print(result.ids)
print(result.fields)
```

Use `bm25_search()` when you explicitly want lexical BM25 retrieval instead of
embedding search:

```python
lexical = collection.bm25_search("pineapple recipes", k=5, return_fields=True)
```

## Streaming or High-Throughput Ingestion

For larger pipelines, group writes with `insert_session()` and call `add()` per
batch:

```python
batch_ids = []
batch_vectors = []
batch_fields = []

with collection.insert_session() as session:
    for source_id, embedding, metadata in embedding_stream:
        batch_ids.append(source_id)
        batch_vectors.append(np.asarray(embedding, dtype=np.float32))
        batch_fields.append(metadata)

        if len(batch_ids) == 1000:
            session.add(
                ids=batch_ids,
                vectors=batch_vectors,
                fields=batch_fields,
                batch_size=1000,
            )
            batch_ids.clear()
            batch_vectors.clear()
            batch_fields.clear()

    if batch_ids:
        session.add(ids=batch_ids, vectors=batch_vectors, fields=batch_fields)
```

`insert_session()` commits when the block succeeds. If an exception is raised,
pending writes from that session are discarded.

## Update Existing Rows

Use upsert methods only when you intentionally want to replace existing rows:

```python
collection.upsert(
    ids="manual-001#3",
    vectors=np.random.rand(128).astype(np.float32),
    fields={"category": "updated"},
)
collection.commit()
```

`add()` rejects duplicate public IDs so accidental repeated ingestion is visible
early. `upsert()` is the explicit replacement path.

## Commit and Durability

Use these calls explicitly in services:

```python
collection.flush()       # flush client and storage buffers
collection.checkpoint()  # force a durable checkpoint
collection.close()       # flush and close this handle
```

The practical rule:

| Call | Use it when |
| --- | --- |
| `add()` | You want a simple write-through batch. |
| `insert_session()` | You want grouped ingestion with automatic commit on success. |
| `with collection:` | You want normal collection calls committed automatically on success. |
| `commit()` | A write batch is complete and should be visible durably. |
| `flush()` | You want pending buffers and bytes flushed without clearing the WAL. |
| `checkpoint()` | You are about to back up, snapshot, or shut down cleanly. |
| `close()` | The process is done with the collection handle. |

## Common Ingestion Checks

```python
print(collection.shape)       # (n_vectors, dim)
print(collection.max_id)      # highest internal numeric ID, mostly diagnostic
print(collection.is_id_exists("manual-001#3"))
print(collection.stats())
print(collection.list_fields())
```

## Common Ingestion Errors

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| dimension error | Vector length does not match collection `dim`. | Validate embedding shape before insertion. |
| duplicate ID error | The public ID already exists and you used `add()`. | Use a new ID or use `upsert()`. |
| missing metadata in results | Search was called without `return_fields=True`. | Set `return_fields=True` or query fields separately. |
| empty query result | `query()` was called without `where` or `filter_ids`. | Pass a filter or explicit IDs. |
| large HTTP request rejected | Server payload or batch limit was exceeded. | Lower `batch_size` or increase server limits. |

## Ingestion Recipe for Production

1. Generate embeddings in `float32`, or choose `dtypes="float16"` when storage
   footprint matters more than full precision.
2. Assign stable public IDs as strings or non-negative integers.
3. Store tenant, language, timestamps, display text, and reranking payloads in
   metadata fields.
4. Insert in batches with `add()` or `insert_session().add()`.
5. Call `build_index()` or rebuild the index after a large load.
6. Run a few known vector, document, BM25, and filter queries.
7. Call `checkpoint()` before snapshot or controlled shutdown.
