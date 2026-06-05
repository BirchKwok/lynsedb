# Tutorial: Add Vectors

This tutorial covers write patterns, commit behavior, IDs, metadata fields, and
high-throughput ingestion.

## Setup

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./lynsedb-ingest")
db = client.create_database("ingest_demo", drop_if_exists=True)
collection = db.require_collection("items", dim=128, drop_if_exists=True)
```

## IDs and vectors

LynseDB uses your integer IDs as stable external IDs. IDs should be unique inside
one collection. Incrementing IDs are recommended because they keep storage
layout predictable and make debugging easier.

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
write:

```python
embedding = np.asarray(embedding, dtype=np.float32)
```

Check dimensions at the boundary of your embedding pipeline:

```python
if embedding.shape != (128,):
    raise ValueError(f"expected a 128-dimensional vector, got {embedding.shape}")
```

## Metadata field design

Fields are optional, but they are usually what make retrieval useful:

```python
field = {
    "doc_id": "manual-001",
    "chunk": 3,
    "tenant": "acme",
    "lang": "en",
    "title": "Install LynseDB",
    "text": "pip install LynseDB",
    "tags": ["install", "python"],
    "created_at": "2026-06-05T10:00:00Z",
}
```

Guidelines:

- store values you will filter on, display, or rerank with;
- keep field value types stable across rows;
- store dates as ISO-8601 strings;
- keep very large raw documents outside LynseDB when you only need short chunks;
- use arrays for tags and `CONTAINS` filters;
- use a string source ID in metadata if your application ID is not an integer.

## Add one item

```python
with collection.insert_session() as session:
    inserted_id = session.add_item(
        vector=np.random.rand(128).astype(np.float32),
        id=1,
        field={
            "category": "docs",
            "score": 0.91,
            "published": True,
            "tags": ["vector", "python"],
            "created_at": "2026-06-02",
        },
    )

print(inserted_id)
```

`insert_session()` commits when the block succeeds. If an exception is raised,
pending buffered writes in that session are discarded.

## Add many items

Use `bulk_add_items()` when each row has metadata:

```python
items = [
    (
        np.random.rand(128).astype(np.float32),
        i,
        {"category": f"cat_{i % 3}", "score": float(i) / 100.0},
    )
    for i in range(2, 1002)
]

with collection.insert_session() as session:
    ids = session.bulk_add_items(
        items,
        batch_size=1000,
        enable_progress_bar=False,
    )

print(len(ids))
```

Tuple shape can be `(vector, id, field)` or `(vector, id)`.

For generators or streams, accumulate manageable batches before calling
`bulk_add_items()`:

```python
batch = []
next_id = 10_000

for embedding, metadata in embedding_stream:
    batch.append((np.asarray(embedding, dtype=np.float32), next_id, metadata))
    next_id += 1

    if len(batch) == 1000:
        with collection.insert_session() as session:
            session.bulk_add_items(batch, enable_progress_bar=False)
        batch.clear()

if batch:
    with collection.insert_session() as session:
        session.bulk_add_items(batch, enable_progress_bar=False)
```

## Buffer control

`add_item()` is buffered by default. You can tune or disable buffering:

```python
with collection.insert_session() as session:
    session.add_item(vector=np.zeros(128, dtype=np.float32), id=2001, buffer_size=5000)
    session.add_item(vector=np.ones(128, dtype=np.float32), id=2002, buffer_size=False)
```

- `buffer_size=True` uses the client default batch size.
- `buffer_size=False` writes the item immediately.
- `buffer_size=<int>` flushes whenever the buffer reaches that size.

Prefer the default buffering for normal ingestion. Disable buffering only when
you need the item to be sent to storage immediately.

## High-throughput dense ingestion

Use `bulk_add_binary()` for large arrays when you do not need per-row metadata in
the same call. IDs are assigned automatically after the current max ID.

```python
vectors = np.random.rand(100_000, 128).astype(np.float32)

added = collection.bulk_add_binary(
    vectors,
    batch_size=50_000,
    enable_progress_bar=False,
)
collection.commit()

print(added)
print(collection.shape)
```

`bulk_add_binary()` does not take fields in the same call. It assigns sequential
integer IDs starting after the current `max_id`. Use it for benchmarks, dense
feature caches, or pipelines where metadata is stored elsewhere.

If you need exact IDs and fields, use `bulk_add_items()` instead.

## Add metadata later

If you inserted vectors first and want to replace or add metadata later, use
upsert methods:

```python
collection.upsert_items(
    [
        (np.random.rand(128).astype(np.float32), 3001, {"category": "updated"}),
        (np.random.rand(128).astype(np.float32), 3002, {"category": "updated"}),
    ],
    enable_progress_bar=False,
)
collection.commit()
```

Two-tuples `(vector, id)` update only vectors and preserve existing fields.
Three-tuples `(vector, id, field)` replace the row fields.

Use `upsert_item()` for one row:

```python
collection.upsert_item(
    np.random.rand(128).astype(np.float32),
    id=3001,
    field={"category": "single-update"},
)
collection.commit()
```

Inside one `upsert_items()` batch, duplicate IDs are rejected so one batch has a
clear final state.

## Commit and durability

Use these calls explicitly in services:

```python
collection.flush()       # flush client and storage buffers
collection.checkpoint()  # force a durable checkpoint
collection.close()       # flush and close this handle
```

`commit()` is enough for normal write batches. `checkpoint()` is useful before
backups or controlled shutdowns.

The practical rule:

| Call | Use it when |
| --- | --- |
| `insert_session()` | You want automatic commit on success and discard on exception. |
| `commit()` | A write batch is complete and should be visible durably. |
| `flush()` | You want pending buffers and bytes flushed without clearing the WAL. |
| `checkpoint()` | You are about to back up, snapshot, or shut down cleanly. |
| `close()` | The process is done with the collection handle. |

## Common ingestion checks

```python
print(collection.shape)       # (n_vectors, dim)
print(collection.max_id)      # highest external ID
print(collection.is_id_exists(1))
print(collection.stats())
print(collection.list_fields())
```

## Common ingestion errors

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| dimension error | Vector length does not match collection `dim`. | Validate embedding shape before insertion. |
| duplicate ID error | The ID already exists and you used `add_item()` or `bulk_add_items()`. | Use a new ID or use `upsert_item()` / `upsert_items()`. |
| missing metadata in results | Search was called without `return_fields=True`. | Set `return_fields=True` or query fields separately. |
| empty query result | `query()` was called without `where` or `filter_ids`. | Pass a filter or explicit IDs. |
| large HTTP request rejected | Server payload or batch limit was exceeded. | Lower `batch_size` or increase server limits. |

## Ingestion recipe for production

1. Generate embeddings in `float32`, or choose `dtypes="float16"` when storage
   footprint matters more than full precision.
2. Assign stable integer IDs.
3. Store source identifiers, tenant, language, timestamps, and display text in
   metadata fields.
4. Insert in batches with `insert_session()` and `bulk_add_items()`.
5. Call `build_index()` or rebuild the index after a large load.
6. Run a few known queries and filters.
7. Call `checkpoint()` before snapshot or controlled shutdown.
