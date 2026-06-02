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

Dense vectors are stored as `float32`. Convert arrays before insertion when you
control the embedding pipeline:

```python
vector = np.random.rand(128).astype(np.float32)
```

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

## Commit and durability

Use these calls explicitly in services:

```python
collection.flush()       # flush client and storage buffers
collection.checkpoint()  # force a durable checkpoint
collection.close()       # flush and close this handle
```

`commit()` is enough for normal write batches. `checkpoint()` is useful before
backups or controlled shutdowns.

## Common ingestion checks

```python
print(collection.shape)       # (n_vectors, dim)
print(collection.max_id)      # highest external ID
print(collection.is_id_exists(1))
print(collection.stats())
print(collection.list_fields())
```
