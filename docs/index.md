# LynseDB

LynseDB is a high-performance, low-cost vector database with a Python-first API
and a Rust storage/search backend. It can run embedded inside a single Python
process or as an HTTP server for multi-process and production deployments.

The API is intentionally small: create a database, create a collection, write
vectors with metadata, build an index when you need faster search, and query the
same collection locally or remotely with the same high-level Python calls.

Use it when you want vector search that starts as an embedded local library,
can move to a service when your app grows, and does not require a heavy
database cluster for common semantic search, RAG, agent memory, and multimodal
retrieval workloads.

## What LynseDB is good at

- Local development, notebooks, scripts, tests, and small services that want an
  embedded vector store.
- Remote deployments where several workers or services need to share one
  database process.
- Low-operational-cost deployments where predictable resource use matters.
- Metadata filtering with SQL-like `where` expressions.
- Multiple vector representations per record through named vector fields.
- Dense, sparse, text, and hybrid retrieval.
- Recall-first search with explicit index and search knobs.

## Execution modes

| Mode | How to start | Best for | Notes |
| --- | --- | --- | --- |
| Local | `lynse.VectorDBClient()` or `lynse.VectorDBClient(uri="./data")` | notebooks, scripts, single-process apps | No server and no network hop. Do not share the same local path between independent processes. |
| Remote | `lynse serve ...` plus `lynse.VectorDBClient("http://host:7637")` | web services, workers, production | Use HTTP mode when more than one process writes or searches the same data. |

## Install

```shell
pip install LynseDB
```

Python 3.9 or newer is required. Wheels are published for common macOS, Linux,
and Linux environments. Native Windows environments are not supported; on
Windows, use WSL 2 (Windows Subsystem for Linux) or Docker.

## First search

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./lynsedb-data")
db = client.create_database("demo", drop_if_exists=True)
collection = db.require_collection("documents", dim=4, drop_if_exists=True)

items = [
    ([0.10, 0.20, 0.30, 0.40], 1, {"title": "intro", "lang": "en"}),
    ([0.11, 0.19, 0.29, 0.39], 2, {"title": "guide", "lang": "en"}),
    ([0.80, 0.10, 0.20, 0.10], 3, {"title": "notes", "lang": "fr"}),
]

with collection.insert_session() as session:
    session.bulk_add_items(items, enable_progress_bar=False)

collection.build_index("FLAT-L2")

result = collection.search(
    np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32),
    k=2,
    where="lang = 'en'",
    return_fields=True,
)

print(result.to_list())
```

## Learn next

- Start with the [Quickstart](quickstart.md) for the complete write, index,
  search, query, and delete lifecycle.
- Use [Connect and deploy](tutorials/connect_and_deploy.md) to choose local or
  remote mode.
- Use [Search and filter](tutorials/search_and_filter.md) for vector search,
  metadata filtering, text search, and reranking.
- Use [Indexing guide](tutorials/indexing.md) to choose between flat, HNSW, IVF,
  DiskANN, and quantized indexes.
- Use [Named, sparse, and hybrid search](tutorials/named_sparse_hybrid.md) for
  multimodal and retrieval-augmented applications.

## Stability notes

LynseDB is still evolving. Pin versions for deployments and test migrations
before upgrading. For concurrent production access, prefer the HTTP server over
sharing one local data directory across independent Python processes.
