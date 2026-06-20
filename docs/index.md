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
- Standard SQL-style metadata filtering with `where` expressions.
- Multiple vector representations per record through named vector fields.
- Dense, sparse, text, and hybrid retrieval.
- Recall-first search with explicit index and search knobs.

## Search more than embeddings

LynseDB provides native distance paths for domain data as well as embedding
vectors. They use the same IDs, metadata filters, persistence, and local or
remote client API.

| Data | Metrics | Example workloads |
| --- | --- | --- |
| Embeddings | inner product, squared L2, cosine | RAG, semantic and multimodal retrieval |
| Numeric features | Manhattan/L1 | anomaly matching, sensor and tabular features |
| Coordinates | Haversine in meters | nearby POI, fleet and device search |
| Binary fingerprints | Hamming, Jaccard/Tanimoto, Sørensen-Dice | molecular fingerprints, deduplication, genomic sketches |
| Aligned profiles | Pearson correlation distance | sensor curves, behavior profiles, gene expression |
| Distributions | Hellinger, Wasserstein-1D | model drift, topics, forecasts and histograms |

Exact Flat search supports the complete suite. HNSW supports the numeric domain
metrics, while binary Flat search uses a one-bit-per-dimension hot
representation for low memory bandwidth. Read
[Domain-aware distance metrics](tutorials/distance_metrics.md) for input
contracts and index compatibility.

## Execution modes

| Mode | How to start | Best for | Notes |
| --- | --- | --- | --- |
| Local | `lynse.VectorDBClient()` or `lynse.VectorDBClient(uri="./data")` | notebooks, scripts, single-process apps | No server and no network hop. Do not share the same local path between independent processes. |
| Remote | `lynse serve ...` plus `lynse.VectorDBClient("http://host:7637")` | web services, workers, production | Use HTTP mode when more than one process writes or searches the same data. |

## Install

```shell
pip install lynsedb
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

collection.add(
    ids=["intro", "guide", "notes-fr"],
    vectors=[
        [0.10, 0.20, 0.30, 0.40],
        [0.11, 0.19, 0.29, 0.39],
        [0.80, 0.10, 0.20, 0.10],
    ],
    fields=[
        {"title": "intro", "lang": "en"},
        {"title": "guide", "lang": "en"},
        {"title": "notes", "lang": "fr"},
    ],
)

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

- Start with the [Learning path](tutorials/learning_path.md) if you want a
  step-by-step curriculum from zero to production.
- Use the [Quickstart](quickstart.md) for the shortest complete write, index,
  search, query, and delete lifecycle.
- Read [Core concepts](tutorials/core_concepts.md) to understand clients,
  databases, collections, IDs, fields, indexes, and result objects.
- Use [Connect and deploy](tutorials/connect_and_deploy.md) to choose local or
  remote mode and configure the server.
- Use [Search and filter](tutorials/search_and_filter.md) and the
  [Metadata filter cookbook](tutorials/metadata_filter_cookbook.md) for vector
  search, standard SQL-style filters, BM25 search, hybrid search, and reranking.
- Use [Indexing guide](tutorials/indexing.md) and
  [Performance tuning](tutorials/performance_tuning.md) to choose between flat,
  HNSW, IVF, SPANN, DiskANN, and quantized indexes with explicit metric suffixes.
- Use [Domain-aware distance metrics](tutorials/distance_metrics.md) for
  coordinates, binary fingerprints, aligned profiles, and distributions.
- Use [Build a RAG workflow](tutorials/rag_workflow.md) for an end-to-end
  retrieval example.

## Stability notes

LynseDB is still evolving. Pin versions for deployments and test migrations
before upgrading. For concurrent production access, prefer the HTTP server over
sharing one local data directory across independent Python processes.
