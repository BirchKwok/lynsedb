# Tutorial: Learning Path

This page is the recommended order for learning LynseDB. It starts with the
smallest useful mental model, then adds write patterns, search, filters,
indexes, advanced retrieval, deployment, operations, and troubleshooting.

You can read the pages in order, or jump to the stage that matches your current
project.

## Stage 1: Know the data model

Start with [Core concepts](core_concepts.md).

After this stage, you should know:

- a `VectorDBClient` chooses local or remote mode;
- a database groups collections;
- a collection stores primary dense vectors, public IDs, and metadata fields;
- named vector fields attach extra dense embeddings to existing IDs;
- sparse vectors attach feature-weight maps to existing IDs;
- indexes change recall, latency, memory, disk usage, and build time;
- every search and query returns a `ResultView`.

Minimal shape:

```python
import lynse

client = lynse.VectorDBClient(uri="./data")
db = client.create_database("app")
collection = db.require_collection("docs", dim=768)
```

## Stage 2: Connect and create storage

Read [Connect and deploy](connect_and_deploy.md) and
[Databases and collections](databases_collections.md).

After this stage, you should be able to:

- choose embedded local mode for one process;
- choose HTTP mode for multi-process or service deployments;
- start a server with `lynse serve`;
- configure API keys, limits, timeouts, and data directories;
- create, open, list, describe, and drop databases and collections;
- understand why `drop_if_exists=True` is destructive.

## Stage 3: Insert data safely

Read [Add vectors](add_vectors.md).

After this stage, you should be able to:

- choose stable public IDs as strings or non-negative integers;
- insert one row or many rows;
- insert documents and let LynseDB embed them when you do not pass vectors;
- attach JSON-like metadata fields;
- use `insert_session()` so writes commit on success;
- upsert existing IDs;
- decide when to call `commit()`, `flush()`, `checkpoint()`, and `close()`.

## Stage 4: Search and inspect results

Read [Search and filter](search_and_filter.md),
[Metadata filter cookbook](metadata_filter_cookbook.md), and
[ResultView](../result_view.md).

After this stage, you should be able to:

- run vector search with `search()`;
- run batch vector search with `batch_search()`;
- run range search with `search_range()`;
- query fields without vector search using `query()`;
- retrieve stored vectors using `query_vectors()`;
- filter by string, number, boolean, array, and ISO date fields;
- convert `ResultView` objects to lists, dicts, NumPy arrays, JSON, pandas,
  polars, or Arrow.

## Stage 5: Tune indexes

Read [Indexing guide](indexing.md) and [Performance tuning](performance_tuning.md).

After this stage, you should be able to:

- select `FLAT`, `HNSW`, `IVF`, `DiskANN`, SQ8, PQ, RaBitQ, PolarVec, or binary
  indexes;
- match the index metric to your embedding model;
- use `n_clusters` for IVF;
- tune `nprobe` for IVF and HNSW;
- use a flat index as the recall baseline;
- profile queries with `search_profile()`;
- understand when quantization is worth the quality tradeoff.

## Stage 6: Combine retrieval signals

Read [Named, sparse, and hybrid search](named_sparse_hybrid.md).

After this stage, you should be able to:

- store text and image embeddings on the same ID;
- build indexes for named vector fields;
- add sparse feature vectors;
- run BM25 search over metadata fields;
- combine vector and text candidates with RRF or weighted fusion;
- plug in a custom reranker.

## Stage 7: Build an application workflow

Read [Build a RAG workflow](rag_workflow.md).

After this stage, you should have a complete pattern for:

- chunking source documents;
- generating or plugging in embeddings;
- storing text chunks and metadata;
- filtering by tenant, source, or timestamp;
- retrieving candidate context;
- optionally reranking;
- building a prompt for an LLM or another downstream system.

## Stage 8: Operate LynseDB

Read [Backup and maintenance](operations.md) and
[Troubleshooting](troubleshooting.md).

After this stage, you should be able to:

- monitor `/healthz`, `/readyz`, `/metrics`, and `/openapi.json`;
- create and restore snapshots;
- export and import portable collection data;
- soft-delete, restore, and compact rows;
- run Docker, systemd, or Kubernetes deployments;
- diagnose common dimension, ID, filter, auth, and server-limit errors.

## A complete learning script

The following script combines the beginner path into one runnable local example:

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./learn-lynsedb")
db = client.create_database("learn", drop_if_exists=True)
collection = db.require_collection("docs", dim=4, drop_if_exists=True)

collection.add(
    ids=["intro", "guide", "notes-fr"],
    vectors=[
        [0.10, 0.20, 0.30, 0.40],
        [0.11, 0.19, 0.31, 0.41],
        [0.80, 0.10, 0.20, 0.10],
    ],
    fields=[
        {"title": "intro", "lang": "en", "rank": 1},
        {"title": "guide", "lang": "en", "rank": 2},
        {"title": "notes", "lang": "fr", "rank": 3},
    ],
)

collection.build_index("FLAT-L2")

query = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)
result = collection.search(
    query,
    k=2,
    where="lang = 'en'",
    return_fields=True,
)
print(result.to_list())

collection.delete(["notes-fr"])
collection.checkpoint()
client.close()
```

When this script makes sense, the rest of the tutorials are refinements of the
same pattern.
