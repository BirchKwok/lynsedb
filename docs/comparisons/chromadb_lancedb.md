# LynseDB vs ChromaDB, LanceDB, and USEARCH

LynseDB is built for teams that want more than a local vector index: one
Python-first API for embedded prototyping, document retrieval, metadata filters,
hybrid search, HTTP service deployment, and a lightweight sharded cluster path.
It keeps the local developer experience simple while using a Rust-backed
storage and search core underneath.

ChromaDB, LanceDB, and USEARCH are useful tools, but they optimize for narrower
or different workflows. ChromaDB is document-first, simple to adopt, and widely
integrated in RAG tutorials. LanceDB is columnar, fast at local batch ingest,
and compact on disk. USEARCH is a fast vector-only HNSW index library rather
than a full vector database. LynseDB is the better fit when you want local
speed, exact and filtered retrieval, document/search features, and a path to a
self-hosted service from the same application code.

Use this page when deciding whether LynseDB is a good fit for a Chroma-style or
LanceDB-style local workflow, whether a USEARCH-style vector index is enough,
or when migrating an embedded retrieval prototype into a self-hosted deployment.

## Quick Comparison

| Area | LynseDB | ChromaDB | LanceDB | USEARCH |
| --- | --- | --- | --- | --- |
| Primary workflow | Unified collection API for vectors, documents, metadata, sparse vectors, BM25, hybrid search, and named vector fields. | Document and embedding collections for AI apps. | Local or cloud vector tables backed by Lance/Arrow-style columnar storage. | In-process vector similarity index for applications that manage their own records. |
| Local usage | Embedded local client backed directly by the Rust engine. | Embedded local client and persistent client. | Embedded local database path with table APIs. | Embedded index object with explicit save/load. |
| Service usage | HTTP server with API keys, metrics, OpenAPI, Docker, systemd, Kubernetes examples, and coordinator-backed cluster mode. | HTTP server and managed cloud options. | Managed/cloud and server-oriented deployment options. | Bring your own service layer. |
| Scale path | Local, single HTTP service, lightweight self-hosted sharded cluster. | Local, server, cloud. | Local table, cloud/server deployments. | Library-level index scaling; application owns distribution. |
| Default document path | `add(documents=...)` and `search(document=...)` use the default local embedding adapter. | Add/query documents directly. | Stores text columns and supports full-text / hybrid paths when indexes are configured. | No document abstraction in this benchmark path. |
| Index defaults | New collections build a `FLAT-IP` index lazily after the first primary vector write. | Designed for low-friction collection search. | Table search over vector columns with optional indexes depending on workload. | HNSW vector index. |
| Retrieval mix | Dense vector search, metadata filters, BM25, sparse vector search, hybrid fusion, named vector fields, range search, and external reranking. | Dense vector search, metadata, text-oriented search features. | Dense vector search, filtering, full-text search, and hybrid search. | Dense vector search only unless the application layers on filtering or text retrieval. |
| Storage posture | Rust storage/search core with mmap storage, WAL, snapshots, restore, export/import, and multiple index families. | Chroma storage/query stack. | Columnar Lance storage, compact local files. | Compact persisted vector index. |
| Operational posture | Strong self-hosted and embedded path, with optional cluster coordination. | Strong managed-service path. | Strong analytics/table-oriented vector workflow. | Strong low-level library posture. |

## Feature Comparison

This table compares native, documented product capabilities rather than features
that can be assembled in application code. `Partial` means the product supports
the general workflow but not the same built-in scope or deployment path. The
comparison reflects the versions and benchmark adapters recorded below; product
capabilities change, so validate requirements against the version you plan to
deploy.

| Capability | LynseDB | ChromaDB | LanceDB | USEARCH |
| --- | :---: | :---: | :---: | :---: |
| One Python client for embedded, self-hosted HTTP, and self-hosted sharded-cluster deployments | **Yes** | Partial | Partial | No |
| Dense vector search and metadata filtering | Yes | Yes | Yes | Vector search only |
| BM25/full-text plus dense hybrid retrieval | Yes | Partial | Yes | No |
| Native sparse-vector search | **Yes** | No | Partial | Sparse vector types only; no database retrieval layer |
| Named vector fields for multiple embeddings on one record | **Yes** | No | Yes | Multiple indexes must be managed by the application |
| External rerank hook in the collection search workflow | **Yes** | No | Yes | No |
| Range search in the collection API | **Yes** | No | Yes | No database collection API |
| Native geospatial distance with Haversine results in meters | **Yes** | No | No | Haversine metric, but no database field/filter layer |
| Native binary-fingerprint similarity: Hamming, Jaccard/Tanimoto, and Dice | **Yes** | No | Partial | Hamming/Jaccard metrics |
| Native distribution/profile distances: Hellinger, Jensen-Shannon, Wasserstein-1D, Bray-Curtis, and correlation | **Yes** | No | No | No |
| Automatic packed-binary flat scan representation | **Yes** | No | No | No |
| WAL, snapshots/restore, and export/import in the self-hosted product | **Yes** | Partial | Partial | Save/load index only |
| Built-in API keys, health/readiness, metrics, and OpenAPI for self-hosting | **Yes** | Partial | Partial | No |
| Coordinator fan-out, stable hash sharding, replica mirroring, and primary promotion | **Yes** | No self-hosted equivalent | No lightweight self-hosted equivalent | No |

The bold LynseDB entries are the main differentiators in this comparison. The
strongest distinction is not any single checkbox: LynseDB exposes specialized
similarity metrics, full retrieval primitives, operational APIs, and an
incremental embedded-to-cluster path through one collection/client model.

## Updated Benchmark Snapshot

The latest comparable float32 run in
[`vector_database_benchmarks.md`](vector_database_benchmarks.md) was recorded on
2026-06-20. It uses 100,000 normalized 128-dimensional vectors, 100 queries,
top-k 10, and batch insert APIs. LynseDB and LanceDB target exact search;
USEARCH is configured as a vector-only HNSW index with
`expansion_search=128`. ChromaDB is included as a persistent local HNSW
collection; its approximate recall should be read alongside its latency.

| Metric | LynseDB float32 | ChromaDB | LanceDB | USEARCH |
| --- | ---: | ---: | ---: | ---: |
| Batch ingest vectors/s | **73,399** | 2,108 | 68,123 | 10,578 |
| Disk after ingest MB | 69.13 | 162.42 | **55.76** | 63.03 |
| Vector search mean ms | 0.661 | 1.233 | 14.581 | **0.555** |
| Vector search recall@10 | **1.0000** | 0.5180 | **1.0000** | 0.6000 |
| Filtered search mean ms | **0.178** | 37.354 | 16.692 | n/a |
| Filtered recall@10 | **1.0000** | 0.9990 | **1.0000** | n/a |
| Hybrid search mean ms | **4.809** | n/a | 17.810 | n/a |
| Startup mean ms | 2.087 | 13.995 | 2.251 | **0.036** |

On this workload, LynseDB combines exact recall with substantially lower vector,
filtered, and hybrid-search latency than LanceDB. USEARCH has the lowest raw
vector-search latency and fastest startup, but its approximate result reaches
0.600 recall@10 and its adapter has no database-level filtered or hybrid search.
ChromaDB also trades recall for approximate-search latency in this run. LanceDB
uses the least disk in this float32 comparison.

The same benchmark suite also includes a 1,000,000-row exact-search scale check:

| Metric | LynseDB | LanceDB |
| --- | ---: | ---: |
| Batch ingest vectors/s | 49,954 | **85,057** |
| Disk after ingest MB | 694.32 | **547.69** |
| Vector search mean ms | **6.013** | 109.009 |
| Vector search recall@10 | **1.0000** | **1.0000** |
| Filtered search mean ms | **2.160** | 148.455 |
| Filtered recall@10 | **1.0000** | **1.0000** |

At 1 million rows, LanceDB ingests faster and uses less persisted space, while
LynseDB records about 18x lower mean exact-vector latency and 69x lower mean
filtered-search latency. These are results from one reproducible machine and
dataset, not universal performance guarantees.

## When LynseDB Fits Better

- You want to start locally and keep the same API when moving to an HTTP service
  or a small self-hosted cluster.
- You need a real retrieval database, not just a vector index: documents,
  metadata, dense vectors, sparse vectors, BM25, hybrid search, and named vector
  fields can live behind one collection API.
- You care about exact or near-exact local vector search and very fast metadata
  filtered search.
- You want a Python-friendly client with a Rust-backed storage/search core,
  mmap storage, WAL, snapshots, restore, export/import, and explicit durability
  operations.
- You prefer self-hosted control without giving up an easy embedded developer
  workflow.

## When ChromaDB May Fit Better

- Your stack already depends on Chroma integrations and you do not need to move
  away from them.
- You want a managed Chroma Cloud path as the primary production deployment.
- You are following tutorials or frameworks that assume Chroma-specific
  collection semantics.
- You need Chroma ecosystem compatibility more than LynseDB's self-hosted,
  exact-recall, and hybrid-retrieval posture.

## When LanceDB May Fit Better

- Your workload is table-oriented and benefits from LanceDB's columnar storage
  model.
- You need very fast local batch ingest or the smallest disk footprint in this
  benchmark profile.
- You already use LanceDB Cloud or Lance/Arrow-style data workflows.
- You want a native local hybrid search path and can tune around table/index
  behavior for your workload instead of using LynseDB's collection-centered API.

## When USEARCH May Fit Better

- You only need a fast in-process vector index and your application already owns
  documents, metadata, filtering, durability policy, and serving.
- You are willing to trade recall for HNSW latency on approximate search.
- You want a compact library dependency rather than a database API, server, or
  retrieval stack.
- You do not need database-level metadata filters, BM25, sparse vectors, hybrid
  search, collection management, or operational APIs from LynseDB.

## API Mapping

| Chroma/Lance/USEARCH-style action | LynseDB equivalent |
| --- | --- |
| Create a persistent local client | `lynse.VectorDBClient("./data")` |
| Connect to a server | `lynse.VectorDBClient("http://host:7637")` |
| Create or open a collection/table | `db.require_collection("docs")` |
| Add documents | `collection.add(ids=..., documents=..., fields=...)` |
| Add embeddings / vectors | `collection.add(ids=..., vectors=..., fields=...)` |
| Query by text | `collection.search(document="...", k=...)` |
| Query by vector | `collection.search(vector, k=...)` |
| Filter metadata | `collection.search(..., where="field = 'value'")` |
| Commit writes | `collection.commit()` or `with collection:` for fast logical commits |
| Durable checkpoint | `collection.checkpoint()` before backups, snapshots, or controlled shutdowns |
| Tune index | `collection.build_index("HNSW-L2")`, `collection.build_index("IVF-L2", n_clusters=...)` |

## Migration Sketch

```python
import lynse


client = lynse.VectorDBClient("./lynsedb-data")
db = client.create_database("rag")
collection = db.require_collection("docs")

with collection:
    collection.add(
        ids=["doc-1", "doc-2"],
        documents=[
            "LynseDB can run embedded in one Python process.",
            "LynseDB can also run as an HTTP service.",
        ],
        fields=[
            {"source": "local"},
            {"source": "server"},
        ],
    )

result = collection.search(
    document="How do I share vector search across workers?",
    k=2,
    return_fields=True,
)

print(result.to_list())
```

For production RAG, pass vectors generated by an explicitly chosen embedding
model. The document-first path is useful for prototypes and local tools, but
embedding model choice should be part of your retrieval contract.

## Positioning

LynseDB should not be treated as a drop-in clone of ChromaDB, LanceDB, or
USEARCH. It is the more complete choice when you want fast embedded retrieval
and a production path in the same system:

> A Python-first, Rust-powered vector database that starts embedded and grows
> into a self-hosted service or lightweight cluster.

Choose LynseDB when local development speed, exact and filtered retrieval,
hybrid search, self-hosted control, and an incremental scale path matter
together.
