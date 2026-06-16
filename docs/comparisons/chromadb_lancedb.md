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

## Benchmark Snapshot

The local 100,000-row benchmark in
[`vector_database_benchmarks.md`](vector_database_benchmarks.md) uses
128-dimensional normalized vectors, 100 queries, and batch insert APIs for the
embedded engines. ChromaDB, LanceDB, and LynseDB numbers below come from the
same 2026-06-15 100k run. USEARCH is shown from the 2026-06-16 float32
follow-up run, where it was configured as a vector-only HNSW index.

| Metric | LynseDB | ChromaDB | LanceDB | USEARCH |
| --- | ---: | ---: | ---: | ---: |
| Batch ingest vectors/s | 53,544 | 2,373 | 85,236 | 10,351 |
| Disk after ingest MB | 87.47 | 162.41 | 55.74 | 63.03 |
| Vector search mean ms | 0.614 | 0.985 | 12.186 | 0.571 |
| Vector search recall@10 | 1.0000 | 0.5210 | 1.0000 | 0.6020 |
| Filtered search mean ms | 0.114 | 36.194 | 18.082 | n/a |
| Filtered recall@10 | 1.0000 | 0.9970 | 1.0000 | n/a |
| Startup mean ms | 1.892 | 11.511 | 2.092 | 2.068 |

The most important result is that LynseDB combines database features with
low-latency exact retrieval. In this benchmark profile it leads filtered search
latency by a wide margin, starts quickly, preserves exact or near-exact recall,
and still offers document search, metadata filters, hybrid retrieval, an HTTP
service, and a self-hosted cluster path. USEARCH posts slightly faster raw
vector-search latency in the float32 follow-up, but with lower recall and no
metadata-filtered or hybrid retrieval path in this suite. LanceDB leads the
original 100k disk footprint and batch ingest, and ChromaDB remains convenient
for Chroma-specific ecosystems, but LynseDB is the stronger default when the
application needs both speed and a complete retrieval/database layer.

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
