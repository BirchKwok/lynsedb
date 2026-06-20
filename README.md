<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png">
    <img alt="LynseDB logo" src="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png" height="100">
  </picture>
</div>
<br>

# LynseDB

**LynseDB is a Python-first vector database with a Rust storage and search
engine, built for AI applications that need to move from a local prototype to a
service, and then to a lightweight sharded cluster, without changing the client
API.**

It is a good fit for RAG, semantic search, agent memory, multimodal retrieval,
document QA, recommendation features, and internal AI tools where you want
strong retrieval primitives without operating a heavyweight database stack on
day one.

LynseDB also searches more than embeddings: native distance paths cover
geospatial coordinates, packed binary fingerprints, aligned numeric profiles,
and probability distributions through the same collection API.

```python
import lynse

client = lynse.VectorDBClient("./ai-memory")              # embedded
client = lynse.VectorDBClient("http://127.0.0.1:7637")    # server or cluster
```

## Why LynseDB

- **Start local, grow in place**: use the same Python client for embedded
  storage, a single HTTP server, or a coordinator-backed cluster.
- **Rust where it matters**: vector storage, search, indexes, filters, WAL,
  snapshots, and server execution are backed by the Rust core.
- **AI-native retrieval**: dense vectors, metadata filters, BM25, hybrid search,
  sparse vectors, named vector fields, external reranking, and result views are
  exposed from one collection API.
- **Domain-aware similarity**: use Haversine for coordinates, packed
  Tanimoto/Dice/Hamming for fingerprints, correlation for aligned profiles,
  Hellinger/Jensen–Shannon/Wasserstein-1D for distributions, and
  Chebyshev/Canberra/Bray–Curtis for feature and abundance data.
- **Small operational footprint**: a single Python process can own the data
  directory during development; production can run as an HTTP service with API
  keys, health checks, readiness checks, metrics, OpenAPI, Docker, systemd, and
  Kubernetes examples.
- **Cluster mode when one node is not enough**: shard groups, stable hash
  bucket routing, coordinator fan-out search, replica write mirroring, health
  checks, primary promotion, and `/cluster_info` diagnostics are included.
- **Cost-conscious by design**: use local mode for notebooks, scripts, tests,
  jobs, and single-service apps; add network and cluster overhead only when
  multiple workers, larger datasets, or failover require it.

## Install

Python 3.9 or newer is required.

```shell
pip install lynsedb
```

For document-first inserts and `search(document=...)`, install the optional
local embedding adapter explicitly:

```shell
pip install "lynsedb[embeddings]"
```

Native Linux and macOS environments are supported. Native Windows environments
are not supported; on Windows, run LynseDB inside WSL 2 or use Docker.

## Quickstart: Build a Tiny AI Knowledge Base

This example stores small knowledge snippets with documents and metadata, then
retrieves context for a user question. LynseDB can embed the documents lazily
through the default local FastEmbed adapter, build a `FLAT-IP` index on first
write, and commit automatically when the collection context exits successfully.
`commit()` is a fast logical write boundary; call `checkpoint()` before backups,
snapshots, controlled shutdowns, or critical durability acknowledgements.

```python
import lynse


docs = [
    {
        "id": "local-mode",
        "title": "Local mode",
        "text": "Use embedded mode for notebooks, tests, jobs, and single-process AI apps.",
        "tags": ["local", "python"],
    },
    {
        "id": "server-mode",
        "title": "Server mode",
        "text": "Run lynse serve when several services or workers need shared vector search.",
        "tags": ["server", "production"],
    },
    {
        "id": "cluster-mode",
        "title": "Cluster mode",
        "text": "Use a coordinator with shard groups when one node is not enough for data or throughput.",
        "tags": ["cluster", "scale"],
    },
]

client = lynse.VectorDBClient("./lynsedb-ai-demo")
db = client.create_database("assistant", drop_if_exists=True)
collection = db.require_collection("knowledge", drop_if_exists=True)

with collection:
    collection.add(
        ids=[doc["id"] for doc in docs],
        documents=[f"{doc['title']} {doc['text']}" for doc in docs],
        fields=docs,
    )

question = "How should I deploy vector search for multiple workers?"
result = collection.search(
    document=question,
    k=1,
    where="tags CONTAINS 'server'",
    return_fields=True,
)

for item in result.to_list():
    print(item["id"], item["title"], item["text"])
```

You now have the core loop behind most AI retrieval systems:

1. Chunk or collect content.
2. Embed it.
3. Store vectors with stable IDs and metadata.
4. Search by semantic similarity plus filters.
5. Send the returned fields to your LLM as grounded context.

For production systems, explicitly choose and version your embedding model. Pass
vectors directly through `vectors=` when you already use OpenAI embeddings,
sentence-transformers, FastEmbed, CLIP, or your own model.

## One API, Three Deployment Shapes

### 1. Embedded Mode

Use embedded mode when one Python process owns the data directory. It avoids a
network hop and is the fastest way to add vector search to a notebook, local
agent, ingestion job, test suite, or single-process app.

```python
import lynse

client = lynse.VectorDBClient("./data")
```

Do not share the same embedded data path between independent processes. When
multiple processes need the same database, run the HTTP server.

### 2. HTTP Service Mode

Use service mode when web workers, background jobs, or multiple applications
need shared access to one LynseDB instance.

```shell
lynse serve --host 0.0.0.0 --port 7637 --data-dir ./server-data
```

```python
import lynse

client = lynse.VectorDBClient("http://127.0.0.1:7637")
```

With API key authentication:

```shell
lynse serve \
  --host 0.0.0.0 \
  --port 7637 \
  --data-dir ./server-data \
  --api-key your_key
```

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="your_key")
```

Useful service endpoints:

```shell
curl http://127.0.0.1:7637/healthz
curl http://127.0.0.1:7637/readyz
curl http://127.0.0.1:7637/metrics
curl http://127.0.0.1:7637/openapi.json
```

### 3. Cluster Mode

Use cluster mode when a single server is no longer enough for data size, query
throughput, or shard-level failover. Applications still connect to one endpoint:

```python
client = lynse.VectorDBClient("http://coordinator:7637")
```

The coordinator owns metadata and request routing. Shard nodes are ordinary
LynseDB HTTP servers, each with its own data directory.

```text
Python / API clients
        |
        v
Coordinator :7637
        |
        +-- shard group sg0
        |     +-- primary http://10.0.0.11:7638
        |     +-- replica http://10.0.0.12:7638
        |
        +-- shard group sg1
              +-- primary http://10.0.0.21:7638
              +-- replica http://10.0.0.22:7638
```

Cluster workflow:

1. Start normal LynseDB servers as shard primaries and replicas.
2. Create a `cluster.json` that lists shard groups and replica layout.
3. Start a coordinator with `--role coordinator`.
4. Point application clients at the coordinator.
5. Monitor `/cluster_info` for shard health, replica state, and promotions.

Cluster mode does not require shared storage. Coordinator metadata is stored on
metadata owner shard(s) over internal RPC, and `--cluster-state` is only a local
cache path for the coordinator process. Make sure each coordinator can reach the
metadata owner shard RPC ports. By default, clusters with three or more
shard primaries use the first three primaries as replicated metadata owners;
smaller clusters use the first primary. Pass `--metadata-owners` only when you
want to pin the owner set explicitly.

```shell
lynse serve --host 127.0.0.1 --port 7638 --data-dir ./data/sg0-primary
lynse serve --host 127.0.0.1 --port 7639 --data-dir ./data/sg0-replica
lynse serve --host 127.0.0.1 --port 7640 --data-dir ./data/sg1-primary
lynse serve --host 127.0.0.1 --port 7641 --data-dir ./data/sg1-replica
```

```shell
lynse serve \
  --role coordinator \
  --host 127.0.0.1 \
  --port 7637 \
  --cluster-config ./cluster.json \
  --cluster-state ./cluster_state.cache.json
```

Cluster advantages:

- **Horizontal data growth**: stable hash buckets distribute collection records
  across shard groups.
- **Parallel retrieval**: searches fan out to shard groups and are merged by the
  coordinator into one top-k result set.
- **Replica-aware writes**: active replicas can receive mirrored writes when
  `write_mirror_replicas` is enabled.
- **Failover foundation**: the coordinator health-checks primaries and replicas;
  a healthy active replica can be promoted if a primary fails.
- **No client rewrite**: application code keeps using `VectorDBClient` and the
  normal database, collection, add, upsert, delete, search, and query APIs.
- **Clear operations model**: authoritative coordinator metadata lives on the
  metadata owner shard(s), while each coordinator keeps only a local
  `cluster_state.cache.json` cache.

Read the full [cluster deployment guide](docs/deployment/cluster-deployment.md)
before using cluster mode in production.

## Retrieval Features

- Dense vector search with flat, HNSW, IVF, SPANN, DiskANN, and quantized index
  families.
- SQL-style metadata filtering through `where` expressions.
- BM25 search over metadata fields for exact and lexical recall.
- Hybrid search with reciprocal-rank fusion or weighted dense/text candidates.
- Named vector fields for multimodal records, such as text and image embeddings
  on the same item.
- Sparse vector search for feature-weight retrieval.
- External rerank hooks for cross-encoders, LLM rerankers, or business rules.
- `ResultView` objects with NumPy arrays plus list, JSON, and dataframe
  conversion helpers.

## Search More Than Embeddings

Vector search is useful anywhere records can be compared numerically, not only
after an embedding model:

| Data | Native metrics | Example workloads |
| --- | --- | --- |
| Embeddings | inner product, squared L2, cosine | RAG, semantic and multimodal retrieval |
| Numeric features | Manhattan/L1, Chebyshev, Canberra | anomaly matching, tolerances, sensor and tabular features |
| Coordinates | Haversine in meters | nearby POI, fleet and device search |
| Binary fingerprints | Hamming, Jaccard/Tanimoto, Sørensen-Dice | molecular fingerprints, deduplication, genomic sketches |
| Aligned profiles | Pearson correlation distance | sensor curves, behavior profiles, gene expression |
| Distributions and abundance | Hellinger, Jensen–Shannon, Wasserstein-1D, Bray–Curtis | model drift, topics, forecasts, histograms and ecology |

The exact Flat path supports every metric above. HNSW supports L1, Haversine,
correlation, Hellinger, Wasserstein-1D, Jensen–Shannon, and Chebyshev. Binary
Flat search lazily packs each dimension to one bit, reducing the hot scan
representation by 32x versus `float32` for word-aligned dimensions without
replacing the durable source vectors.

See [Domain-aware distance metrics](docs/tutorials/distance_metrics.md) for
input contracts and the index compatibility matrix.

## Indexing

New collections build a `FLAT-IP` index automatically after the first primary
vector write. Disable this with `default_index=None`, or choose another default
when creating the collection:

```python
collection = db.require_collection("docs", dim=384, default_index="FLAT-COS")
```

Call `build_index()` when you want to rebuild or switch index modes. Move to
HNSW, IVF, or SPANN when latency matters, DiskANN when memory pressure matters,
and quantized variants when you want a smaller memory or disk footprint:

```python
collection.build_index("HNSW-L2")
collection.build_index("IVF-L2", n_clusters=256)
collection.build_index("SPANN-L2", n_clusters=256)
collection.build_index("DiskANN-L2")
collection.build_index("FLAT-IP-SQ8")
collection.build_index("FLAT-L2-PQ")
```

See the [indexing guide](docs/tutorials/indexing.md) for metric names, `nprobe`
tuning, binary indexes, and quantized index variants.

## Docker

```shell
docker run -p 7637:7637 -v lynsedb-data:/data birchkwok/lynsedb:latest
docker run -p 7637:7637 -e LYNSE_API_KEY=your_key -v lynsedb-data:/data birchkwok/lynsedb:latest
```

On Windows, use this Docker image or install/run LynseDB from a Linux
environment in WSL 2.

Deployment examples:

- [Docker Compose](docs/deployment/docker-compose.yml)
- [systemd](docs/deployment/lynsedb.service)
- [Kubernetes](docs/deployment/kubernetes.yaml)
- [Cluster deployment](docs/deployment/cluster-deployment.md)

## Documentation

- [Learning path](docs/tutorials/learning_path.md)
- [Core concepts](docs/tutorials/core_concepts.md)
- [Quickstart](docs/quickstart.md)
- [Connect and deploy](docs/tutorials/connect_and_deploy.md)
- [Databases and collections](docs/tutorials/databases_collections.md)
- [Add vectors](docs/tutorials/add_vectors.md)
- [Search and filter](docs/tutorials/search_and_filter.md)
- [Domain-aware distance metrics](docs/tutorials/distance_metrics.md)
- [Metadata filter cookbook](docs/tutorials/metadata_filter_cookbook.md)
- [Indexing guide](docs/tutorials/indexing.md)
- [Named, sparse, and hybrid search](docs/tutorials/named_sparse_hybrid.md)
- [Build a RAG workflow](docs/tutorials/rag_workflow.md)
- [Performance tuning](docs/tutorials/performance_tuning.md)
- [Backup and maintenance](docs/tutorials/operations.md)
- [Troubleshooting](docs/tutorials/troubleshooting.md)
- [Client API](docs/client.md)
- [ResultView](docs/result_view.md)

## Stability Notes

LynseDB is evolving quickly. Pin package and server image versions for
deployments, test migrations before upgrading, and back up data directories plus
cluster state before operational changes. For concurrent production access,
prefer the HTTP server or coordinator cluster over sharing one local data
directory across independent Python processes.
