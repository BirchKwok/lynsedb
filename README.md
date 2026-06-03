<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png">
    <img alt="LynseDB logo" src="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png" height="100">
  </picture>
</div>
<br>

<p align="center">
  <a href="https://discord.com/invite/rcYK5nYF"><img src="https://img.shields.io/badge/Discord-Online-brightgreen" alt="Discord"></a>
  <a href="https://badge.fury.io/py/LynseDB"><img src="https://badge.fury.io/py/LynseDB.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/LynseDB/"><img src="https://img.shields.io/pypi/pyversions/LynseDB" alt="PyPI - Python Version"></a>
  <a href="https://github.com/BirchKwok/LynseDB/actions/workflows/python-tests.yml"><img src="https://github.com/BirchKwok/LynseDB/actions/workflows/python-tests.yml/badge.svg" alt="Python testing"></a>
  <a href="https://github.com/BirchKwok/LynseDB/actions/workflows/docker-tests.yml"><img src="https://github.com/BirchKwok/LynseDB/actions/workflows/docker-tests.yml/badge.svg" alt="Docker build"></a>
</p>

# LynseDB

**LynseDB** is a high-performance, low-cost vector database that can run either
embedded inside your Python process or as a standalone HTTP service. It keeps a
Python-first developer experience while moving storage and search execution into
a Rust backend.

Use LynseDB when you want vector search that starts as a local library, can be
deployed as a service when your app grows, and does not require a heavy database
cluster for common semantic search, RAG, agent memory, and multimodal retrieval
workloads.

## Why LynseDB

- **Embedded and service-ready**: the same `lynse.VectorDBClient` works with a
  local data path or a remote HTTP endpoint.
- **High-performance core**: dense vector search, metadata filtering, index
  building, and storage paths are backed by Rust.
- **Low operating cost**: run in-process for single-service apps and jobs, then
  switch to the HTTP server only when multiple workers or services need shared
  access.
- **Retrieval beyond dense vectors**: supports metadata filters, named vector
  fields, sparse vectors, BM25 text search, and hybrid search.
- **Deployment basics included**: API key auth, health checks, readiness checks,
  Prometheus metrics, OpenAPI schema, snapshots, restore, export/import, Docker,
  systemd, and Kubernetes examples.

## Best Fit

LynseDB is designed for teams that need a practical vector database with a small
footprint:

- local semantic search inside Python apps, scripts, notebooks, and tests;
- RAG and agent-memory storage where one process can own the data directory;
- web APIs and background workers that need a shared vector service;
- multimodal records with multiple embeddings per item;
- deployments where predictable cost and simple operations matter more than
  running a distributed vector database cluster.

For concurrent production access from independent processes, use the HTTP server
rather than sharing one embedded data directory.

## Install

Python 3.9 or newer is required.

Native Linux and macOS environments are supported. Native Windows environments
are not supported; on Windows, run LynseDB inside WSL 2 (Windows Subsystem for
Linux) or use Docker.

```shell
pip install LynseDB
```

## Quickstart

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./lynsedb-data")
db = client.create_database("demo", drop_if_exists=True)
collection = db.require_collection("documents", dim=4, drop_if_exists=True)

items = [
    ([0.10, 0.20, 0.30, 0.40], 1, {"title": "LynseDB intro", "lang": "en"}),
    ([0.11, 0.19, 0.29, 0.39], 2, {"title": "Vector guide", "lang": "en"}),
    ([0.80, 0.10, 0.20, 0.10], 3, {"title": "French note", "lang": "fr"}),
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

## One API, Two Deployment Modes

### Embedded mode

Use embedded mode when one Python process owns the data directory. It avoids a
network hop and is the lowest-cost way to add vector search to a local app,
notebook, test suite, job, or small service.

```python
client = lynse.VectorDBClient(uri="./data")
```

### Service mode

Use service mode when multiple processes, web workers, or applications need to
share the same database.

```shell
lynse serve --host 0.0.0.0 --port 7637 --data-dir ./server-data
```

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637")
```

With API key authentication:

```shell
lynse serve --host 0.0.0.0 --port 7637 --data-dir ./server-data --api-key your_key
```

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="your_key")
```

Health, readiness, metrics, and OpenAPI endpoints are available in service mode:

```shell
curl http://127.0.0.1:7637/healthz
curl http://127.0.0.1:7637/readyz
curl http://127.0.0.1:7637/metrics
curl http://127.0.0.1:7637/openapi.json
```

## Retrieval Features

- Dense vector search with flat, HNSW, IVF, DiskANN, and quantized index
  families.
- SQL-like metadata filtering through `where` expressions.
- Named vector fields for multimodal records, such as text and image embeddings
  on the same item.
- Sparse vector search for feature-weight retrieval.
- BM25 text search over metadata fields.
- Hybrid search with vector and text candidates.
- `ResultView` return objects with NumPy arrays plus list, JSON, and dataframe
  conversion helpers.

## Indexing

Start with a flat index as a correctness baseline:

```python
collection.build_index("FLAT-L2")
```

Move to HNSW or IVF when latency matters, DiskANN when memory pressure matters,
and quantized variants when you want a smaller memory or disk footprint:

```python
collection.build_index("HNSW-L2")
collection.build_index("IVF-L2", n_clusters=256)
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

Deployment examples are included in:

- [Docker Compose](docs/deployment/docker-compose.yml)
- [systemd](docs/deployment/lynsedb.service)
- [Kubernetes](docs/deployment/kubernetes.yaml)

## Documentation

- [Quickstart](docs/quickstart.md)
- [Connect and deploy](docs/tutorials/connect_and_deploy.md)
- [Add vectors](docs/tutorials/add_vectors.md)
- [Search and filter](docs/tutorials/search_and_filter.md)
- [Indexing guide](docs/tutorials/indexing.md)
- [Named, sparse, and hybrid search](docs/tutorials/named_sparse_hybrid.md)
- [Backup and maintenance](docs/tutorials/operations.md)
- [Client API](docs/client.md)
- [Field filters](docs/FieldExpression.md)
- [ResultView](docs/result_view.md)

## Stability Notes

LynseDB is still evolving. Pin package and server image versions for
deployments, and test migrations before upgrading. For concurrent production
access, prefer the HTTP server over sharing one local data directory across
independent Python processes.
