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


**LynseDB** is a lightweight vector database with a Python-first API and a Rust
storage/search backend. It can run embedded in one Python process or as an HTTP
server for multi-process and production deployments.

## Features

- Server-optional: local embedded mode and remote HTTP mode share the same high-level Python API.
- Dense vector search with flat, HNSW, IVF, DiskANN, and quantized index families.
- Metadata filtering with SQL-like `where` expressions.
- Named vector fields for multimodal records.
- Sparse, BM25 text, and hybrid retrieval.
- `ResultView` return objects with NumPy arrays and dataframe/JSON conversion helpers.
- HTTP health checks, metrics, OpenAPI schema, API key auth, snapshots, export/import, and compaction.

## Install

Python 3.9 or newer is required.

Native Linux and macOS environments are supported. Native Windows environments
are not supported; on Windows, run LynseDB inside WSL 2 (Windows Subsystem for
Linux) or use Docker.

```shell
pip install LynseDB
```

## Quick example

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./lynsedb-data")
db = client.create_database("demo", drop_if_exists=True)
collection = db.require_collection("documents", dim=4, drop_if_exists=True)

with collection.insert_session() as session:
    session.bulk_add_items(
        [
            ([0.10, 0.20, 0.30, 0.40], 1, {"title": "intro", "lang": "en"}),
            ([0.11, 0.19, 0.29, 0.39], 2, {"title": "guide", "lang": "en"}),
            ([0.80, 0.10, 0.20, 0.10], 3, {"title": "notes", "lang": "fr"}),
        ],
        enable_progress_bar=False,
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

## Local and remote modes

Local embedded mode:

```python
client = lynse.VectorDBClient(uri="./data")
```

Remote HTTP mode:

```shell
lynse serve --host 0.0.0.0 --port 7637 --data-dir ./server-data
```

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637")
```

With API key auth:

```shell
lynse serve --host 0.0.0.0 --port 7637 --data-dir ./server-data --api-key your_key
```

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="your_key")
```

Use local mode for notebooks, tests, and single-process apps. Use remote mode
when multiple processes need to share the same database.

## Docker

```shell
docker run -p 7637:7637 -v lynsedb-data:/data birchkwok/lynsedb:latest
docker run -p 7637:7637 -e LYNSE_API_KEY=your_key -v lynsedb-data:/data birchkwok/lynsedb:latest
```

On Windows, use this Docker image or install/run LynseDB from a Linux
environment in WSL 2.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Connect and deploy](docs/tutorials/connect_and_deploy.md)
- [Add vectors](docs/tutorials/add_vectors.md)
- [Search and filter](docs/tutorials/search_and_filter.md)
- [Indexing guide](docs/tutorials/indexing.md)
- [Named, sparse, and hybrid search](docs/tutorials/named_sparse_hybrid.md)
- [Backup and maintenance](docs/tutorials/operations.md)
- [Field filters](docs/FieldExpression.md)
- [ResultView](docs/result_view.md)

## Stability notes

LynseDB is still evolving. Pin package and server image versions for
deployments. For concurrent production access, prefer the HTTP server over
sharing one local data directory across independent Python processes.
