# Tutorial: Connect and Deploy

LynseDB has one public entry point, `lynse.VectorDBClient`. The client selects
local or remote mode from the `uri` value.

## Local mode

Local mode embeds the Rust backend directly in the Python process.

```python
import lynse

client = lynse.VectorDBClient(uri="./local-data")
db = client.create_database("app", drop_if_exists=False)
collection = db.require_collection("items", dim=768)
```

Local mode is supported on Linux and macOS. Native Windows environments are not
supported; use WSL 2 (Windows Subsystem for Linux) for local mode on Windows, or
run the HTTP server with Docker.

Use local mode for:

- notebooks and experiments;
- single-process applications;
- unit tests;
- edge jobs where one process owns the data directory.

Do not share the same local data path between independent writer processes. Use
the HTTP server when more than one process needs access.

## Read-only local mode

Open existing storage without writes:

```python
client = lynse.VectorDBClient(uri="./local-data", read_only=True)
db = client.get_database("app")
collection = db.get_collection("items")
```

Read-only handles are useful for offline inspection and safety checks.

## Remote mode

Start a server:

```shell
lynse serve --host 0.0.0.0 --port 7637 --data-dir ./server-data
```

Connect from Python:

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637")
db = client.create_database("app")
collection = db.require_collection("items", dim=768)
```

Remote mode is the recommended deployment mode for:

- web APIs with multiple workers;
- background jobs and online services sharing the same data;
- containerized deployments;
- environments where request logging, metrics, health checks, and auth matter.

## API key authentication

Start the server with a key:

```shell
lynse serve \
  --host 0.0.0.0 \
  --port 7637 \
  --data-dir ./server-data \
  --api-key your_key
```

Connect with the same key:

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="your_key")
```

The Python client sends `Authorization: Bearer <api_key>`. The server also
accepts HTTP Basic auth where the password is the API key.

Public endpoints are `/`, `/healthz`, and `/readyz`. Operational and data
endpoints require authentication when `--api-key` is configured.

## Docker

```shell
docker run \
  -p 7637:7637 \
  -v lynsedb-data:/data \
  birchkwok/lynsedb:latest
```

With authentication:

```shell
docker run \
  -p 7637:7637 \
  -e LYNSE_API_KEY=your_key \
  -v lynsedb-data:/data \
  birchkwok/lynsedb:latest
```

Docker is the recommended Windows deployment path when you do not want to run
inside WSL 2.

Deployment examples are included in:

- `docs/deployment/docker-compose.yml`
- `docs/deployment/lynsedb.service`
- `docs/deployment/kubernetes.yaml`

## Server tuning

Common production guards:

```shell
lynse serve \
  --host 0.0.0.0 \
  --port 7637 \
  --data-dir ./server-data \
  --json-limit-mb 256 \
  --payload-limit-mb 512 \
  --request-timeout-secs 300 \
  --keep-alive-secs 75 \
  --max-top-k 1000 \
  --max-batch-vectors 50000 \
  --max-collection-vectors 10000000 \
  --audit-log
```

Slow query warnings default to 1000 ms:

```shell
LYNSE_SLOW_QUERY_WARN_MS=250 lynse serve --data-dir ./server-data
```

## Health, readiness, metrics, and OpenAPI

```shell
curl http://127.0.0.1:7637/healthz
curl http://127.0.0.1:7637/readyz
curl http://127.0.0.1:7637/metrics
curl http://127.0.0.1:7637/openapi.json
```

`/metrics` exposes Prometheus text for request counts, latency, WAL bytes, data
directory size, vector index bytes, process memory, and index build progress.

Server logs are structured JSON and include `request_id`. Mutating requests emit
`audit` events unless audit logging is disabled.

## Connection lifecycle

Use context managers or call `close()` explicitly:

```python
with lynse.VectorDBClient("http://127.0.0.1:7637") as client:
    db = client.get_database("app")
    collection = db.get_collection("items")
    print(collection.shape)
```

For long-lived services, create one client per process and reuse it.
