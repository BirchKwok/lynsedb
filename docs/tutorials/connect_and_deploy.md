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

### Default local path

If `uri=None`, LynseDB uses the configured default root path:

```python
client = lynse.VectorDBClient()
```

By default this comes from `LYNSE_DEFAULT_ROOT_PATH` in the generated
`~/.lynsedb_configs.ini` file. Passing an explicit path is usually clearer for
applications and tests:

```python
client = lynse.VectorDBClient(uri="./app-data")
```

### Local lifecycle

Use one local client per process for a given root path:

```python
client = lynse.VectorDBClient(uri="./app-data")
db = client.create_database("app")
collection = db.require_collection("items", dim=768)

# ... use collection ...

collection.close()
client.close()
```

For command-line scripts, the context manager is convenient:

```python
with lynse.VectorDBClient(uri="./app-data") as client:
    db = client.get_database("app")
    collection = db.get_collection("items")
    print(collection.stats())
```

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

### Remote client lifecycle

Create one client per process and reuse it:

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637")
db = client.get_database("app")
collection = db.get_collection("items")
```

The remote client uses an HTTP connection pool. Avoid creating a new client for
every request in a web application.

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

Use a persistent volume for `/data`. Without a volume, data is tied to the
container filesystem and can disappear when the container is removed.

## Server configuration file

`lynse serve` accepts a JSON or INI config file. Environment variables override
the config file, and explicit CLI flags override both.

Example INI:

```ini
[server]
# Bind address for the HTTP server.
host = 0.0.0.0
# TCP port for the HTTP server.
port = 7637
# Root directory for database files.
data_dir = ./server-data
# Optional API key for Bearer or Basic auth.
api_key = your_key
# HTTP worker thread count; omit to let the server choose.
workers = 4
# HTTP keep-alive timeout in seconds.
keep_alive_secs = 75
# Client request timeout in seconds.
request_timeout_secs = 300
# Maximum JSON request body size in MB.
json_limit_mb = 256
# Maximum raw payload request body size in MB.
payload_limit_mb = 512
# Search/query latency threshold for slow-query warnings; 0 disables it.
slow_query_warn_ms = 1000
# Maximum top-k style result size accepted by the server; 0 disables it.
max_top_k = 10000
# Maximum vectors, IDs, or queries accepted in one request; 0 disables it.
max_batch_vectors = 100000
# Maximum primary vectors allowed per collection; 0 disables it.
max_collection_vectors = 10000000
# Maximum vector bytes allowed per collection; 0 disables it.
max_collection_vector_bytes = 1099511627776
# Whether to emit audit logs for server requests.
audit_log = true
```

Run with:

```shell
lynse serve --config ./server.ini
```

JSON config files are also supported:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 7637,
    "data_dir": "./server-data"
  }
}
```

## Environment variables

Common server environment variables:

| Variable | Meaning |
| --- | --- |
| `LYNSE_HOST` | Bind host. |
| `LYNSE_PORT` or `PORT` | Server port. |
| `LYNSE_DATA_DIR` or `LYNSE_ROOT` | Data directory. |
| `LYNSE_API_KEY` | API key for Bearer or Basic auth. |
| `LYNSE_SERVER_WORKERS` | HTTP worker threads. |
| `LYNSE_KEEP_ALIVE_SECS` | Keep-alive timeout. |
| `LYNSE_CLIENT_REQUEST_TIMEOUT_SECS` | Request timeout. |
| `LYNSE_JSON_LIMIT_MB` | Maximum JSON body size. |
| `LYNSE_PAYLOAD_LIMIT_MB` | Maximum raw payload size. |
| `LYNSE_SLOW_QUERY_WARN_MS` | Slow query warning threshold; `0` disables. |
| `LYNSE_MAX_TOP_K` | Maximum accepted `k`, `max_results`, `head`, or `tail` size; `0` disables. |
| `LYNSE_MAX_BATCH_VECTORS` | Maximum vector, ID, or query count per request; `0` disables. |
| `LYNSE_MAX_COLLECTION_VECTORS` | Maximum primary vectors per collection; `0` disables. |
| `LYNSE_MAX_COLLECTION_VECTOR_BYTES` | Maximum estimated dense vector bytes per collection; `0` disables. |
| `LYNSE_AUDIT_LOG` | Enable mutating-request audit logs. |

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

Use limits as guardrails for public or shared services:

- lower `--max-top-k` to prevent expensive accidental searches;
- lower `--max-batch-vectors` to protect server memory;
- set `--max-collection-vector-bytes` according to available disk and memory;
- keep `--request-timeout-secs` high enough for index builds and large imports;
- enable `--audit-log` when you need an operations trail for writes.

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

## Deployment decision table

| Situation | Recommended mode | Reason |
| --- | --- | --- |
| Notebook or experiment | Local | Fastest setup and no server process. |
| Unit tests | Local with a temp directory | Easy cleanup and isolated data. |
| One Python service with one worker | Local or remote | Local is simpler; remote is better if you want health checks and metrics. |
| Web API with multiple workers | Remote | Avoid sharing one local path across independent processes. |
| Multiple applications sharing data | Remote | One server owns the data directory. |
| Windows host | Docker or WSL 2 | Native Windows is not supported. |
| Production container | Remote | Use persistent volume, API key, health checks, and metrics. |

## Production checklist

- Pin the Python package and Docker image versions.
- Store data on a persistent disk or volume.
- Use the HTTP server for multi-process access.
- Enable API keys outside trusted local networks.
- Monitor `/healthz`, `/readyz`, and `/metrics`.
- Set request and payload limits for your workload.
- Take snapshots or exports before risky maintenance.
- Test restore in a separate database before relying on backups.
- Compare ANN recall against a flat baseline before changing index settings.
