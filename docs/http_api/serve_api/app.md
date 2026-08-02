# HTTP Server Overview

The HTTP server exposes the same database and collection operations used by the
Python remote client.

## Start the server

```shell
lynse serve --host 0.0.0.0 --port 7637 --data-dir ./server-data
```

With authentication:

```shell
lynse serve --host 0.0.0.0 --port 7637 --data-dir ./server-data --api-key your_key
```

## Response envelope

Successful JSON responses use this shape:

```json
{
  "status": "success",
  "params": {}
}
```

JSON error responses include a human-readable `error` and a stable `code`:

```json
{
  "error": "Collection not found: examples",
  "code": "not_found"
}
```

The server uses HTTP `400` for invalid arguments, `401` for authentication
failures, `404` for missing databases or collections, `409` for conflicts such
as duplicate resources or an unavailable index/quantizer, and `500` for
internal failures. Binary endpoints use the same HTTP status classes with a
plain-text body.

## Authentication

When `--api-key` is configured, all endpoints except `/`, `/healthz`, and
`/readyz` require authentication.

Bearer token:

```shell
curl -H "Authorization: Bearer your_key" http://127.0.0.1:7637/list_databases
```

Basic auth is also accepted; the password is treated as the API key.

## Operational endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/` | public | server banner and status |
| `GET` | `/healthz` | public | liveness check |
| `GET` | `/readyz` | public | readiness check |
| `GET` | `/metrics` | protected | Prometheus metrics |
| `GET` | `/openapi.json` | protected | generated OpenAPI schema |

## Binary endpoints

Some high-throughput operations use compact binary payloads internally:

- `/search_binary`
- `/batch_search_binary`
- `/head_binary`
- `/tail_binary`

User blobs also use binary bodies: `POST /write_blob` stores arbitrary bytes
under a collection-local key, and `GET /read_blob` returns the whole value or a
byte range. The Python client exposes these as `write_blob`, `read_blob`,
`read_blob_range`, and `delete_blob` in both local and HTTP modes.

The Python HTTP client handles these protocols. Prefer the Python client unless
you are implementing another language client.
