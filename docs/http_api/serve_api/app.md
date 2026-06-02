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

Error responses include an `error` field:

```json
{
  "error": "message"
}
```

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

- `/bulk_add_binary`
- `/search_binary`
- `/batch_search_binary`
- `/head_binary`
- `/tail_binary`

The Python HTTP client handles these protocols. Prefer the Python client unless
you are implementing another language client.
