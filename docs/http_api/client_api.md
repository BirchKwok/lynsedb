# Python HTTP Client API

Most users should connect through `lynse.VectorDBClient("http://host:7637")`.
It creates the HTTP client internally and keeps local/remote behavior aligned.

```python
import lynse

client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="optional_key")
db = client.create_database("app")
collection = db.require_collection("items", dim=768)
```

You can instantiate `HTTPClient` directly when you already know the database
name:

```python
from lynse.api.http_api.client_api import HTTPClient

db = HTTPClient("http://127.0.0.1:7637", "app", api_key="optional_key")
```

## HTTPClient

```python
HTTPClient(uri, database_name, api_key=None)
```

| Parameter | Description |
| --- | --- |
| `uri` | Server base URL, for example `http://127.0.0.1:7637`. |
| `database_name` | Database this client operates on. |
| `api_key` | Optional Bearer token. |

The methods mirror the database client described in [Python Client API](../client.md):

- `require_collection`
- `get_collection`
- `drop_collection`
- `show_collections`
- `show_collections_details`
- `snapshot_collection`
- `restore_collection`
- `export_collection`
- `import_collection`
- `snapshot_database`
- `restore_database`
- `drop_database`
- `database_exists`

## HTTP Collection

The remote `Collection` class mirrors the local collection API:

- writes: `add`, `upsert`, `commit`, `flush`,
  `checkpoint`, `close`;
- indexes: `build_index`, `remove_index`;
- named/sparse vectors: `create_vector_field`, `list_vector_fields`,
  `add_named_vectors`, `add_sparse_vectors`;
- search: `search`, `batch_search`, `search_range`, `search_profile`,
  `search_sparse`, `bm25_search`, `hybrid_search`;
- query/data access: `query`, `query_vectors`, `head`, `tail`, `read_by_only_id`,
  `list_fields`, `is_id_exists`, `max_id`, `shape`, `stats`, `index_mode`;
- maintenance: `delete`, `restore`, `list_deleted_ids`, `compact`,
  `snapshot_to`, `export_to`, `update_description`.

The HTTP Python client uses the same explicit method signatures and parameter
ignore rules as the local client. For example, `build_index(...,
n_clusters=...)` uses `n_clusters` only for IVF and SPANN indexes, and `search(...,
nprobe=..., approx=..., eps=...)` ignores parameters that do not apply to the
active index or metric.

## Error behavior

The HTTP client raises `ExecutionError` when the server returns a non-200 JSON
error response. Connection failures and authentication failures during
`VectorDBClient(...)` initialization raise `ConnectionError`.

## Binary protocol

The HTTP client uses compact binary endpoints for high-throughput operations
such as binary bulk add, search, batch search, head, and tail. This is an
implementation detail of the Python client; application code should use the
normal methods above.
