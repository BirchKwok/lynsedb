# Tutorial: Core Concepts

LynseDB is a vector database with a Python-first API and a Rust backend. The
main workflow is:

1. connect with `VectorDBClient`;
2. create or open a database;
3. create or open a collection;
4. insert vectors, IDs, and metadata fields;
5. build an index when needed;
6. search, filter, query, update, delete, and maintain the collection.

## Client

`lynse.VectorDBClient` is the entry point.

```python
import lynse

local_client = lynse.VectorDBClient(uri="./data")
remote_client = lynse.VectorDBClient("http://127.0.0.1:7637")
```

The `uri` decides the mode:

| `uri` value | Mode | Meaning |
| --- | --- | --- |
| `None` | Local | Use the default root path from LynseDB config. |
| filesystem path | Local | Use the Rust backend directly in this Python process. |
| `http://...` or `https://...` | Remote | Use the HTTP server. |

Use local mode when one process owns the data directory. Use remote mode when
more than one process, worker, or service needs shared access.

## Database

A database is a named group of collections:

```python
db = local_client.create_database("app")
same_db = local_client.get_database("app")

print(local_client.list_databases())
```

Use separate databases for separate applications, tenants, or environments when
you want independent lifecycle operations such as drop, snapshot, or restore.

## Collection

A collection is the unit of vector storage and search:

```python
collection = db.require_collection("docs", dim=768)
```

The primary collection dimension is fixed. Every primary dense vector inserted
into this collection must have `dim` values.

Use separate collections when:

- vector dimensions differ;
- index or metric choices differ;
- data has a different lifecycle;
- permission or tenant boundaries should be physically separate.

Use metadata fields when the records belong together but need filtering.

## Row

Each row has:

| Part | Required | Notes |
| --- | --- | --- |
| ID | yes | Public string or non-negative integer ID, unique inside the collection. |
| primary vector | yes | Dense `float32` vector with collection dimension. |
| metadata field | no | JSON-like dict used for filters, BM25 search, and display. |
| named vectors | no | Extra dense vectors attached to the same ID. |
| sparse vector | no | Feature-ID weights attached to the same ID. |

Example:

```python
collection.add(
    ids="doc-1001",
    vectors=[0.1, 0.2, 0.3, 0.4],
    fields={
        "title": "vector database intro",
        "lang": "en",
        "tenant": "acme",
        "published": True,
        "tags": ["vector", "python"],
        "created_at": "2026-06-05",
    },
)
```

## IDs

IDs passed to `add()` are public external IDs owned by your application. LynseDB
keeps those IDs stable and maps them to internal monotonic integer IDs allocated
by the Rust backend.

Good ID practice:

- use strings or non-negative integers;
- keep IDs unique within one collection;
- use strings for natural IDs such as `"doc-123#chunk-4"`;
- store source document IDs, chunk numbers, and display payloads in metadata
  when they are useful for filtering or rendering;
- do not depend on internal IDs for application logic.

## Metadata fields

Fields are JSON-like dictionaries:

```python
field = {
    "title": "LynseDB guide",
    "score": 0.92,
    "active": True,
    "tags": ["docs", "retrieval"],
    "source": {"name": "manual", "page": 3},
}
```

Use fields for:

- result display;
- filters through `where=...`;
- BM25 search;
- reranker payloads;
- application bookkeeping.

Keep field types stable. For example, do not store `"rank": "10"` in some rows
and `"rank": 10` in others.

## Vector metrics

The metric describes how similarity is measured:

| Metric | Common index suffix | Meaning | Result ordering |
| --- | --- | --- | --- |
| Inner product | `FLAT`, `HNSW`, `IVF`, `DiskANN` | Larger score is better. | descending score |
| Squared L2 | `-L2` | Smaller distance is better. | ascending distance |
| Cosine | `-COS` or `-Cos` | Larger similarity is better. | descending score |
| Hamming | `-HAMMING-BINARY` | Smaller binary distance is better. | ascending distance |
| Jaccard | `-JACCARD-BINARY` | Smaller set distance is better. | ascending distance |

Choose the metric that matches your embedding model. Many modern embedding
models are evaluated with cosine similarity or inner product after
normalization.

## Indexes

An index controls how search scans candidates:

```python
collection.build_index("FLAT-L2")
collection.build_index("HNSW-L2")
collection.build_index("IVF-L2", n_clusters=256)
```

Flat indexes are simplest and make good correctness baselines. ANN indexes such
as HNSW and IVF trade exactness for latency. Quantized indexes trade some
quality or extra reranking work for lower memory or disk use.

## ResultView

Search and query methods return `ResultView`:

```python
result = collection.search([0.1, 0.2, 0.3, 0.4], k=3, return_fields=True)

print(result.ids)
print(result.distances)
print(result.fields)
print(result.to_list())
```

Use attributes for program logic and `to_list()` for row-shaped display.

## Commits and durability

`add()` is the simple write-through path. For grouped ingestion, prefer
`insert_session()`:

```python
with collection.insert_session() as session:
    session.add(
        ids="doc-1",
        vectors=[0.1, 0.2, 0.3, 0.4],
        fields={"title": "first row"},
    )
```

The session commits when the block succeeds. If the block raises an exception,
pending buffered writes from that session are discarded and the original
exception is preserved.

Use explicit lifecycle calls for services and operations:

```python
collection.commit()      # fast logical commit
collection.checkpoint()  # durable checkpoint
collection.flush()       # advanced: flush bytes without clearing WAL
collection.close()
client.close()
```

`commit()` is optimized for write latency. It makes the batch visible and clears
WAL state, but it does not promise that data has reached stable storage at the
instant the call returns. `checkpoint()` is the deterministic durability
boundary; call it before backups, snapshots, controlled shutdowns, or critical
write acknowledgements. `flush()` is mostly useful for storage-level workflows
that need bytes pushed out while keeping WAL state.

## Local and remote parity

The high-level Python API is intentionally similar in local and remote mode:

```python
client = lynse.VectorDBClient(uri="./data")
# or
client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="secret")

db = client.create_database("app")
collection = db.require_collection("docs", dim=4)
```

This makes it practical to prototype locally, then move to HTTP mode when the
application needs multiple processes or deployment controls.
