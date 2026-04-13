# Quick Start

```python linenums="1"
import lynse

print("LynseDB version is: ", lynse.__version__)
```

    LynseDB version is:  0.2.0


## Initialize Client

LynseDB supports two modes:

- **Local mode** — direct Rust backend, no server required (recommended for development and single-process use).
- **Remote mode** — HTTP API, suitable for multi-process and production deployments.

### Local mode

```python linenums="1"
import lynse

# No URI → local mode, data stored under LYNSE_DEFAULT_ROOT_PATH
client = lynse.VectorDBClient()

# Create (or open) a database
my_db = client.create_database("test_db", drop_if_exists=True)
```

### Remote / HTTP mode

Start the server first:

```shell
lynse serve --host localhost --port 7637 --data-dir ./data
# enable auth if needed:
# lynse serve --host localhost --port 7637 --data-dir ./data --api-key your_key
# optional runtime limits:
# lynse serve --host localhost --port 7637 --data-dir ./data \
#   --json-limit-mb 256 --payload-limit-mb 512 \
#   --request-timeout-secs 300 --keep-alive-secs 75
# slow query warnings default to 1000 ms; set to 0 to disable:
# LYNSE_SLOW_QUERY_WARN_MS=250 lynse serve --host localhost --port 7637 --data-dir ./data
# production guards can cap result size, request batch size, collection size,
# estimated dense vector bytes, and audit logging:
# lynse serve --max-top-k 1000 --max-batch-vectors 50000 --max-collection-vectors 10000000 \
#   --max-collection-vector-bytes 1099511627776 --audit-log
# or via Docker:
docker run -p 7637:7637 birchkwok/lynsedb:latest
# docker run -p 7637:7637 -e LYNSE_API_KEY=your_key birchkwok/lynsedb:latest
```

```python linenums="1"
# Connect with optional API key
client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="your_key")
my_db = client.create_database("test_db", drop_if_exists=True)
```

Operational endpoints are available from the HTTP server:

```shell
curl http://127.0.0.1:7637/healthz
curl http://127.0.0.1:7637/readyz
curl http://127.0.0.1:7637/metrics
curl http://127.0.0.1:7637/openapi.json
```

`/metrics` exposes Prometheus text for request counts and latency, WAL bytes, total
data directory bytes, vector index bytes, process resident memory, and live index
build counters/progress. HTTP server logs are structured JSON and include
`request_id`; slow search/query requests emit `slow_query` warnings. Mutating
server requests emit `audit` events unless `LYNSE_AUDIT_LOG=false`.

Deployment examples live in `docs/deployment/` for Docker Compose, systemd, and Kubernetes.

---

## Create a Collection

**`WARNING`** — Setting `drop_if_exists=True` permanently deletes existing data.
Use `get_collection` to open an existing collection safely.

```python linenums="1"
collection = my_db.require_collection(
    "test_collection",
    dim=4,
    drop_if_exists=True,
    description="demo collection",
)
```

### Show collections

If pandas is installed, `show_collections_details` returns a DataFrame; otherwise a list of dicts.

```python linenums="1"
my_db.show_collections_details()
```

    collection_name    dim  n_threads  description
    test_collection      4         10  demo collection

### Update description

```python linenums="1"
collection.update_description("Hello World")
```

---

## Add Vectors

Use `insert_session` to guarantee data is committed automatically on exit.
You can also call `collection.commit()` manually after `add_item` / `bulk_add_items`.

### Single item at a time

```python linenums="1"
with collection.insert_session() as session:
    session.add_item(vector=[0.01, 0.34, 0.74, 0.31], id=1, field={'field': 'test_1', 'order': 0})
    session.add_item(vector=[0.36, 0.43, 0.56, 0.12], id=2, field={'field': 'test_1', 'order': 1})
    session.add_item(vector=[0.03, 0.04, 0.10, 0.51], id=3, field={'field': 'test_2', 'order': 2})
    session.add_item(vector=[0.11, 0.44, 0.23, 0.24], id=4, field={'field': 'test_2', 'order': 3})
    session.add_item(vector=[0.91, 0.43, 0.44, 0.67], id=5, field={'field': 'test_2', 'order': 4})
    session.add_item(vector=[0.92, 0.12, 0.56, 0.19], id=6, field={'field': 'test_3', 'order': 5})
    session.add_item(vector=[0.18, 0.34, 0.56, 0.71], id=7, field={'field': 'test_1', 'order': 6})
    session.add_item(vector=[0.01, 0.33, 0.14, 0.31], id=8, field={'field': 'test_2', 'order': 7})
    session.add_item(vector=[0.71, 0.75, 0.91, 0.82], id=9, field={'field': 'test_3', 'order': 8})
    session.add_item(vector=[0.75, 0.44, 0.38, 0.75], id=10, field={'field': 'test_1', 'order': 9})
```

### Bulk add

```python linenums="1"
import numpy as np

items = [
    (np.random.rand(4).astype(np.float32), i, {'tag': f'item_{i}'})
    for i in range(11, 21)
]

with collection.insert_session() as session:
    ids = session.bulk_add_items(items)

print(ids)  # [11, 12, ..., 20]
```

### High-throughput binary bulk add (no fields)

```python linenums="1"
vecs = np.random.rand(1000, 4).astype(np.float32)
n_added = collection.bulk_add_binary(vecs)
collection.commit()
```

### Named vector fields

Use named vector fields when one record has multiple embeddings, such as
`default` for semantic text and `image` for CLIP image vectors.

```python linenums="1"
collection.create_vector_field("image", dim=3, metric="l2")

image_vectors = np.array(
    [
        [0.10, 0.20, 0.30],
        [0.12, 0.19, 0.28],
        [0.90, 0.20, 0.10],
    ],
    dtype=np.float32,
)
collection.add_named_vectors("image", image_vectors, ids=[1, 2, 3])
collection.build_vector_field_index("image", "HNSW-L2")
collection.commit()

print(collection.list_vector_fields())
```

### Sparse vectors

Use sparse vectors for keyword, token, or feature-ID signals that should be scored with inner product.

```python linenums="1"
collection.add_sparse_vectors(
    vectors=[
        {10: 1.0, 42: 0.5},
        {11: 1.0, 42: 0.8},
        {12: 1.0, 90: 0.4},
    ],
    ids=[1, 2, 3],
)
collection.commit()
```

---

## Collection Info

```python linenums="1"
print(collection.shape)   # (n_vectors, dim)
print(collection.max_id)  # highest user ID stored
print(collection.stats())
# {'n_vectors': ..., 'n_live': ..., 'n_tombstoned': ...,
#  'dimension': 4, 'index_mode': 'FLAT', 'max_id': ...}
```

---

## Search

`search` returns a `ResultView` object. Unpack with `ids, distances, fields = result` or
use `.ids`, `.distances`, `.fields` directly.

### Basic search (Inner Product)

```python linenums="1"
result = collection.search(vector=[0.36, 0.43, 0.56, 0.12], k=3)
print(result.ids)        # array of top-k IDs
print(result.distances)  # array of distances/scores
```

### Search a named vector field

```python linenums="1"
result = collection.search(
    vector=[0.11, 0.20, 0.29],
    k=2,
    vector_field="image",
    return_fields=True,
)
print(result.ids, result.fields)
```

### Sparse search

```python linenums="1"
result = collection.search_sparse(
    {42: 1.0},
    k=5,
    return_fields=True,
)
print(result.ids, result.distances, result.fields)
```

### Search with field filtering (SQL WHERE)

```python linenums="1"
result = collection.search(
    vector=[0.36, 0.43, 0.56, 0.12],
    k=10,
    where="\"field\" = 'test_1' AND \"order\" <= 8",
    return_fields=True,
)
print(result.ids)
print(result.fields)
```

### Batch search

```python linenums="1"
queries = np.random.rand(5, 4).astype(np.float32)
results = collection.batch_search(queries, k=3)
for r in results:
    print(r.ids, r.distances)
```

### Range search

Return all vectors within a distance threshold (L2: ≤ threshold; IP/Cos: ≥ threshold).

```python linenums="1"
result = collection.search_range(
    vector=[0.36, 0.43, 0.56, 0.12],
    threshold=0.5,
    max_results=100,
)
print(result.ids, result.distances)
```

### Search profile

Use `search_profile` to inspect filter cardinality, estimated scanned vectors, index path, and timings.

```python linenums="1"
profile = collection.search_profile(
    vector=[0.36, 0.43, 0.56, 0.12],
    k=5,
    where="\"field\" = 'test_1'",
)
print(profile["items"]["ids"])
print(profile["profile"])
```

### Text search over metadata

`text_search` runs BM25 over a persistent inverted index of stored metadata fields. `text_fields` can limit matching to specific columns.

```python linenums="1"
result = collection.text_search(
    "vector database",
    k=5,
    text_fields=["title", "body"],
    return_fields=True,
)
print(result.ids, result.distances)
```

### Hybrid search

`hybrid_search` combines vector search and BM25 metadata search with RRF or weighted fusion.

```python linenums="1"
result = collection.hybrid_search(
    vector=[0.36, 0.43, 0.56, 0.12],
    text="vector database",
    text_fields=["title", "body"],
    fusion="rrf",
    k=5,
    return_fields=True,
)
print(result.ids, result.distances)
```

### External rerank hook

Use `reranker` to inject a cross-encoder / LLM rerank stage on client side.
The callback receives `{"query": ..., "items": [...]}` and can return:

- ordered IDs: `[id1, id2, ...]`
- `(id, score)` pairs: `[(id1, 0.91), ...]`
- score array aligned with input items: `np.array([...])`
- score map: `{id1: 0.91, id2: 0.77}`

```python linenums="1"
def rerank(payload):
    query_text = payload["query"].get("text", "")
    # Replace with your model inference; higher score = better rank
    return [
        (item["id"], 1.0 if query_text in (item["field"] or {}).get("tag", "") else 0.0)
        for item in payload["items"]
    ]

result = collection.hybrid_search(
    vector=[0.36, 0.43, 0.56, 0.12],
    text="item_3",
    text_fields=["tag"],
    k=20,                # retrieve more candidates first
    reranker=rerank,
    rerank_k=5,          # keep top-5 after rerank
    return_fields=True,  # return reranked fields
)
print(result.ids, result.distances)
```

---

## Index Modes

Build an ANN index for faster search on large collections.

```python linenums="1"
# Flat brute-force variants
collection.build_index("FLAT")        # Inner Product (default)
collection.build_index("FLAT-L2")     # Squared L2
collection.build_index("FLAT-COS")    # Cosine similarity

# Graph-based ANN
collection.build_index("HNSW")        # HNSW + Inner Product
collection.build_index("HNSW-L2")
collection.build_index("HNSW-Cos")

# Disk-friendly graph ANN
collection.build_index("DiskANN")
collection.build_index("DiskANN-L2")

# Inverted-file ANN
collection.build_index("IVF", n_clusters=256)
collection.build_index("IVF-L2", n_clusters=256)

# Quantized variants (SQ8 / PQ / RaBitQ / PolarVec)
collection.build_index("FLAT-IP-SQ8")
collection.build_index("FLAT-L2-PQ")
collection.build_index("FLAT-L2-RABITQ")
collection.build_index("FLAT-IP-POLARVEC")

print(collection.index_mode)  # e.g. "HNSW"

# Remove an existing index (revert to brute-force)
collection.remove_index()
```

Search with `nprobe` to tune recall vs. speed for IVF / HNSW:

```python linenums="1"
result = collection.search(
    vector=[0.36, 0.43, 0.56, 0.12], k=5, nprobe=20
)
```

---

## List Data

```python linenums="1"
head_result = collection.head(5)
print(head_result.ids)      # first 5 IDs
print(head_result.vectors)  # shape (5, dim)

tail_result = collection.tail(5)
print(tail_result.ids)
```

---

## Query Fields

`query` returns a `ResultView`; iterate over `.fields` or use `.to_list()` / `.to_pandas()`.

```python linenums="1"
# Filter by SQL WHERE expression (column names must be double-quoted)
result = collection.query(where="\"field\" = 'test_1' AND \"order\" <= 6")
print(result.ids)
print(result.fields)

# Indexed metadata filters support numeric ranges, booleans, ISO date strings,
# keyword equality, and array membership.
result = collection.query(where="\"order\" >= 2 AND \"order\" < 8")
result = collection.query(where="\"active\" = true")
result = collection.query(where="\"tags\" CONTAINS 'rust'")
result = collection.query(where="\"created_at\" >= '2026-04-01'")

# Filter by specific IDs only
result = collection.query(filter_ids=[1, 2, 3], return_ids_only=True)
print(result.ids)
```

---

## Query Vectors

Retrieve vectors along with their fields using a WHERE filter or ID list.

```python linenums="1"
result = collection.query_vectors(where="\"field\" = 'test_1'")
print(result.ids)
print(result.vectors)  # shape (n, dim)
print(result.fields)

result2 = collection.query_vectors(filter_ids=[1, 2, 3])
print(result2.ids, result2.vectors)
```

---

## ResultView

All search, query, head, and tail operations return a `ResultView` object.

```python linenums="1"
result = collection.search([0.36, 0.43, 0.56, 0.12], k=3, return_fields=True)

# Attribute access
print(result.ids)           # numpy int64 array
print(result.distances)     # numpy float32 array
print(result.fields)        # list of dicts

# Tuple unpacking  (ids, distances, fields)
ids, distances, fields = result

# Conversion helpers
result.to_dict()
result.to_list()
result.to_json()
result.to_pandas()    # requires pandas
result.to_polars()    # requires polars
result.to_arrow()     # requires pyarrow

# Indexing and iteration
print(result[0])      # first result row as dict
for row in result:
    print(row)

print(len(result))    # number of results
```

---

## Soft Delete & Restore

Vectors can be logically deleted (tombstoned) without physically removing them.
Deleted vectors are excluded from all search results.

```python linenums="1"
collection.delete_items([3, 5])
print(collection.list_deleted_ids())   # [3, 5]

# Verify exclusion from search
result = collection.search([0.03, 0.04, 0.10, 0.51], k=10)
assert 3 not in result.ids

# Restore
collection.restore_items([3])
print(collection.list_deleted_ids())   # [5]
```

---

## Compaction

Physically remove all tombstoned vectors and rebuild storage.

```python linenums="1"
removed = collection.compact()
print(f"Physically removed {removed} vectors")
print(collection.list_deleted_ids())  # []
```

---

## Manage Collections and Databases

```python linenums="1"
# List collections
print(my_db.show_collections())

# Check existence
print(my_db.database_exists())

# Get an existing collection by name
coll = my_db.get_collection("test_collection")

# Drop a collection (irreversible)
my_db.drop_collection("test_collection")
print(my_db.show_collections())  # []

# Drop the whole database (irreversible)
my_db.drop_database()
```

---

## Utility Functions

```python linenums="1"
from lynse._backend import compute_distance, top_k_search
import numpy as np

a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

print(compute_distance(a, b, "IP"))     # inner product
print(compute_distance(a, b, "L2"))     # squared L2
print(compute_distance(a, b, "cosine")) # cosine similarity

candidates = np.random.rand(1000, 3).astype(np.float32)
ids, dists = top_k_search(a, candidates, metric="IP", k=5)
print(ids, dists)
```
