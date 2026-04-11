# Quick Start

```python linenums="1"
import lynse

print("LynseDB version is: ", lynse.__version__)
```

    LynseDB version is:  0.3.0


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
lynse run --host localhost --port 7637
# enable auth if needed:
# lynse run --host localhost --port 7637 --api-key your_key
# or via Docker:
docker run -p 7637:7637 birchkwok/lynsedb:latest
# docker run -p 7637:7637 -e LYNSE_API_KEY=your_key birchkwok/lynsedb:latest
```

```python linenums="1"
# Connect with optional API key
client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="your_key")
my_db = client.create_database("test_db", drop_if_exists=True)
```

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
