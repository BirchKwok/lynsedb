# Tutorial: Troubleshooting

This page maps common LynseDB symptoms to likely causes and fixes.

## Installation

### Native Windows does not work

Native Windows environments are not supported. Use one of:

- WSL 2 with Linux Python;
- Docker server mode;
- a Linux or macOS environment.

### Import fails

Check the Python version:

```shell
python --version
```

LynseDB requires Python 3.9 or newer.

Reinstall in the active environment:

```shell
python -m pip install -U LynseDB
```

## Connection

### Cannot connect to remote server

Start the server:

```shell
lynse serve --host 127.0.0.1 --port 7637 --data-dir ./server-data
```

Check the root endpoint:

```shell
curl http://127.0.0.1:7637/
```

Then connect:

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637")
```

Make sure the client URL includes `http://` or `https://`.

### Authentication failed

If the server was started with `--api-key`, pass the same key:

```python
client = lynse.VectorDBClient(
    "http://127.0.0.1:7637",
    api_key="your_key",
)
```

Raw HTTP requests need:

```shell
curl -H "Authorization: Bearer your_key" http://127.0.0.1:7637/list_databases
```

Public endpoints are `/`, `/healthz`, and `/readyz`. Other endpoints require
auth when an API key is configured.

### Local data path is shared by multiple processes

Do not let independent writer processes share the same local root path. Use
HTTP server mode so one process owns the data directory:

```shell
lynse serve --host 0.0.0.0 --port 7637 --data-dir ./server-data
```

## Database and collection setup

### Database does not exist

Use `create_database()` once, then `get_database()` later:

```python
db = client.create_database("app")
db = client.get_database("app")
```

Inspect names:

```python
print(client.list_databases())
```

### Collection does not exist

Create or open:

```python
collection = db.require_collection("docs", dim=768)
```

Open only if it exists:

```python
collection = db.get_collection("docs")
```

Inspect names:

```python
print(db.show_collections())
```

### Data disappeared after a test

Look for `drop_if_exists=True`:

```python
client.create_database("app", drop_if_exists=True)
db.require_collection("docs", dim=768, drop_if_exists=True)
```

These flags are destructive and should be limited to tests or explicit reset
scripts.

## Ingestion

### Vector dimension error

The collection dimension is fixed:

```python
collection = db.require_collection("docs", dim=768)
```

Every inserted primary vector must have length 768:

```python
vector = np.asarray(vector, dtype=np.float32)
assert vector.shape == (768,)
```

For a different embedding model dimension, create a different collection or a
named vector field with that dimension.

### Duplicate ID error

`add()` expects new public IDs. Use upsert to replace or insert by ID:

```python
collection.upsert(ids=123, vectors=vector, fields={"title": "updated"})
collection.commit()
```

Check existing IDs:

```python
print(collection.is_id_exists(123))
print(collection.max_id)
```

### Writes are not visible as expected

Use `insert_session()` or call `commit()`:

```python
with collection.insert_session() as session:
    session.add(ids="doc-1", vectors=vector)

# or
collection.add(ids="doc-2", vectors=vector)
```

For backup or shutdown, call:

```python
collection.checkpoint()
```

## Search and query

### Search returns no rows

Check:

- vectors were inserted and committed;
- the query vector has the correct dimension;
- `k` is greater than zero;
- the `where` filter is not too restrictive;
- rows were not soft-deleted.

Useful inspection:

```python
print(collection.shape)
print(collection.stats())
print(collection.list_deleted_ids())
print(collection.search_profile(query, k=5, where="tenant = 'acme'"))
```

### `query()` returns empty results

This is expected when no filter is provided:

```python
collection.query()
collection.query_vectors()
```

Pass a filter or explicit IDs:

```python
collection.query(where="tenant = 'acme'")
collection.query(filter_ids=[1, 2, 3])
```

### Metadata is missing from search results

Set `return_fields=True`:

```python
result = collection.search(query, k=10, return_fields=True)
```

Or fetch fields after search:

```python
result = collection.search(query, k=10)
rows = collection.query(filter_ids=result.ids.tolist())
```

### Filter syntax is wrong

Common valid filters:

```python
where = "lang = 'en'"
where = "rank >= 10 AND rank < 20"
where = "published = true"
where = "tags CONTAINS 'vector'"
where = "created_at >= '2026-06-01'"
where = "\"document.lang\" = 'en'"
```

See the [Metadata filter cookbook](metadata_filter_cookbook.md) for more
examples.

## Indexes

### IVF index build fails or behaves badly

Pass `n_clusters`:

```python
collection.build_index("IVF-L2", n_clusters=256)
```

Then tune `nprobe` at search time:

```python
collection.search(query, k=10, nprobe=20)
```

If recall is low, compare with `FLAT-L2`, increase `nprobe`, or use fewer
clusters.

### ANN results differ from expected exact neighbors

Approximate indexes trade recall for speed. Rebuild a flat baseline:

```python
collection.build_index("FLAT-L2")
baseline = collection.search(query, k=10)
```

Then tune the ANN index:

```python
collection.build_index("HNSW-L2")
candidate = collection.search(query, k=10, nprobe=64)
```

### `nprobe` appears to do nothing

`nprobe` controls IVF and HNSW search breadth. Flat, PQ, RaBitQ, PolarVec, and
named vector-field searches may ignore it.

### Binary index scores seem reversed

Hamming and Jaccard are lower-is-better distances. Cosine also returns a
lower-is-better `1 - similarity` distance; only inner product is a
higher-is-better score.

## Named and sparse vectors

### Adding named vectors fails

Check:

- the named field exists;
- vector dimension matches the field dimension;
- IDs already exist in the primary collection;
- the number of vectors equals the number of IDs.

```python
collection.create_vector_field("image", dim=512, metric="l2")
collection.add_named_vectors("image", image_vectors, ids=image_ids)
```

### Sparse vector search fails

Sparse feature IDs must be non-negative integers and weights must be numeric:

```python
collection.add_sparse_vectors([{10: 1.0, 42: 0.5}], ids=[1])
collection.search_sparse({42: 1.0}, k=10)
```

## HTTP server limits

### Request rejected because it is too large

Lower your client batch size:

```python
collection.add(ids=ids, vectors=vectors, fields=fields, batch_size=1000)
```

Or increase server limits:

```shell
lynse serve \
  --data-dir ./server-data \
  --json-limit-mb 512 \
  --payload-limit-mb 1024 \
  --max-batch-vectors 200000
```

### `k` or batch size is rejected

Check:

- `--max-top-k`;
- `--max-batch-vectors`;
- `--max-collection-vectors`;
- `--max-collection-vector-bytes`.

Set a limit to `0` only when you intentionally want to disable that guardrail.

## Deletes and compaction

### Deleted rows still take disk space

Deletes are soft deletes:

```python
collection.delete([1, 2, 3])
collection.commit()
```

Physically remove tombstoned rows during maintenance:

```python
removed = collection.compact()
print(removed)
```

After compaction, rows cannot be restored from the collection itself.

### Deleted row appears again

It may have been restored:

```python
collection.restore([1])
```

Inspect tombstones:

```python
print(collection.list_deleted_ids())
```

## Backups

### Snapshot path is not where expected

In local mode, snapshot and export paths are on the Python process filesystem.
In remote mode, they are on the server filesystem.

```python
client.snapshot_database("app", "./app.snapshot")
```

For remote mode, `./app.snapshot` is relative to the server process working
directory.

### Restore confidence is low

Run a restore drill:

```python
client.restore_database("app_restore_test", "./app.snapshot", overwrite=True)
test_db = client.get_database("app_restore_test")
print(test_db.show_collections_details())
```

## Quick diagnostic script

```python
def inspect_collection(collection):
    print("shape:", collection.shape)
    print("stats:", collection.stats())
    print("index:", collection.index_mode)
    print("fields:", collection.list_fields())
    print("vector_fields:", collection.list_vector_fields())
    print("deleted:", collection.list_deleted_ids()[:10])
    print("max_id:", collection.max_id)
```

Use this before changing data or rebuilding indexes. It gives you a quick view
of the collection state.
