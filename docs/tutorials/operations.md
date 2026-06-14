# Tutorial: Backup and Maintenance

This tutorial covers operational calls for durability, backups, restores, soft
deletes, compaction, and monitoring.

## Flush, checkpoint, and close

```python
collection.flush()
collection.checkpoint()
collection.close()
client.close()
```

Use `commit()` after normal write batches:

```python
with collection.insert_session() as session:
    session.add(ids="ops-example", vectors=[0.1, 0.2, 0.3, 0.4])

# committed automatically by the session
```

Use `checkpoint()` before backups or controlled shutdowns.

Operational meaning:

| Call | Purpose |
| --- | --- |
| `commit()` | Commit normal writes. |
| `flush()` | Flush pending buffers and bytes without clearing the WAL. |
| `checkpoint()` | Force a durable checkpoint and clear committed WAL state. |
| `close()` | Flush and close the handle from the API perspective. |

For scripts, `insert_session()` is usually enough. For services and backup jobs,
call `checkpoint()` before taking snapshots or stopping the process.

## Collection snapshots

Create a collection snapshot:

```python
db.snapshot_collection("items", "./items.snapshot")
```

Restore it into a collection:

```python
db.restore_collection(
    "items_restored",
    "./items.snapshot",
    overwrite=True,
)
```

You can also snapshot from the collection handle:

```python
collection.snapshot_to("./items.snapshot")
```

Snapshot paths are interpreted where the operation runs:

- local mode: path is on the Python process filesystem;
- remote mode: path is on the server filesystem.

Use snapshots when you want a compact operational backup for the same LynseDB
version and environment.

## Database snapshots

```python
client.snapshot_database("app", "./app.snapshot")
client.restore_database("app_restored", "./app.snapshot", overwrite=True)
```

Database-level snapshots are useful for complete environment backups.

Restore into a new name first when testing:

```python
client.restore_database("app_restore_test", "./app.snapshot", overwrite=True)
test_db = client.get_database("app_restore_test")
print(test_db.show_collections())
```

This validates the backup without replacing production data.

## Export and import

Exports are filesystem directories containing JSONL metadata and binary vector
payloads.

```python
db.export_collection("items", "./items-export")
db.import_collection("items_copy", "./items-export", overwrite=True)
```

Use export/import when you want a portable representation instead of a snapshot.

Collection handle shortcuts:

```python
collection.export_to("./items-export")
```

Export/import is useful for:

- moving data between environments;
- inspecting metadata outside LynseDB;
- long-term portable backups;
- migration scripts.

Snapshot/restore is usually faster for operational backups. Export/import is
usually easier to inspect and transform.

## Soft delete and restore

Deletes are logical tombstones:

```python
collection.delete([10, 11])
print(collection.list_deleted_ids())

collection.restore([10])
print(collection.list_deleted_ids())
```

Soft-deleted IDs are excluded from search and query results.

Deletion lifecycle:

1. `delete()` tombstones rows.
2. Search and query exclude tombstoned IDs.
3. `restore()` can bring tombstoned IDs back.
4. `compact()` physically removes tombstoned rows.

After compaction, removed rows cannot be restored from the collection itself.
Use a snapshot or export if you need recovery after compaction.

## Compaction

Compaction physically removes tombstoned vectors and rebuilds storage.

```python
removed = collection.compact()
print(f"removed {removed} rows")
```

Run compaction during a maintenance window for large collections.

Check before and after:

```python
before = collection.stats()
removed = collection.compact()
after = collection.stats()

print(before)
print(removed)
print(after)
```

## Stats and inspection

```python
print(collection.shape)
print(collection.stats())
print(collection.index_mode)
print(collection.list_fields())
print(collection.list_vector_fields())
```

Database-level inspection:

```python
print(client.list_databases())
print(db.show_collections())
print(db.show_collections_details())
```

`show_collections_details()` returns a pandas DataFrame if pandas is installed;
otherwise it returns a list of dictionaries.

Field and vector-field inspection:

```python
print(collection.list_fields())
print(collection.list_vector_fields())
print(collection.max_id)
print(collection.is_id_exists(123))
```

## HTTP operations

For remote deployments:

```shell
curl http://127.0.0.1:7637/healthz
curl http://127.0.0.1:7637/readyz
curl http://127.0.0.1:7637/metrics
curl http://127.0.0.1:7637/openapi.json
```

With authentication:

```shell
curl -H "Authorization: Bearer your_key" http://127.0.0.1:7637/list_databases
```

Important metrics include:

- request counts and request latency;
- WAL bytes;
- data directory bytes;
- vector index bytes;
- process memory;
- index build started, completed, failed, in-progress, and duration values.

Use `/openapi.json` to inspect the exact server schema for automation or custom
clients.

## Backup recipes

### Local collection backup

```python
collection.checkpoint()
collection.snapshot_to("./backups/items.snapshot")
```

### Local database backup

```python
for name in client.list_databases():
    client.snapshot_database(name, f"./backups/{name}.snapshot")
```

### Remote backup

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637", api_key="your_key")
client.snapshot_database("app", "/backups/app.snapshot")
```

Remember that remote paths are server paths.

### Restore drill

```python
client.restore_database("app_drill", "./backups/app.snapshot", overwrite=True)
drill_db = client.get_database("app_drill")
print(drill_db.show_collections_details())
```

Run restore drills regularly. A backup is only useful after you have verified
that it can be restored.

## Migration recipe

Use export/import when you want to transform or move collection data:

```python
db.export_collection("items", "./items-export")
db.import_collection("items_copy", "./items-export", overwrite=True)
```

After import:

```python
copied = db.get_collection("items_copy")
print(copied.shape)
print(copied.stats())
```

Rebuild indexes if your migration changes index strategy:

```python
copied.build_index("HNSW-L2")
copied.checkpoint()
```

## Production checklist

- Use HTTP mode for multi-process access.
- Pin the LynseDB package and server image versions.
- Keep `/healthz`, `/readyz`, and `/metrics` monitored.
- Use API keys when the server is reachable outside a trusted network.
- Call `checkpoint()` before snapshots.
- Test restore procedures before relying on backups.
- Compare ANN recall against a flat baseline before changing index settings.
- Use persistent volumes for Docker and Kubernetes deployments.
- Set request, payload, top-k, batch, and collection-size limits.
- Keep an index rebuild plan for large ingestion or migration jobs.
- Compact tombstoned rows during planned maintenance windows.
- Keep old snapshots until new snapshots have passed a restore drill.
