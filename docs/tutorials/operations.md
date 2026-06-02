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
    session.add_item([0.1, 0.2, 0.3, 0.4], id=1)

# committed automatically by the session
```

Use `checkpoint()` before backups or controlled shutdowns.

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

## Database snapshots

```python
client.snapshot_database("app", "./app.snapshot")
client.restore_database("app_restored", "./app.snapshot", overwrite=True)
```

Database-level snapshots are useful for complete environment backups.

## Export and import

Exports are filesystem directories containing JSONL metadata and binary vector
payloads.

```python
db.export_collection("items", "./items-export")
db.import_collection("items_copy", "./items-export", overwrite=True)
```

Use export/import when you want a portable representation instead of a snapshot.

## Soft delete and restore

Deletes are logical tombstones:

```python
collection.delete_items([10, 11])
print(collection.list_deleted_ids())

collection.restore_items([10])
print(collection.list_deleted_ids())
```

Soft-deleted IDs are excluded from search and query results.

## Compaction

Compaction physically removes tombstoned vectors and rebuilds storage.

```python
removed = collection.compact()
print(f"removed {removed} rows")
```

Run compaction during a maintenance window for large collections.

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

## Production checklist

- Use HTTP mode for multi-process access.
- Pin the LynseDB package and server image versions.
- Keep `/healthz`, `/readyz`, and `/metrics` monitored.
- Use API keys when the server is reachable outside a trusted network.
- Call `checkpoint()` before snapshots.
- Test restore procedures before relying on backups.
- Compare ANN recall against a flat baseline before changing index settings.
