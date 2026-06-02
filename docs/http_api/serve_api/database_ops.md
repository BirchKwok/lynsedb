# HTTP Database Endpoints

All database endpoints return the standard response envelope:

```json
{
  "status": "success",
  "params": {}
}
```

When authentication is enabled, include:

```shell
-H "Authorization: Bearer your_key"
```

## Endpoints

| Method | Path | Body fields | Description |
| --- | --- | --- | --- |
| `POST` | `/create_database` | `database_name`, `drop_if_exists` | Create a database or recreate it when requested. |
| `POST` | `/drop_database` | `database_name` | Drop a database. |
| `POST` | `/delete_database` | `database_name` | Alias for dropping a database. |
| `POST` | `/database_exists` | `database_name` | Check whether a database exists. |
| `GET` | `/list_databases` | none | List database names. |
| `POST` | `/snapshot_database` | `database_name`, `snapshot_path` | Create a database snapshot on the server filesystem. |
| `POST` | `/restore_database` | `database_name`, `snapshot_path`, `overwrite` | Restore a database snapshot. |

## Examples

Create a database:

```shell
curl -X POST http://127.0.0.1:7637/create_database \
  -H "Content-Type: application/json" \
  -d '{"database_name": "app", "drop_if_exists": false}'
```

List databases:

```shell
curl http://127.0.0.1:7637/list_databases
```

Restore a snapshot:

```shell
curl -X POST http://127.0.0.1:7637/restore_database \
  -H "Content-Type: application/json" \
  -d '{"database_name": "app_restored", "snapshot_path": "./app.snapshot", "overwrite": true}'
```

## Python equivalent

```python
client = lynse.VectorDBClient("http://127.0.0.1:7637")

db = client.create_database("app")
print(client.list_databases())

client.snapshot_database("app", "./app.snapshot")
client.restore_database("app_restored", "./app.snapshot", overwrite=True)
```
