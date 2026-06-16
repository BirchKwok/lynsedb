# HTTP Collection Endpoints

The Python HTTP client wraps these endpoints. Use this page when implementing a
custom client or debugging raw HTTP calls.

Most request bodies include:

```json
{
  "database_name": "app",
  "collection_name": "items"
}
```

## Collection management

| Method | Path | Extra body fields | Description |
| --- | --- | --- | --- |
| `POST` | `/required_collection` | `dim`, `drop_if_exists`, `description` | Create or open a collection. |
| `POST` | `/drop_collection` | none | Drop a collection. |
| `POST` | `/show_collections` | database only | List collections in a database. |
| `POST` | `/collection_shape` | none | Return `(n_vectors, dim)`. |
| `POST` | `/stats` | none | Return collection statistics. |
| `POST` | `/get_collection_config` | none | Return collection config. |
| `POST` | `/update_description` | `description` | Update collection description. |
| `POST` | `/close_collection` | none | Flush and close a collection handle. |

## Writes

| Method | Path | Extra body fields | Description |
| --- | --- | --- | --- |
| `POST` | `/add` | `ids`, `vectors`, `fields` | Add one or more records with string or integer public IDs. |
| `POST` | `/upsert` | `ids`, `vectors`, `fields` | Insert or update one or more records by public ID. |
| `POST` | `/commit` | none | Fast logical commit; clears WAL without forcing recursive fsync. |
| `POST` | `/flush` | none | Flush pending buffers and bytes without clearing WAL. |
| `POST` | `/checkpoint` | none | Force a durable checkpoint, sync committed state, and clear WAL. |

Use `/commit` for normal ingestion latency and `/checkpoint` before backups,
snapshots, controlled shutdowns, or critical acknowledgements that require
deterministic on-disk durability.

## Indexes

| Method | Path | Extra body fields | Description |
| --- | --- | --- | --- |
| `POST` | `/build_index` | `index_mode`, `n_clusters` | Build the primary vector index. |
| `POST` | `/remove_index` | none | Remove the primary vector index. |
| `POST` | `/build_vector_field_index` | `field_name`, `index_mode`, `n_clusters` | Build an index for a named vector field. |
| `POST` | `/remove_vector_field_index` | `field_name` | Remove a named vector field index. |

`n_clusters` is accepted only by IVF index modes.

## Named and sparse vectors

| Method | Path | Extra body fields | Description |
| --- | --- | --- | --- |
| `POST` | `/create_vector_field` | `field_name`, `dimension`, `metric`, `index_mode` | Create a named vector field. |
| `POST` | `/list_vector_fields` | none | List vector fields. |
| `POST` | `/add_named_vectors` | `field_name`, `vectors`, `ids` | Attach named vectors to existing IDs. |
| `POST` | `/add_sparse_vectors` | `vectors`, `ids` | Attach sparse feature vectors. |

## Search and retrieval

| Method | Path | Extra body fields | Description |
| --- | --- | --- | --- |
| `POST` | `/search` | `vector` or `document`, `k`, `where`, `return_fields`, `nprobe`, `field_name`, `approx`, `eps` | Vector search, or document search with automatic embedding. |
| `POST` | `/search_binary` | binary payload | Compact vector search protocol. |
| `POST` | `/batch_search` | `vectors`, `k`, `where`, `return_fields`, `nprobe` | Batch vector search. |
| `POST` | `/batch_search_binary` | binary payload | Compact batch search protocol. |
| `POST` | `/search_range` | `vector`, `threshold`, `max_results` | Range search. |
| `POST` | `/search_profile` | `vector`, `k`, `where`, `nprobe` | Search with profile metadata. |
| `POST` | `/bm25_search` | `text`, `text_fields`, `k`, `where` | BM25 search over metadata fields. |
| `POST` | `/sparse_search` | `vector`, `k`, `where` | Sparse vector search. |
| `POST` | `/hybrid_search` | `vector`, `text`, `text_fields`, `fusion`, `k`, `where` | Vector and text hybrid search. |
| `POST` | `/query` | `where`, `filter_ids`, `return_ids_only` | Query IDs and fields. |
| `POST` | `/query_vectors` | `where`, `filter_ids` | Query IDs, vectors, and fields. |
| `POST` | `/head` | `n` | First rows. |
| `GET` | `/head_binary` | query params | First rows through compact protocol. |
| `POST` | `/tail` | `n` | Last rows. |
| `GET` | `/tail_binary` | query params | Last rows through compact protocol. |
| `POST` | `/read_by_only_id` | `id` | Read one or more IDs. |
| `POST` | `/list_fields` | none | List metadata field names. |
| `POST` | `/index_mode` | none | Return current index mode. |
| `POST` | `/is_id_exists` | `id` | Check whether an ID exists. |
| `POST` | `/max_id` | none | Return max internal numeric ID. |

## Delete, restore, compact

| Method | Path | Extra body fields | Description |
| --- | --- | --- | --- |
| `POST` | `/delete` | `ids` | Soft-delete IDs. |
| `POST` | `/restore` | `ids` | Restore soft-deleted IDs. |
| `POST` | `/list_deleted_ids` | none | List tombstoned IDs. |
| `POST` | `/compact` | none | Physically remove tombstoned vectors. |

## Backup and portability

| Method | Path | Extra body fields | Description |
| --- | --- | --- | --- |
| `POST` | `/snapshot_collection` | `snapshot_path` | Create collection snapshot on server filesystem. |
| `POST` | `/restore_collection` | `snapshot_path`, `overwrite` | Restore collection snapshot. |
| `POST` | `/export_collection` | `export_path` | Export collection to JSONL plus binary vectors. |
| `POST` | `/import_collection` | `export_path`, `overwrite` | Import exported collection. |

## Raw search example

```shell
curl -X POST http://127.0.0.1:7637/search \
  -H "Content-Type: application/json" \
  -d '{
    "database_name": "app",
    "collection_name": "items",
    "vector": [0.1, 0.2, 0.3, 0.4],
    "k": 5,
    "where": "lang = '\''en'\''",
    "return_fields": true,
    "nprobe": 20,
    "field_name": "default",
    "approx": false
  }'
```

For the complete machine-readable schema, open `/openapi.json` on a running
server.
