# Python Client API

The recommended entry point is `lynse.VectorDBClient`. It selects local or
remote mode from `uri` and returns database and collection clients with the same
high-level API.

## VectorDBClient

```python
import lynse

client = lynse.VectorDBClient(uri=None, api_key=None, read_only=False)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `uri` | `str | pathlib.Path | None` | `None` or a filesystem path uses local mode. `http://...` or `https://...` uses remote mode. |
| `api_key` | `str | None` | Bearer token for remote HTTP auth. Ignored in local mode. |
| `read_only` | `bool` | Open local storage read-only. Ignored in remote mode. |

| Method | Description |
| --- | --- |
| `create_database(database_name, drop_if_exists=False)` | Create or open a database. |
| `get_database(database_name)` | Open an existing database. |
| `list_databases()` | Return database names. |
| `drop_database(database_name)` | Drop a database. |
| `snapshot_database(database_name, snapshot_path)` | Snapshot a database to a server/local filesystem path. |
| `restore_database(database_name, snapshot_path, overwrite=False)` | Restore a database snapshot. |
| `close()` | Close the underlying HTTP client or local manager handle. |

## Database Client

Returned by `create_database()` or `get_database()`.

| Method | Description |
| --- | --- |
| `require_collection(collection, dim=None, n_threads=10, warm_up=False, drop_if_exists=False, description=None, dtypes="float32")` | Create or open a collection. Use `dtypes="float16"` for half-precision vector storage. |
| `get_collection(collection, warm_up=True)` | Open an existing collection. |
| `drop_collection(collection)` | Drop a collection. |
| `show_collections()` | List collection names. |
| `show_collections_details()` | Return collection details as pandas DataFrame when pandas is available, otherwise list of dicts. |
| `database_exists()` | Check database existence. |
| `update_collection_description(collection, description)` | Update collection description. |
| `snapshot_collection(collection, snapshot_path)` | Snapshot one collection. |
| `restore_collection(collection, snapshot_path, overwrite=False)` | Restore one collection snapshot. |
| `export_collection(collection, export_path)` | Export one collection to JSONL plus binary vectors. |
| `import_collection(collection, export_path, overwrite=False)` | Import a collection export. |
| `snapshot_database(snapshot_path)` | Snapshot this database. |
| `restore_database(snapshot_path, overwrite=False)` | Restore this database from a snapshot. |
| `drop_database()` | Drop this database. |
| `is_read_only` | Local database property indicating whether storage was opened read-only. |

Remote-only database helpers:

| Method | Description |
| --- | --- |
| `set_environment(env)` | Set server environment values. |
| `get_environment()` | Read server environment values. |

## Collection Writes

| Method | Description |
| --- | --- |
| `insert_session()` | Context manager that commits on success and discards pending buffered writes on exception. |
| `add_item(vector, id, *, field=None, buffer_size=True)` | Add one dense vector with optional metadata. |
| `bulk_add_items(vectors, batch_size=1000, enable_progress_bar=True)` | Add `(vector, id)` or `(vector, id, field)` tuples. |
| `bulk_add_binary(vectors, batch_size=50000, enable_progress_bar=True)` | Add a dense `float32` array without metadata in the same call; IDs are assigned sequentially after `max_id`. |
| `upsert_item(vector, id, *, field=None)` | Insert or update one row. |
| `upsert_items(vectors, batch_size=1000, enable_progress_bar=True)` | Insert or update many rows. |
| `commit()` | Commit pending writes. |
| `flush()` | Flush client/storage buffers. |
| `checkpoint()` | Force a durable checkpoint. |
| `close()` | Flush and close the collection handle. |

## Collection Indexes and Vector Fields

| Method | Description |
| --- | --- |
| `build_index(index_mode="FLAT", field_name="default", n_clusters=None)` | Build or change an index. IVF uses `n_clusters`; other index modes ignore it. |
| `remove_index(field_name="default")` | Remove the primary or named-field index. |
| `create_vector_field(name, dim, metric="ip", index_mode=None)` | Create a named vector field. |
| `list_vector_fields()` | List `default` and named vector fields. |
| `add_named_vectors(field_name, vectors, ids)` | Attach named-field vectors to existing IDs. |
| `add_sparse_vectors(vectors, ids)` | Attach sparse feature vectors to existing IDs. |

## Collection Search

| Method | Description |
| --- | --- |
| `search(vector, k=10, *, where=None, return_fields=False, vector_field="default", reranker=None, rerank_k=None, rerank_with_fields=False, nprobe=10, approx=False, eps=1e-4)` | Dense vector search. |
| `batch_search(vectors, k=10, *, where=None, return_fields=False, nprobe=10, reranker=None, rerank_k=None, rerank_with_fields=False)` | Search multiple query vectors. |
| `search_range(vector, threshold, max_results=1000)` | Return all matches within a metric-specific threshold. |
| `search_profile(vector, k=10, *, where=None, nprobe=10)` | Search with explain/profile metadata. |
| `search_sparse(vector, k=10, *, where=None, return_fields=False, reranker=None, rerank_k=None, rerank_with_fields=True)` | Sparse inner-product search. |
| `text_search(text, k=10, *, text_fields=None, where=None, return_fields=False, reranker=None, rerank_k=None, rerank_with_fields=True)` | BM25 search over metadata fields. |
| `hybrid_search(vector=None, text=None, k=10, *, where=None, text_fields=None, fusion="rrf", vector_weight=1.0, text_weight=1.0, rrf_k=60.0, candidate_limit=None, nprobe=10, return_fields=False, reranker=None, rerank_k=None, rerank_with_fields=True)` | Dense plus text hybrid search. |

Parameter behavior is the same for local and HTTP Python clients:

- `n_clusters` is used only by IVF index modes. Passing it to Flat, PQ, RaBitQ,
  PolarVec, HNSW, or DiskANN modes is allowed and ignored.
- `nprobe` controls IVF partitions and HNSW search breadth. Flat, PQ, RaBitQ,
  PolarVec, and named vector-field searches ignore it.
- `approx` and `eps` apply only to supported flat IP, L2, and cosine paths.
  Hamming/Jaccard metrics and non-approximate paths ignore them.

`where` accepts standard SQL-style metadata filters, for example
`"lang = 'en' AND rank <= 10"`, `"tags CONTAINS 'vector'"`, and
`"created_at >= '2026-06-01'"`.

## Collection Query and Data Access

| Method | Description |
| --- | --- |
| `query(where=None, filter_ids=None, return_ids_only=False)` | Return IDs and optional fields. Empty args return an empty result. |
| `query_vectors(where=None, filter_ids=None)` | Return vectors, IDs, and fields. Empty args return an empty `(0, dim)` result. |
| `head(n=5)` | First rows. |
| `tail(n=5)` | Last rows. |
| `list_fields()` | List metadata field names. |
| `exists()` | Check whether the collection exists. |
| `is_read_only` | Local collection property indicating whether storage is read-only. |
| `is_id_exists(id)` | Check whether an ID exists. |
| `max_id` | Highest stored external ID. |
| `shape` | `(n_vectors, dim)`. |
| `stats()` | Collection statistics. |
| `index_mode` | Current primary index mode. |

Remote-only collection helpers:

| Method | Description |
| --- | --- |
| `read_by_only_id(id)` | Read one ID or a list of IDs and return vectors, IDs, and fields. Prefer `query_vectors(filter_ids=...)` for portable local/remote code. |
| `get_collection_path()` | Return the collection path on the server filesystem. Useful for debugging server deployments. |

## Delete and Maintenance

| Method | Description |
| --- | --- |
| `delete_items(ids)` | Soft-delete IDs. |
| `restore_items(ids)` | Restore soft-deleted IDs. |
| `list_deleted_ids()` | List tombstoned IDs. |
| `compact()` | Physically remove tombstoned rows. |
| `snapshot_to(snapshot_path)` | Snapshot this collection. |
| `export_to(export_path)` | Export this collection. |
| `update_description(description)` | Update collection description. |

## Further Reading

- [Learning path](tutorials/learning_path.md) for the recommended tutorial order.
- [Metadata filter cookbook](tutorials/metadata_filter_cookbook.md) for `where`
  examples.
- [Indexing guide](tutorials/indexing.md) for supported index names and tuning.
- [ResultView](result_view.md) for return object conversions.
