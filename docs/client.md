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
| `require_collection(collection, dim=None, n_threads=10, warm_up=False, drop_if_exists=False, description=None, dtypes="float32", default_index="FLAT-IP")` | Create or open a collection. Use `dtypes="float16"` for half-precision vector storage. New collections build `default_index` after the first primary vector write; pass `default_index=None` to disable this. |
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
| `with collection:` | Collection context manager that calls `commit()` on successful exit. |
| `add(ids, *, vectors=None, documents=None, fields=None, batch_size=1000, wire_dtype="float32")` | Add one or more records. Provide vectors directly, or provide documents without vectors to trigger lazy default embedding. IDs may be strings or non-negative integers. |
| `upsert(ids, *, vectors, fields=None, batch_size=1000, wire_dtype="float32")` | Insert or update one or more records by public ID. |
| `commit()` | Fast logical commit: make writes visible and clear WAL without forcing recursive fsync. |
| `flush()` | Flush pending buffers and bytes without clearing WAL. Advanced storage operation. |
| `checkpoint()` | Force a durable checkpoint, sync committed state, and clear WAL. |
| `close()` | Flush and close the collection handle. |

`commit()` is optimized for normal ingestion latency. It does not guarantee that
data has reached stable storage at the instant the call returns; operating
system writeback controls that timing. Call `checkpoint()` when a batch must be
durably on disk before backup, snapshot, shutdown, or acknowledging a critical
write. `flush()` is lower-level: it pushes pending bytes but intentionally keeps
WAL state.

## Collection Indexes and Vector Fields

| Method | Description |
| --- | --- |
| `build_index(index_mode="FLAT-IP", field_name="default", n_clusters=None)` | Build or change an index. IVF and SPANN use `n_clusters`; other index modes ignore it. |
| `remove_index(field_name="default")` | Remove the primary or named-field index. |
| `create_vector_field(name, dim, metric="ip", index_mode=None)` | Create a named vector field. |
| `list_vector_fields()` | List `default` and named vector fields. |
| `add_named_vectors(field_name, vectors, ids)` | Attach named-field vectors to existing IDs. |
| `add_sparse_vectors(vectors, ids)` | Attach sparse feature vectors to existing IDs. |

## Collection Search

| Method | Description |
| --- | --- |
| `search(vector=None, k=10, *, document=None, where=None, return_fields=False, vector_field="default", reranker=None, rerank_k=None, rerank_with_fields=False, nprobe=10, approx=False, eps=1e-4, wire_dtype="float32")` | Dense vector search, or semantic document search when `document` is provided instead of `vector`. |
| `batch_search(vectors, k=10, *, where=None, return_fields=False, nprobe=10, reranker=None, rerank_k=None, rerank_with_fields=False)` | Search multiple query vectors. |
| `search_range(vector, threshold, max_results=1000)` | Return all matches within a metric-specific threshold. |
| `search_profile(vector, k=10, *, where=None, nprobe=10)` | Search with explain/profile metadata. |
| `search_sparse(vector, k=10, *, where=None, return_fields=False, reranker=None, rerank_k=None, rerank_with_fields=True)` | Sparse inner-product search. |
| `bm25_search(text, k=10, *, text_fields=None, where=None, return_fields=False, reranker=None, rerank_k=None, rerank_with_fields=True)` | BM25 search over metadata fields. |
| `hybrid_search(vector=None, text=None, k=10, *, where=None, text_fields=None, fusion="rrf", vector_weight=1.0, text_weight=1.0, rrf_k=60.0, candidate_limit=None, nprobe=10, return_fields=False, reranker=None, rerank_k=None, rerank_with_fields=True)` | Dense plus text hybrid search. |

Parameter behavior is the same for local and HTTP Python clients:

- New collections default to `default_index="FLAT-IP"`. The index is built lazily
  after the first primary vector write so collections with inferred dimensions
  can still be created without specifying `dim`, and the default inner-product
  metric is explicit.
- `n_clusters` is used only by IVF and SPANN index modes. Passing it to Flat,
  PQ, RaBitQ, PolarVec, HNSW, or DiskANN modes is allowed and ignored.
- `nprobe` controls IVF/SPANN partitions and HNSW search breadth. Flat, PQ,
  RaBitQ, PolarVec, and named vector-field searches ignore it.
- `approx` and `eps` apply only to supported flat IP, L2, and cosine paths.
  Hamming, Jaccard/Tanimoto, Dice, and domain metric paths ignore them.
- Domain metrics include L1, Haversine, correlation, Hellinger,
  Wasserstein-1D, Tanimoto, and Dice. See
  [Domain-aware distance metrics](tutorials/distance_metrics.md) for aliases,
  input contracts, and index compatibility.

`where` accepts standard SQL-style metadata filters, for example
`"lang = 'en' AND rank <= 10"`, `"tags CONTAINS 'vector'"`, and
`"created_at >= '2026-06-01'"`.

## Default Document Embedding

`add(documents=...)` and `search(document=...)` use the default local text
embedding adapter when vectors are not provided. Install it explicitly for
repeatable environments:

```shell
pip install "lynsedb[embeddings]"
```

Configuration environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `LYNSE_TEXT_EMBEDDING_ADAPTER` | `fastembed` | Default text embedding adapter. |
| `LYNSE_TEXT_EMBEDDING_MODEL` | `Qdrant/clip-ViT-B-32-text` | Model used by the default adapter. |
| `LYNSE_MODEL_CACHE` | `~/.cache/lynse/models` | Local model cache directory. |
| `LYNSE_AUTO_INSTALL_EMBEDDINGS` | `1` | Set to `0` to disable lazy `fastembed` installation and require explicit dependencies. |

For production retrieval, choose and version your embedding model deliberately,
or pass precomputed vectors through `vectors=`.

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
| `max_id` | Highest internal numeric ID, mainly for diagnostics. |
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
| `delete(ids)` | Soft-delete IDs. |
| `restore(ids)` | Restore soft-deleted IDs. |
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
