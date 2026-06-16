# Tutorial: Performance Tuning

Performance tuning in LynseDB is mostly about four choices:

- data shape: dimension, number of vectors, metadata size, named fields;
- write path: batch size, commit frequency, local vs remote mode;
- search path: index family, metric, filters, `k`, `nprobe`, and returned fields;
- operations: server limits, snapshots, compaction, and monitoring.

Always measure with your own embeddings and query workload. Vector databases are
sensitive to data distribution.

## 1. Establish a baseline

Start with a small benchmark that matches your application:

```python
import time
import numpy as np
import lynse

dim = 128
n = 50_000
k = 10

client = lynse.VectorDBClient(uri="./perf-demo")
db = client.create_database("perf", drop_if_exists=True)
collection = db.require_collection("vectors", dim=dim, drop_if_exists=True)

vectors = np.random.rand(n, dim).astype(np.float32)
collection.add(ids=list(range(n)), vectors=vectors, batch_size=10_000)

query = np.random.rand(dim).astype(np.float32)

collection.build_index("FLAT-L2")
start = time.perf_counter()
flat = collection.search(query, k=k)
flat_ms = (time.perf_counter() - start) * 1000

collection.build_index("HNSW-L2")
start = time.perf_counter()
hnsw = collection.search(query, k=k, nprobe=64)
hnsw_ms = (time.perf_counter() - start) * 1000

print("flat_ms", flat_ms, flat.ids.tolist())
print("hnsw_ms", hnsw_ms, hnsw.ids.tolist())
```

Use the flat result as a quality reference when evaluating approximate indexes.

## 2. Tune ingestion

Use `add()` for both vector-only and metadata-rich batches:

```python
collection.add(
    ids=ids,
    vectors=vectors,
    fields=fields,
    batch_size=1000,
)
```

Use `insert_session()` when a pipeline produces many batches:

```python
with collection.insert_session() as session:
    for ids, vectors, fields in embedding_batches:
        session.add(
            ids=ids,
            vectors=vectors,
            fields=fields,
            batch_size=1000,
        )
```

Ingestion tips:

- convert embeddings to contiguous `float32` arrays before insertion;
- choose batch sizes that fit memory comfortably;
- commit after meaningful batches, not after every row;
- checkpoint at backup, shutdown, or critical durability boundaries;
- build or rebuild indexes after bulk loading;
- use local mode for single-process offline ingestion when possible;
- use remote mode when several processes must share one database.

## 3. Tune search payload size

Returning fields increases payload size:

```python
fast = collection.search(query, k=10)
rich = collection.search(query, k=10, return_fields=True)
```

Use `return_fields=False` in hot paths when IDs and scores are enough. Fetch
fields later for the final IDs if needed:

```python
result = collection.search(query, k=10)
rows = collection.query(filter_ids=result.ids.tolist())
```

Use `query_vectors()` only when raw vectors are actually needed.

## 4. Tune `k` and server limits

Large `k` values can dominate latency and response size. Keep `k` close to what
the user interface or downstream model actually consumes.

In server mode, protect shared deployments:

```shell
lynse serve \
  --data-dir ./server-data \
  --max-top-k 1000 \
  --max-batch-vectors 50000 \
  --max-collection-vectors 10000000 \
  --max-collection-vector-bytes 1099511627776
```

Set a lower `--max-top-k` for user-facing APIs.

## 5. Tune filters

Selective metadata filters reduce candidate work:

```python
collection.search(
    query,
    k=10,
    where="tenant = 'acme' AND lang = 'en' AND published = true",
)
```

Filter tips:

- keep field types stable;
- use tenant, language, visibility, category, and date filters early;
- use `CONTAINS` for tag arrays;
- use `filter_ids` when you already know candidate IDs;
- inspect surprising behavior with `search_profile()`.

```python
profile = collection.search_profile(
    query,
    k=10,
    where="tenant = 'acme'",
)
print(profile)
```

## 6. Tune index family

| Need | Try |
| --- | --- |
| maximum recall and simple behavior | `FLAT-*` |
| low-latency online search | `HNSW-*` |
| explicit recall/latency tradeoff | `IVF-*` with `n_clusters` and `nprobe` |
| lower memory pressure from graph search | `DiskANN-*` |
| smaller memory or disk footprint | SQ8, PQ, RaBitQ, or PolarVec variants |
| binary vectors | Hamming or Jaccard binary indexes |

Start every tuning session with a flat baseline:

```python
collection.build_index("FLAT-COS")
baseline = collection.search(query, k=20)
```

Then compare alternatives:

```python
collection.build_index("HNSW-Cos")
candidate = collection.search(query, k=20, nprobe=64)
```

## 7. Tune IVF

Build:

```python
collection.build_index("IVF-L2", n_clusters=256)
```

Search:

```python
result = collection.search(query, k=10, nprobe=20)
```

IVF knobs:

- more clusters can reduce scanned vectors per query;
- too many clusters can hurt recall unless `nprobe` also increases;
- higher `nprobe` improves recall and increases latency;
- compare against flat results for representative queries.

## 8. Tune HNSW

```python
collection.build_index("HNSW-L2")
result = collection.search(query, k=10, nprobe=64)
```

For HNSW, `nprobe` acts as the search breadth. Increase it for recall. Decrease
it for latency.

## 9. Tune quantized indexes

Quantized indexes are useful when memory bandwidth, index size, or disk size is
the bottleneck:

```python
collection.build_index("FLAT-L2-SQ8")
collection.build_index("FLAT-L2-PQ")
collection.build_index("FLAT-L2-RABITQ")
collection.build_index("FLAT-L2-POLARVEC")
```

Evaluate quality carefully:

```python
flat_ids = collection.search(query, k=20).ids.tolist()

collection.build_index("FLAT-L2-PQ")
pq_ids = collection.search(query, k=20).ids.tolist()

overlap = len(set(flat_ids) & set(pq_ids)) / max(1, len(flat_ids))
print(overlap)
```

Overlap with flat results is not the same as user relevance, but it is a useful
first check.

## 10. Monitor server mode

Server mode exposes:

```shell
curl http://127.0.0.1:7637/healthz
curl http://127.0.0.1:7637/readyz
curl http://127.0.0.1:7637/metrics
```

Watch:

- request counts and latency;
- slow query warnings;
- WAL bytes;
- data directory bytes;
- vector index bytes;
- process memory;
- index build progress and failures.

Set slow query warnings:

```shell
LYNSE_SLOW_QUERY_WARN_MS=250 lynse serve --data-dir ./server-data
```

## 11. Maintain storage

Deletes are soft deletes. Many tombstones can waste space and affect
inspection:

```python
collection.delete(ids_to_remove)
collection.commit()

print(collection.stats())
collection.checkpoint()
removed = collection.compact()
print(removed)
```

Run compaction during a maintenance window for large collections. Take a
snapshot or export before risky maintenance. Use `checkpoint()` when you need a
deterministic durability boundary; normal `commit()` is the faster logical write
boundary.

## 12. Performance checklist

- Use contiguous `float32` vectors.
- Batch writes and avoid per-row commits.
- Use `checkpoint()` at operational boundaries instead of after every small
  batch.
- Build indexes after large ingestion jobs.
- Keep metadata fields useful but not bloated.
- Use filters to reduce candidate work.
- Keep `k` as small as your product allows.
- Return fields only when needed.
- Use flat search as a recall baseline.
- Tune `nprobe` for HNSW and IVF.
- Use quantized indexes only after measuring quality.
- Monitor server metrics and slow query logs.
