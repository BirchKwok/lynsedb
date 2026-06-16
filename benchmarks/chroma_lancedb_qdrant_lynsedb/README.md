# ChromaDB, LanceDB, LynseDB, and USEARCH Benchmarks

This benchmark suite compares embedded vector database workloads across:

- LynseDB embedded
- LynseDB HTTP service mode
- ChromaDB persistent local mode
- LanceDB local mode
- USEARCH vector-only HNSW mode

The suite uses one deterministic synthetic dataset per run:

- normalized dense vectors, so inner product, cosine, and L2 rankings are
  comparable for exact recall checks;
- integer `category` metadata for filtered search;
- short `text` fields for BM25 or native hybrid paths where an engine exposes
  one through a simple local API.

Missing third-party engines are reported as `skipped` rows instead of failing
the full benchmark.

## Install

From the repository root:

```shell
pip install -e .
pip install -r benchmarks/chroma_lancedb_qdrant_lynsedb/requirements.txt
```

For a LynseDB-only smoke run, the second command is optional.

## One Command Per Benchmark

Ingest throughput:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py ingest
```

Query latency and QPS:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py query
```

Recall@k against exact NumPy inner-product ground truth:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py recall
```

Filtered vector search latency and filtered recall@k:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py filtered
```

Native hybrid search where the adapter has a simple supported path:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py hybrid
```

Memory and disk footprint after ingest:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py resources
```

Startup time:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py startup
```

LynseDB HTTP mode:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py http
```

Run the full suite:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py all
```

## Useful Options

Run only specific engines:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py query \
  --engines lynsedb,chroma,lancedb,usearch
```

Compare LynseDB `float32`, LanceDB, and USEARCH:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py query \
  --engines lynsedb-f32,lancedb,usearch
```

Run LynseDB embedded and HTTP side by side:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py query \
  --engines lynsedb,lynsedb-http
```

Scale the workload:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py query \
  --n 1000000 \
  --dim 384 \
  --queries 1000 \
  --k 10 \
  --batch-size 5000
```

Append machine-readable results:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py all \
  --jsonl benchmarks/chroma_lancedb_qdrant_lynsedb/results.jsonl
```

Keep generated database files for inspection:

```shell
python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py resources --keep-data
```

By default, data is written under `/tmp/vector_db_bench` and removed after each
engine run.

## Metrics

| Command | Main columns |
| --- | --- |
| `ingest` | persisted vector + metadata/text-index `vectors_per_s`, `ingest_s`, `disk_mb`, RSS delta when `psutil` is installed |
| `query` | `mean_ms`, `p50_ms`, `p95_ms`, `qps`, `recall_at_k` |
| `recall` | same query workload, with recall@k emphasized |
| `filtered` | filtered latency plus `filtered_recall_at_k` |
| `hybrid` | native hybrid latency and `topic_hit_rate` |
| `resources` | memory and disk footprint after ingest |
| `startup` | startup `mean_ms`, `p50_ms`, `p95_ms` |
| `http` | LynseDB HTTP query latency and recall after remote ingest |

`topic_hit_rate` is not a relevance benchmark. It is a lightweight sanity check
for the synthetic hybrid workload: returned IDs should mostly match the topic
encoded in the query text.

## Notes

- LynseDB uses `dtypes="float16"` storage, builds `FLAT-IP` after ingest, and
  uses a lightweight embedded fast checkpoint path for the local benchmark
  adapter.
- `lynsedb-f32` uses the same embedded adapter and `FLAT-IP` index with
  `dtypes="float32"`.
- The `ingest` command measures durable vector writes plus metadata storage and
  the text index used by the hybrid-search workload; it is not a vector-only
  append microbenchmark.
- Chroma uses a persistent local collection with `hnsw:space=ip`.
- LanceDB stores normalized vectors; for normalized vectors, cosine, inner
  product, and L2 produce compatible nearest-neighbor orderings.
- USEARCH is included as a vector-only HNSW index. Metadata filters and hybrid
  search are reported as `skipped` instead of emulated outside USEARCH.
- Hybrid search is reported only where this suite has a simple native adapter.
  Engines without one are marked `skipped` for the `hybrid` command.
