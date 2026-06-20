# Vector Database Benchmarks

This page records a local run of the cross-engine benchmark suite in
`benchmarks/chroma_lancedb_qdrant_lynsedb`.

Benchmark results are environment-sensitive. Treat these numbers as one
reproducible reference run, not as universal performance guarantees. RSS
deltas, startup latency, and short resource-run ingest throughput are especially
sensitive to allocator state, import order, and OS cache behavior.

## Run Details

| Item | Value |
| --- | --- |
| Date | 2026-06-20 |
| Command | `python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py all --engines lynsedb-f32,chroma,lancedb,usearch --n 100000 --queries 100 --dim 128 --k 10 --batch-size 1000 --warmup 5 --jsonl benchmarks/chroma_lancedb_qdrant_lynsedb/results-2026-06-20-rerun.jsonl` |
| Dataset | 100,000 vectors, 128 dimensions, 100 queries, top-k 10 |
| Python | 3.12.2 |
| Platform | macOS 26.5.1, arm64 |
| LynseDB | 0.7.0 |
| ChromaDB | 1.5.9 |
| LanceDB | 0.33.0 |
| USEARCH | 2.25.3 |
| Raw results | `benchmarks/chroma_lancedb_qdrant_lynsedb/results-2026-06-20-rerun.jsonl` |

The suite uses normalized synthetic vectors and exact NumPy inner-product
ground truth for recall checks. This run uses LynseDB `float32` storage with
`FLAT-IP`, ChromaDB's persistent local HNSW collection, LanceDB's normalized
vector table, and USEARCH's float32 HNSW index. Engines are ingested through
their batch APIs.

## Summary

In this 100k-row float32 run, LynseDB leads persisted batch ingest and exact
vector, filtered, and hybrid-search latency among the database adapters while
maintaining 1.000 recall@10. LanceDB uses the least disk. ChromaDB and USEARCH
return approximate results in this configuration, so their latency must be read
alongside recall.

## 100k Float32 Comparison

A run on 2026-06-20 compares LynseDB in `float32` mode with ChromaDB, LanceDB,
and USEARCH:

`python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py all --engines lynsedb-f32,chroma,lancedb,usearch --n 100000 --queries 100 --dim 128 --k 10 --batch-size 1000 --warmup 5 --jsonl benchmarks/chroma_lancedb_qdrant_lynsedb/results-2026-06-20-rerun.jsonl`

LynseDB `float32` uses the same `FLAT-IP` index and exact recall target as the
main embedded run. USEARCH 2.25.3 is included as a vector-only HNSW index with
`metric="ip"`, `dtype="f32"`, and `USEARCH_EXPANSION_SEARCH=128` default. It
does not implement metadata filters or text/vector hybrid search in this suite,
so those rows are skipped instead of emulated in Python.

| Metric | LynseDB-f32 | ChromaDB | LanceDB | USEARCH |
| --- | ---: | ---: | ---: | ---: |
| Batch ingest vectors/s | 73,399 | 2,108 | 68,123 | 10,578 |
| Ingest time | 1.36 s | 47.45 s | 1.47 s | 9.45 s |
| Disk after ingest MB | 69.13 | 162.42 | 55.76 | 63.03 |
| Vector search mean ms | 0.661 | 1.233 | 14.581 | 0.555 |
| Vector search QPS | 1,513 | 811 | 69 | 1,803 |
| Vector search recall@10 | 1.0000 | 0.5180 | 1.0000 | 0.6000 |
| Recall run mean ms | 0.609 | 1.110 | 14.996 | 0.531 |
| Recall run QPS | 1,643 | 901 | 67 | 1,882 |
| Recall run recall@10 | 1.0000 | 0.5180 | 1.0000 | 0.5970 |
| Filtered search mean ms | 0.178 | 37.354 | 16.692 | n/a |
| Filtered search QPS | 5,618 | 27 | 60 | n/a |
| Filtered recall@10 | 1.0000 | 0.9990 | 1.0000 | n/a |
| Hybrid search mean ms | 4.809 | n/a | 17.810 | n/a |
| Hybrid search QPS | 208 | n/a | 56 | n/a |
| Hybrid topic hit rate | 0.6050 | n/a | 0.6000 | n/a |
| Resource-run vectors/s | 70,042 | 1,937 | 75,840 | 11,359 |
| Startup mean ms | 2.087 | 13.995 | 2.251 | 0.036 |

In this `float32` validation, LynseDB leads persisted ingest and exact vector,
filtered, and hybrid-search latency among the database adapters. USEARCH's HNSW
search is slightly faster than LynseDB's exact flat search, but at much lower
recall and without database-level metadata or hybrid retrieval features.

## 1M Row Scale Check

A larger single-machine check was also run on 2026-06-20 with 1,000,000 vectors,
128 dimensions, 100 queries, top-k 10, and batch insert APIs:

`python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py all --engines lynsedb-f32,lancedb --n 1000000 --dim 128 --queries 100 --k 10 --batch-size 5000 --warmup 5 --jsonl benchmarks/chroma_lancedb_qdrant_lynsedb/results-2026-06-20-1m-rerun.jsonl`

ChromaDB is not included in this 1M summary because the recorded run failed
under the selected batch-size settings. LanceDB completed the scale check, so
the table compares LynseDB and LanceDB directly.

| Metric | LynseDB | LanceDB |
| --- | ---: | ---: |
| Batch ingest vectors/s | 49,954 | 85,057 |
| Ingest time | 20.02 s | 11.76 s |
| Disk after ingest MB | 694.32 | 547.69 |
| Vector search mean ms | 6.013 | 109.009 |
| Vector search QPS | 166.31 | 9.17 |
| Vector search recall@10 | 1.0000 | 1.0000 |
| Recall run mean ms | 5.573 | 148.741 |
| Recall run QPS | 179.42 | 6.72 |
| Filtered search mean ms | 2.160 | 148.455 |
| Filtered search QPS | 463.03 | 6.74 |
| Filtered recall@10 | 1.0000 | 1.0000 |

This run shows LynseDB handling 1 million 128-dimensional float32 vectors on a
single machine: ingest completes in about 20.0 seconds, persisted storage is
about 694 MB, exact vector search stays around 6.0 ms mean latency, and
metadata-filtered search stays around 2.2 ms mean latency. LanceDB ingests faster
and uses less disk, while LynseDB has lower exact and filtered-search latency.

## Persisted Vector + Metadata/Text-Index Ingest

This ingest metric includes committed vector writes, metadata persistence, and
the text index needed by the hybrid-search benchmark. It is not a pure
vector-only append microbenchmark.

| Engine | Vectors/s | Disk MB | RSS delta MB |
| --- | ---: | ---: | ---: |
| lynsedb-f32 | 73,399 | 69.13 | 399.83 |
| chroma | 2,108 | 162.42 | 218.95 |
| lancedb | 68,123 | 55.76 | 149.02 |
| usearch | 10,578 | 63.03 | 70.38 |

## Vector Search

| Engine | Mean ms | P50 ms | P95 ms | QPS | Recall@10 | Disk MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lynsedb-f32 | 0.661 | 0.640 | 0.868 | 1,513 | 1.0000 | 69.13 |
| chroma | 1.233 | 1.197 | 1.525 | 811 | 0.5180 | 162.33 |
| lancedb | 14.581 | 14.281 | 16.845 | 69 | 1.0000 | 55.75 |
| usearch | 0.555 | 0.532 | 0.698 | 1,803 | 0.6000 | 63.03 |

## Recall

| Engine | Mean ms | P50 ms | P95 ms | QPS | Recall@10 | Disk MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lynsedb-f32 | 0.609 | 0.596 | 0.698 | 1,643 | 1.0000 | 69.13 |
| chroma | 1.110 | 1.068 | 1.360 | 901 | 0.5180 | 162.38 |
| lancedb | 14.996 | 14.770 | 17.297 | 67 | 1.0000 | 55.75 |
| usearch | 0.531 | 0.527 | 0.654 | 1,882 | 0.5970 | 63.03 |

## Filtered Vector Search

| Engine | Mean ms | P50 ms | P95 ms | QPS | Filtered recall@10 | Disk MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lynsedb-f32 | 0.178 | 0.168 | 0.257 | 5,618 | 1.0000 | 69.13 |
| chroma | 37.354 | 37.247 | 39.086 | 27 | 0.9990 | 162.35 |
| lancedb | 16.692 | 16.500 | 18.656 | 60 | 1.0000 | 55.75 |

## Hybrid Search

| Engine | Mean ms | P50 ms | P95 ms | QPS | Topic hit rate / note | Disk MB |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| lynsedb-f32 | 4.809 | 2.925 | 16.042 | 208 | 0.6050 | 69.13 |
| lancedb | 17.810 | 17.661 | 20.329 | 56 | 0.6000 | 55.77 |

`topic_hit_rate` is a synthetic sanity check for the benchmark's generated
topics. It is not a relevance score for a production retrieval workload.

## Resources

| Engine | Resource-run vectors/s | Disk MB | RSS delta MB |
| --- | ---: | ---: | ---: |
| lynsedb-f32 | 70,042 | 69.13 | 0.02 |
| chroma | 1,937 | 162.40 | 84.09 |
| lancedb | 75,840 | 55.75 | 5.52 |
| usearch | 11,359 | 63.03 | 67.61 |

RSS deltas are measured in-process and can vary with allocator state and import
order. Negative deltas can occur when a later run reuses allocator state or the
runtime releases memory during the measurement. Use disk size and repeated runs
when comparing storage footprint.

## Startup

| Engine | Startup mean ms | Startup P50 ms | Startup P95 ms | QPS |
| --- | ---: | ---: | ---: | ---: |
| lynsedb-f32 | 2.087 | 1.900 | 2.746 | 479 |
| chroma | 13.995 | 14.085 | 14.826 | 71 |
| lancedb | 2.251 | 2.201 | 2.706 | 444 |
| usearch | 0.036 | 0.023 | 0.073 | 27,868 |

## LynseDB HTTP

| Engine | Mean ms | P50 ms | P95 ms | QPS | Recall@10 | Disk MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lynsedb-http | 2.439 | 2.163 | 4.812 | 410 | 0.9990 | 44.71 |

The HTTP row starts a local LynseDB service, ingests the same dataset remotely,
and runs the same query workload through the client.
