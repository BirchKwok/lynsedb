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
| Date | 2026-06-16 |
| Command | `python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py all --engines lynsedb,lancedb --n 100000 --queries 100 --dim 128 --k 10 --batch-size 1000 --warmup 5 --jsonl benchmarks/chroma_lancedb_qdrant_lynsedb/results-2026-06-16-optimized-final.jsonl` |
| Dataset | 100,000 vectors, 128 dimensions, 100 queries, top-k 10 |
| Python | 3.12.2 |
| Platform | macOS 26.5.1, arm64 |
| LynseDB | 0.5.0 |
| ChromaDB | Not rerun in this LynseDB/LanceDB validation |
| LanceDB | 0.33.0 |
| Raw results | `benchmarks/chroma_lancedb_qdrant_lynsedb/results-2026-06-16-optimized-final.jsonl` |

The suite uses normalized synthetic vectors and exact NumPy inner-product
ground truth for recall checks. LynseDB uses `FLAT-IP`, `float16` dense-vector
storage, compact persisted text indexes, and the embedded benchmark adapter's
fast commit path. LanceDB stores normalized vectors. Both embedded engines
are ingested through their batch insert APIs.

## Summary

In this optimized 100k-row run, LynseDB is faster than LanceDB on persisted
batch ingest, vector search, filtered search, hybrid search, resource-run
ingest, and startup, while using less disk. LynseDB's `float16` vector storage
keeps the footprint smaller; against the benchmark's original `float32` NumPy
ground truth this run measured 0.999 recall@10 on unfiltered vector search and
1.000 filtered recall@10.

## Float32 + USEARCH Follow-Up

A follow-up run on 2026-06-16 compares LynseDB in `float32` mode with LanceDB
and USEARCH:

`python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py <command> --engines lynsedb-f32,lancedb,usearch --n 100000 --queries 100 --dim 128 --k 10 --batch-size 1000 --warmup 5 --jsonl benchmarks/chroma_lancedb_qdrant_lynsedb/results-2026-06-16-f32-usearch.jsonl`

LynseDB `float32` uses the same `FLAT-IP` index and exact recall target as the
main embedded run. USEARCH 2.25.3 is included as a vector-only HNSW index with
`metric="ip"`, `dtype="f32"`, and `USEARCH_EXPANSION_SEARCH=128` default. It
does not implement metadata filters or text/vector hybrid search in this suite,
so those rows are skipped instead of emulated in Python.

| Metric | LynseDB-f32 | LanceDB | USEARCH |
| --- | ---: | ---: | ---: |
| Batch ingest vectors/s | 88,542 | 70,577 | 10,351 |
| Ingest time | 1.13 s | 1.42 s | 9.66 s |
| Disk after ingest MB | 66.75 | 55.75 | 63.03 |
| Vector search mean ms | 0.671 | 12.620 | 0.571 |
| Vector search QPS | 1,491 | 79 | 1,751 |
| Vector search recall@10 | 1.0000 | 1.0000 | 0.6020 |
| Recall run mean ms | 0.606 | 12.385 | 0.510 |
| Recall run QPS | 1,650 | 81 | 1,961 |
| Recall run recall@10 | 1.0000 | 1.0000 | 0.5850 |
| Filtered search mean ms | 0.118 | 14.442 | n/a |
| Filtered search QPS | 8,474 | 69 | n/a |
| Filtered recall@10 | 1.0000 | 1.0000 | n/a |
| Hybrid search mean ms | 4.634 | 13.103 | n/a |
| Hybrid search QPS | 216 | 76 | n/a |
| Hybrid topic hit rate | 0.6040 | 0.6020 | n/a |
| Resource-run vectors/s | 92,086 | 86,914 | 14,762 |
| Startup mean ms | 6.782 | 160.300 | 2.068 |

In this `float32` validation, LynseDB remains ahead of LanceDB on persisted
ingest, exact vector search, filtered search, hybrid search, and the separate
resource-run ingest check. USEARCH's HNSW search is slightly faster than
LynseDB's exact flat search on this run, but at much lower recall and without
database-level metadata or hybrid retrieval features.

## 1M Row Scale Check

A larger single-machine check was also run on 2026-06-16 with 1,000,000 vectors,
128 dimensions, 100 queries, top-k 10, and batch insert APIs:

`python benchmarks/chroma_lancedb_qdrant_lynsedb/bench.py ... --n 1000000 --dim 128 --queries 100 --k 10 --jsonl benchmarks/chroma_lancedb_qdrant_lynsedb/results-2026-06-16-1m.jsonl`

ChromaDB is not included in this 1M summary because the recorded run failed
under the selected batch-size settings. LanceDB completed the scale check, so
the table compares LynseDB and LanceDB directly.

| Metric | LynseDB | LanceDB |
| --- | ---: | ---: |
| Batch ingest vectors/s | 87,760 | 94,007 |
| Ingest time | 11.39 s | 10.64 s |
| Disk after ingest MB | 405.68 | 541.83 |
| Vector search mean ms | 15.547 | 113.123 |
| Vector search QPS | 64.32 | 8.84 |
| Vector search recall@10 | 1.0000 | 1.0000 |
| Recall run mean ms | 13.518 | 120.522 |
| Recall run QPS | 73.97 | 8.30 |
| Filtered search mean ms | 2.139 | 122.961 |
| Filtered search QPS | 467.42 | 8.13 |
| Filtered recall@10 | 0.9990 | 1.0000 |

This run shows LynseDB comfortably handling 1 million 128-dimensional vectors on
a single machine: ingest completes in about 11.4 seconds, persisted storage is
about 406 MB, exact vector search stays around 15.5 ms mean latency, and
metadata-filtered search stays around 2.1 ms mean latency.

## Persisted Vector + Metadata/Text-Index Ingest

This ingest metric includes committed vector writes, metadata persistence, and
the text index needed by the hybrid-search benchmark. It is not a pure
vector-only append microbenchmark.

| Engine | Vectors/s | Disk MB | RSS delta MB |
| --- | ---: | ---: | ---: |
| lynsedb | 96,940 | 42.34 | 390.67 |
| lancedb | 88,788 | 55.74 | 211.30 |

## Vector Search

| Engine | Mean ms | P50 ms | P95 ms | QPS | Recall@10 | Disk MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lynsedb | 1.811 | 1.782 | 2.138 | 552 | 0.9990 | 42.34 |
| lancedb | 11.449 | 11.293 | 12.908 | 87 | 1.0000 | 55.76 |

## Recall

| Engine | Mean ms | P50 ms | P95 ms | QPS | Recall@10 | Disk MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lynsedb | 1.856 | 1.789 | 2.044 | 539 | 0.9990 | 42.34 |
| lancedb | 14.475 | 14.385 | 16.969 | 69 | 1.0000 | 55.75 |

## Filtered Vector Search

| Engine | Mean ms | P50 ms | P95 ms | QPS | Filtered recall@10 | Disk MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lynsedb | 0.204 | 0.211 | 0.272 | 4,898 | 1.0000 | 42.34 |
| lancedb | 18.373 | 18.131 | 21.630 | 54 | 1.0000 | 55.75 |

## Hybrid Search

| Engine | Mean ms | P50 ms | P95 ms | QPS | Topic hit rate / note | Disk MB |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| lynsedb | 5.820 | 4.094 | 16.494 | 172 | 0.6040 | 42.34 |
| lancedb | 18.545 | 18.531 | 21.542 | 54 | 0.6030 | 55.74 |

`topic_hit_rate` is a synthetic sanity check for the benchmark's generated
topics. It is not a relevance score for a production retrieval workload.

## Resources

| Engine | Resource-run vectors/s | Disk MB | RSS delta MB |
| --- | ---: | ---: | ---: |
| lynsedb | 96,756 | 42.34 | -0.44 |
| lancedb | 75,190 | 55.75 | 11.31 |

RSS deltas are measured in-process and can vary with allocator state and import
order. Negative deltas can occur when a later run reuses allocator state or the
runtime releases memory during the measurement. Use disk size and repeated runs
when comparing storage footprint.

## Startup

| Engine | Startup mean ms | Startup P50 ms | Startup P95 ms | QPS |
| --- | ---: | ---: | ---: | ---: |
| lynsedb | 1.986 | 1.815 | 2.713 | 504 |
| lancedb | 2.106 | 2.211 | 2.386 | 475 |

## LynseDB HTTP

| Engine | Mean ms | P50 ms | P95 ms | QPS | Recall@10 | Disk MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lynsedb-http | 2.876 | 2.837 | 3.354 | 348 | 0.9990 | 44.71 |

The HTTP row starts a local LynseDB service, ingests the same dataset remotely,
and runs the same query workload through the client.
