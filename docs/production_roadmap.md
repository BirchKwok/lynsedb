# LynseDB Embedded And Server Roadmap

This document turns the embedded/server vector database plan into an
engineering checklist. The near-term goal is to make LynseDB a reliable
embedded-first database that can also run as a standalone single-node service.

## Product Positioning

LynseDB should optimize for this shape:

- Embedded-first local vector database for Python applications.
- Server-optional deployment with the same storage engine and data directory.
- Single-node production reliability before distributed features.
- Clear APIs, stable storage format, and predictable recovery behavior.

The target user experience is:

- Use `LocalClient` in notebooks, scripts, and applications without running a
  separate service.
- Run `lynsedb serve` against the same data directory when a process-safe or
  remotely accessible service is needed.
- Move from embedded mode to server mode without data migration.

## Current Baseline

The repository already has the main building blocks:

- Rust core engine with collections, vector storage, field storage, WAL, ID map,
  tombstones, and compaction.
- Multiple vector search paths: Flat, IVF, HNSW, DiskANN, PQ, RaBitQ, PolarVec,
  SQ8, and binary distances.
- Python local client through PyO3 bindings.
- HTTP server and Python HTTP client.
- Basic API-key authentication.
- Metadata filtering through SQL-like field expressions.

## Milestone 1: Reliability Core

This is the first coding priority. No higher-level feature should depend on
undefined durability or ID semantics.

- Define write visibility and commit semantics.
- Make WAL replay idempotent.
- Store user IDs in WAL records.
- Ensure WAL tracks vector record count even when metadata fields are absent.
- Recover custom user IDs after an uncommitted batch.
- Prevent duplicate ID ambiguity on plain insert.
- Add explicit upsert/update behavior after insert semantics are stable.
- Add crash/reopen tests for vectors, IDs, fields, tombstones, and indexes.
- Ensure index metadata and index files are atomically swapped.
- Define and persist storage format version metadata.

## Milestone 2: Embedded Safety

Embedded mode should be safe and unsurprising.

- Add database and collection file locks.
- Define supported concurrency model: single process writer, multi-thread safe.
- Add read-only open mode for safe multi-process readers.
- Add explicit `flush`, `close`, and `checkpoint` APIs.
- Improve errors when another writer owns the data directory.
- Keep LocalClient and HTTPClient behavior aligned through shared tests.

## Milestone 3: Backup And Restore

Single-node deployment needs operational escape hatches.

- Add database and collection snapshots.
- Add restore from snapshot.
- Support consistent snapshot while reads continue.
- Define write blocking or LSN-based snapshot isolation.
- Add import/export for JSONL metadata plus binary vectors.
- Add migration hooks for future storage format upgrades.

## Milestone 4: Server Mode

The standalone service should feel like a real database process.

- Add CLI: `lynsedb serve --data-dir ./data --host 0.0.0.0 --port 7637`.
- Add config file support with environment variable overrides.
- Add `/healthz`, `/readyz`, and `/metrics`.
- Add graceful shutdown with WAL flush and checkpoint.
- Add request limits, batch limits, and timeout configuration.
- Generate OpenAPI documentation from the HTTP API.
- Publish Docker Compose, systemd, and Kubernetes examples.

## Milestone 5: Query Capabilities

After the storage core is trustworthy, broaden retrieval quality.

- Add named vector fields per record.
- Allow each vector field to have its own dimension, metric, and index.
- Add sparse vector storage and sparse inner-product search.
- Add BM25 or inverted-index text retrieval.
- Add hybrid search with RRF and weighted fusion.
- Add rerank hooks for cross-encoders and LLM rerankers.
- Expand metadata indexes: range, bitmap, keyword, datetime, and arrays.
- Add filter explain/profile output.

## Milestone 6: Observability And Governance

Production users need to understand and control resource usage.

- Add structured logs.
- Add Prometheus metrics for latency, QPS, WAL size, memory, disk, and index
  build progress.
- Add query profiling: filter matches, scanned vectors, index path, and rerank
  cost.
- Add collection-level limits for `top_k`, batch size, vector count, and memory.
- Add slow-query warnings.
- Add audit log for server mode.

## Milestone 7: Ecosystem

LynseDB should fit into common AI application stacks.

- Add LangChain integration.
- Add LlamaIndex integration.
- Add Haystack integration.
- Add examples for OpenAI embeddings, sentence-transformers, and FastEmbed.
- Add import tools for NumPy, Parquet, FAISS, Chroma, and Qdrant-style exports.
- Publish reproducible benchmarks for recall, latency, memory, disk usage, and
  index build time.

## Milestone 8: Distributed Features Later

Distributed work should wait until single-node semantics are stable.

- Start with snapshot shipping and read replicas.
- Add collection-level or partition-key sharding.
- Add replication only after backup, restore, WAL, and manifest semantics are
  stable.
- Add coordinator and rolling-upgrade support only when there is a clear
  operational story.

## Immediate Coding Queue

- [x] Fix WAL record counting for batches without metadata.
- [x] Extend WAL segments to persist user IDs.
- [x] Make collection recovery use persisted IDs and skip already-applied WAL rows.
- [x] Add insert duplicate-ID validation.
- [x] Add tests for reopen after uncommitted vector-only and custom-ID batches.
- [x] Add tests documenting duplicate insert behavior.
- [x] Add explicit `upsert_items` after insert behavior is stable.
- [x] Add reopen tests for upserted vectors, IDs, fields, and tombstones.
- [x] Extend crash/reopen coverage to persisted index files.
- [x] Ensure index metadata and index files are atomically swapped.
- [x] Define and persist storage format version metadata.

## Milestone 2 Immediate Coding Queue

- [x] Add a collection-level writer lock for embedded mode.
- [x] Add explicit `flush`, `checkpoint`, and `close` APIs.
- [x] Expose `flush`, `checkpoint`, and `close` through Python and HTTP clients.
- [x] Add regression tests for writer-lock rejection and checkpoint recovery.
- [x] Add database-level and manager-level writer locks.
- [ ] Add read-only open mode for safe multi-process readers.
- [ ] Add graceful server shutdown that checkpoints open collections.
