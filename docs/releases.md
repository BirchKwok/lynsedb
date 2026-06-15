# Release Notes

This page documents the major features and improvements in each version of LynseDB. Only versions with the `v` prefix are official releases.

## v0.4.0

**Major Release - Cluster Coordination, Transport, and API Consistency**

**Key Features:**
- 🌐 **Lightweight Cluster Mode**: Added a coordinator role for sharding remote HTTP deployments across multiple shard groups. The coordinator owns collection metadata, routes IDs with stable hash buckets, fans out searches, merges results, and can mirror writes to active replicas.
- 🔁 **Shard Failover Foundations**: Added coordinator state tracking, health checks, replica states, primary promotion behavior, and `/cluster_info` diagnostics for cluster operations.
- ⚡ **Internal Cluster RPC**: Added a private length-prefixed TCP RPC path between coordinator and shard nodes for hot shard operations including ping, search, batch search, binary add, upsert, delete, and restore, with HTTP fallback when RPC is unavailable.
- 📦 **Compact Binary Transport Improvements**: Extended binary client/server paths for high-throughput writes and searches, including float16 wire encoding support for remote calls.
- 🧭 **Consistent Python API Signatures**: Aligned local and HTTP collection method signatures, removed public `**kwargs` surfaces, and made parameter behavior explicit across local and remote clients.
- 🔧 **IVFIndex Refactor**: Moved IVF training and assignment to a shared KMeans implementation used by both in-memory IVF and mmap-backed IVF storage.
- 🧪 **Index Parameter Handling**: `n_clusters` is now accepted by the Python API for all index modes but is only sent to or used by IVF indexes. Non-IVF indexes ignore it for easier generic tuning code.
- 🌐 **Local/HTTP Parity for `wire_dtype`**: Local collection methods now accept `wire_dtype` for API compatibility while continuing to use the direct numpy float32 path locally.
- 📚 **Cluster Deployment Documentation**: Added a full cluster deployment and maintenance guide covering local clusters, production layout, coordinator state, health checks, promotion, recovery, backup, and upgrade notes.

**Improvements:**
- Refined Python binding structure around records, embeddings, reranking, result views, sessions, and HTTP client behavior.
- Added server CLI support for coordinator mode, cluster config/state paths, shard API keys, and cluster health tuning.
- Expanded index documentation with canonical names, aliases, metric variants, binary metric variants, IVF tuning guidance, and troubleshooting notes.
- Updated quickstart, client reference, HTTP API docs, tutorials, performance tuning, and operations documentation to reflect the current API surface.
- Added transport benchmarking utilities for measuring client/server and cluster paths.
- Improved CI release workflow with tag resolution and version checking steps.

**Testing:**
- Added standard cluster tests covering state initialization, ID routing, replica state changes, binary item preparation, and coordinator behavior.
- Added explicit API parameter tests to keep local and HTTP collection signatures aligned and verify `n_clusters` and `wire_dtype` behavior.
- Expanded collection, search, metadata index, Docker API, and server CLI coverage for the v0.4.0 API surface.

**Compatibility Notes:**
- No required data migration is expected from v0.3.0 for single-node local or HTTP deployments.
- Existing single-node `VectorDBClient` usage continues to work. Cluster mode is opt-in via `lynse serve --role coordinator`.
- Applications that passed extra unsupported keyword arguments to public Python methods should switch to the documented explicit parameters.

---

## v0.3.0

**Major Release - Float16 Support and Advanced Indexing**

**Key Features:**
- ✨ **Float16 Vector Storage**: Added native `float16`/`f16` storage in `FlatMmap` and `VectorStore`, reducing dense vector storage size by roughly half for collections that can tolerate half-precision storage.
- 📊 **Metric-Aware PolarVec**: Enhanced PolarVec auxiliary storage with metric-aware data so quantized search can better match the active distance metric.
- 📚 **Documentation Refresh**: Reworked README positioning, quickstart material, tutorials, deployment docs, production roadmap, and API references for the Rust-backed API introduced in v0.2.x.
- 🧰 **Documentation Release Workflow**: Improved docs CI with explicit version input, concurrency control, and existing docs-branch fetching for versioned documentation publishing.

**Improvements:**
- Updated package versions to `0.3.0` across Cargo, Python package metadata, and runtime `__version__`.
- Clarified embedded mode versus HTTP service mode in the user-facing documentation.
- Expanded tutorials for adding vectors, searching and filtering, indexing, named/sparse/hybrid retrieval, operations, and production tuning.
- Removed outdated README badges and aligned docs with the current supported platform policy.

**Compatibility Notes:**
- Existing `float32` collections remain the default and require no migration.
- New half-precision collections should be created explicitly with `dtypes="float16"`.
- Consider rebuilding PolarVec indexes after upgrading if you want the metric-aware auxiliary data generated by this release.

---

## v0.2.0

**Major Release - Rust Architecture and Advanced Features**

**Key Features:**
- 🏗️ **Rust-First Architecture**: Rebuilt LynseDB around a Rust storage/search backend exposed through PyO3, with a simplified Python API for embedded and remote usage.
- 🧭 **Unified `VectorDBClient` Entry Point**: Introduced one high-level client that selects local embedded mode for filesystem paths and HTTP mode for remote URLs.
- 🚀 **Optimized Rust HNSW Path**: Added a Rust core with optimized HNSW search and removed the Python fallback execution paths.
- 📦 **Mmap-Backed Storage**: Added memory-mapped vector storage, custom binary storage protocols, append-friendly vector files, and WAL-backed durability.
- 🗂️ **Index Families**: Added Flat, HNSW, IVF, DiskANN, PQ, RaBitQ, and PolarVec index families, including IVF_FLAT storage and two-pass quantized search paths.
- 🔎 **Approximate Search**: Added configurable approximate search with `approx` and `eps` for supported flat metrics.
- 🧮 **Metadata, Text, and Hybrid Retrieval**: Enhanced SQL-like filtering, numeric equality and `IN` filters, inverted text indexing, BM25, sparse vector search, and hybrid retrieval.
- 🧩 **Named Vector Fields**: Added support for multiple vector fields per record, enabling multimodal records and field-specific indexes.
- 🛠️ **Operations Surface**: Added snapshots, restore, export/import, compaction, HTTP health/readiness/metrics/OpenAPI surfaces, API-key auth, Docker, systemd, and Kubernetes examples.
- 🪵 **Observability Hooks**: Added slow query alerts and audit logging capabilities.
- 🧪 **ResultView API**: Added structured result objects with NumPy arrays and conversion helpers for downstream processing.

**Improvements:**
- Improved context management and resource cleanup in `VectorDBClient`, database managers, vector storage, and field storage.
- Added in-memory field indexes and apex ID mapping for faster metadata-filtered query paths.
- Optimized filtered search with a dual strategy that can choose between prefiltering and vector-first execution.
- Updated dependencies, internalized assertion helpers, and reorganized the package layout under the Rust-backed `python/lynse` tree.
- Dropped native Windows support. Windows users should run LynseDB through WSL 2 or Docker.
- Added benchmark scripts for flat search, filtered search, IVF/KMeans, approximate search, and client query paths.

**Testing:**
- Added comprehensive tests for the Rust backend, collection operations, database operations, metadata indexes, search, result views, Docker API behavior, and server CLI behavior.
- Updated CI to run on version tags and refreshed Python versions used by workflows.

**Compatibility Notes:**
- This is the largest compatibility boundary in the project history. Data and code built for the pure Python v0.1.x implementation should be validated before production migration.
- Native Windows installs are no longer supported from this release forward.
- HNSW remains available, but storage layout, server behavior, and many implementation details changed substantially.

---

## v0.1.6

**Final Stability Release - Pure Python**

**Key Features:**
- 🐛 **Trie Query Fix**: Fixed `AttributeError: 'Trie' object has no attribute '_search_single'` in the pure Python metadata/indexing path.
- 📦 **Final v0.1.x Package Version**: Marked the final pure Python release before the Rust-backed v0.2.0 architecture.

**Improvements:**
- Kept the lightweight pure Python client/server-optional behavior stable for users not ready to move to v0.2.x.
- Preserved the existing Python 3.9+ install and Docker-oriented documentation flow.

**Compatibility Notes:**
- Recommended maintenance release for users staying on the pure Python v0.1 line.

---

## v0.1.5

**Maintenance Release - Pure Python**

**Key Features:**
- 🔧 **Workflow Improvements**: Iterated on documentation and release workflows, including versioned docs publishing experiments with `mike`.
- 🧱 **Chunk Size Guardrails**: Added range limits around chunk sizing to reduce invalid storage configuration behavior.
- 📝 **API Naming Cleanup**: Renamed a function parameter for clearer API usage.

**Improvements:**
- Updated documentation content and site publishing configuration.
- Continued cleanup of CI workflows around docs and package release automation.

---

## v0.1.4

**Performance Release - Pure Python**

**Key Features:**
- 🛡️ **Safer MMAP Reading**: Continued hardening of memory-mapped file reading to avoid unstable reads and improve cross-platform behavior.

**Improvements:**
- Follow-up refinements to the safer mmap strategy introduced in v0.1.3.

---

## v0.1.3

**Bug Fix Release - Pure Python**

**Key Features:**
- 🛡️ **Enhanced MMAP Reading**: Replaced the original `.npy` mmap reading path with a safer strategy for vector file access.
- 📄 **Documentation CI Setup**: Added and refined docs publishing workflow pieces, MkDocs configuration, and site assets.
- 📦 **Version File**: Added a `VERSION` file and release workflow support around versioned package/docs builds.

**Improvements:**
- Updated dependencies in `requirements.txt`.
- Performed code style cleanup and removed unnecessary code.
- Iterated heavily on release and documentation workflows to support future tagged releases.

---

## v0.1.2

**Compatibility Release - Pure Python**

**Key Features:**
- 🐛 **Warning Fixes**: Fixed a Python `DeprecationWarning` caused by an invalid escape sequence.
- 🪟 **Windows MMAP Follow-Up**: Continued testing and fixes around mmap-related Windows failures.

**Improvements:**
- Removed unnecessary code after the mmap compatibility pass.

---

## v0.1.1

**Stability Release - Pure Python**

**Key Features:**
- 🔒 **Resource Management**: Used context managers to prevent errors caused by unclosed mmap files.
- ⚙️ **Safer File Deletion**: Added logic to wait for handle release when file deletion hits `PermissionError`.
- 🔄 **Thread Safety**: Added thread locks to reduce race conditions in the pure Python storage path.
- 🪟 **Windows Validation**: Tested core functionality under Windows and fixed Windows-specific mmap/file-handle behavior.

**Improvements:**
- Cleaned up unnecessary code.
- Improved tests and CI workflow coverage for early pure Python behavior.

---

## v0.1.0

**Major Release - Pure Python Vector Database Foundation**

**Key Features:**
- 🐍 **Pure Python Core**: Established LynseDB as a lightweight vector database implemented in Python with local embedded usage and optional HTTP deployment.
- 🧭 **Unified Client Entry Point**: Added `VectorDBClient`, which routes to the local native API for filesystem paths and to the HTTP client for remote URLs.
- 🗄️ **Storage Layer**: Added file-backed vector storage, memory-mapped access, ID tracking, and persistence primitives.
- 🔍 **Metadata Filtering**: Added FieldExpression-style metadata filtering and field indexing for filtered vector search.
- 📐 **Index and Distance Options**: Included flat, scalar-quantized, binary, and IVF-oriented index code paths with L2, inner-product, cosine, Hamming, and Jaccard-style search support.
- 🧵 **Single-Process Native API**: Documented the native Python API as best suited for single-process use, with HTTP mode recommended for process-safe deployments.

**Improvements:**
- Built the initial Python-first API around simple database, collection, insert, and search workflows.
- Added early tests, Docker usage notes, and MkDocs documentation.
- Clarified early project limitations: backward compatibility was not yet guaranteed and million-scale or smaller workloads were recommended.

---

## v0.0.2

**Historical Tag - Same Code as v0.1.0**

The `v0.0.2` tag points to the same commit as `v0.1.0`. The package metadata in that commit reports version `0.1.0`, so no separate `0.0.2` wheel should be generated from this tag.

Use the `v0.1.0` release assets for this historical code state.

---

## v0.0.1

**Historical Tag - Same Code as v0.1.0**

The `v0.0.1` tag points to the same commit as `v0.1.0`. The package metadata in that commit reports version `0.1.0`, so no separate `0.0.1` wheel should be generated from this tag.

**Compatibility Notes:**
- Treat `v0.0.1` and `v0.0.2` as historical aliases for the first `0.1.0` code state.
- Use the `v0.1.0` release assets if you need installable artifacts for this commit.

---

## Migration Guide

### From v0.3.0 to v0.4.0
- Single-node local and HTTP deployments do not require a storage migration.
- Cluster mode is new and opt-in. Start ordinary LynseDB HTTP servers as shards, then run a coordinator with `lynse serve --role coordinator --cluster-config ... --cluster-state ...`.
- Keep the coordinator `cluster_state.json` on persistent storage and back it up together with all shard data directories.
- If shard nodes use `--api-key`, pass the same secret to the coordinator with `--shard-api-key`.
- `wire_dtype` is now accepted by local and HTTP write/search methods for signature parity. Local calls still use direct numpy arrays; remote calls can use compact wire encodings.
- `n_clusters` remains meaningful only for IVF indexes. Passing it to non-IVF indexes is allowed and ignored by the Python API.
- Public Python methods now use explicit signatures. Replace any calls that depended on arbitrary `**kwargs` forwarding with documented parameters.
- The IVFIndex refactor is internal; existing IVF index names and search behavior remain compatible.

### From v0.2.0 to v0.3.0
- Float16 vectors are now available; specify `dtypes="float16"` when creating collections
- PolarVec index is improved; consider rebuilding indices for better performance
- All existing APIs remain compatible

### From v0.1.x to v0.2.0
- **Major architectural change**: v0.2.0 migrates from the pure Python implementation to the Rust-backed storage and search engine.
- Validate existing v0.1.x data on a staging copy before upgrading production deployments.
- Review client code that uses low-level pure Python modules; the recommended entry point is now `lynse.VectorDBClient`.
- HNSW remains available, but storage layout, persistence behavior, and server behavior changed substantially.
- Native Windows installs are no longer supported. Use WSL 2 or Docker on Windows.
- Rebuild indexes after migration so they are created by the Rust-backed index implementations.

### Within v0.1.x
- v0.1.1 through v0.1.6 are pure Python maintenance releases.
- Upgrade within the v0.1 line for safer mmap handling, Windows file-handle fixes, chunk-size guardrails, workflow updates, and the Trie query fix.
- No intentional major API migration is documented within the v0.1.x line, but early releases did not guarantee strict backward compatibility.

### From v0.0.x to v0.1.0
- `v0.0.1`, `v0.0.2`, and `v0.1.0` currently point to the same commit.
- The installable package version in that commit is `0.1.0`; prefer the `v0.1.0` release when installing artifacts from this code state.

---

## Support

- For issues or questions about a specific version, please check the [Troubleshooting Guide](tutorials/troubleshooting.md)
- For API documentation, see the [Python Client Reference](client.md) and [HTTP API Documentation](http_api/serve_api/app.md)
- For upcoming features, see the [Production Roadmap](production_roadmap.md)
