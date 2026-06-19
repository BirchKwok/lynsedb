#!/usr/bin/env python3
"""Coverage-oriented LynseDB public API and index matrix benchmark.

This benchmark is intentionally broader than scripts/bench_lynse_query.py:

* exercises local public APIs from VectorDBClient, LocalClient, and LocalCollection;
* measures common read/query/search APIs repeatedly on a stable collection;
* builds and searches every documented index family and quantized flat mode;
* records unsupported/error cases without aborting the whole run.

Environment knobs:
  LYNSE_MATRIX_DIR=/tmp/lynse_public_api_matrix
  LYNSE_MATRIX_API_N=20000
  LYNSE_MATRIX_INDEX_N=12000
  LYNSE_MATRIX_DIM=64
  LYNSE_MATRIX_LOOPS=5
  LYNSE_MATRIX_WARMUP=2
  LYNSE_MATRIX_INDEX_MODES=FLAT-IP,IVF-IP,...
  LYNSE_MATRIX_INCLUDE_BINARY=1
  LYNSE_MATRIX_OUTPUT=/tmp/lynse_public_api_matrix/results.jsonl
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np

import lynse


ROOT = Path(os.environ.get("LYNSE_MATRIX_DIR", "/tmp/lynse_public_api_matrix"))
API_N = int(os.environ.get("LYNSE_MATRIX_API_N", "20000"))
INDEX_N = int(os.environ.get("LYNSE_MATRIX_INDEX_N", "12000"))
BINARY_N = int(os.environ.get("LYNSE_MATRIX_BINARY_N", str(min(INDEX_N, 8000))))
DIM = int(os.environ.get("LYNSE_MATRIX_DIM", "64"))
K = int(os.environ.get("LYNSE_MATRIX_K", "10"))
LOOPS = int(os.environ.get("LYNSE_MATRIX_LOOPS", "5"))
WARMUP = int(os.environ.get("LYNSE_MATRIX_WARMUP", "2"))
BUILD_CHUNK = int(os.environ.get("LYNSE_MATRIX_BUILD_CHUNK", "5000"))
BATCH_QUERIES = int(os.environ.get("LYNSE_MATRIX_BATCH_QUERIES", "4"))
NPROBE = int(os.environ.get("LYNSE_MATRIX_NPROBE", "10"))
FEATURE_N = int(os.environ.get("LYNSE_MATRIX_FEATURE_N", str(min(API_N, 5000))))
SEED = int(os.environ.get("LYNSE_MATRIX_SEED", "20260619"))
INCLUDE_BINARY = os.environ.get("LYNSE_MATRIX_INCLUDE_BINARY", "1") != "0"
OUTPUT = Path(os.environ.get("LYNSE_MATRIX_OUTPUT", str(ROOT / "results.jsonl")))


FLOAT_INDEX_MODES = [
    # Flat exact and SQ8.
    "FLAT-IP",
    "FLAT-L2",
    "FLAT-COS",
    "FLAT-IP-SQ8",
    "FLAT-L2-SQ8",
    "FLAT-COS-SQ8",
    # Flat auxiliary quantizers.
    "FLAT-IP-PQ",
    "FLAT-L2-PQ",
    "FLAT-COS-PQ",
    "FLAT-IP-PQ8",
    "FLAT-IP-PQ16",
    "FLAT-L2-PQ8",
    "FLAT-COS-PQ8",
    "FLAT-IP-RABITQ",
    "FLAT-L2-RABITQ",
    "FLAT-COS-RABITQ",
    "FLAT-IP-POLARVEC",
    "FLAT-L2-POLARVEC",
    "FLAT-COS-POLARVEC",
    "FLAT-IP-POLARVEC3",
    "FLAT-IP-POLARVEC4",
    "FLAT-IP-POLARVEC8",
    # Graph and partition families.
    "HNSW-IP",
    "HNSW-L2",
    "HNSW-COS",
    "HNSW-IP-SQ8",
    "HNSW-L2-SQ8",
    "HNSW-COS-SQ8",
    "DISKANN-IP",
    "DISKANN-L2",
    "DISKANN-COS",
    "DISKANN-IP-SQ8",
    "DISKANN-L2-SQ8",
    "DISKANN-COS-SQ8",
    "IVF-IP",
    "IVF-L2",
    "IVF-COS",
    "IVF-IP-SQ8",
    "IVF-L2-SQ8",
    "IVF-COS-SQ8",
    "SPANN-IP",
    "SPANN-L2",
    "SPANN-COS",
    "SPANN-IP-SQ8",
    "SPANN-L2-SQ8",
    "SPANN-COS-SQ8",
]

BINARY_INDEX_MODES = [
    "FLAT-JACCARD-BINARY",
    "FLAT-HAMMING-BINARY",
    "IVF-JACCARD-BINARY",
    "IVF-HAMMING-BINARY",
]


def configured_index_modes() -> list[str]:
    raw = os.environ.get("LYNSE_MATRIX_INDEX_MODES")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    modes = list(FLOAT_INDEX_MODES)
    if INCLUDE_BINARY:
        modes.extend(BINARY_INDEX_MODES)
    return modes


INDEX_MODES = configured_index_modes()
ROWS: list[dict[str, Any]] = []


def percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def result_size(value: Any) -> int | None:
    if hasattr(value, "ids") and value.ids is not None:
        return int(len(value.ids))
    if hasattr(value, "vectors") and value.vectors is not None:
        return int(value.vectors.shape[0])
    if isinstance(value, (list, tuple, dict, str, bytes)):
        return len(value)
    return None


def append_row(row: dict[str, Any]) -> None:
    ROWS.append(row)


def print_once(group: str, name: str, row: dict[str, Any]) -> None:
    status = row.get("status", "ok")
    elapsed = row.get("elapsed_ms")
    size = row.get("size")
    if status == "ok":
        size_part = "" if size is None else f" size={size}"
        print(f"{group:14s} {name:34s} {elapsed:10.3f} ms{size_part}", flush=True)
    else:
        print(f"{group:14s} {name:34s} ERROR {row.get('error')}", flush=True)


def timed_once(group: str, name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        value = fn()
        row = {
            "group": group,
            "name": name,
            "status": "ok",
            "elapsed_ms": (time.perf_counter() - t0) * 1e3,
            "size": result_size(value),
        }
    except Exception as exc:  # noqa: BLE001 - benchmark must keep going.
        row = {
            "group": group,
            "name": name,
            "status": "error",
            "elapsed_ms": (time.perf_counter() - t0) * 1e3,
            "error": f"{type(exc).__name__}: {exc}",
        }
    append_row(row)
    print_once(group, name, row)
    return row


def bench_many(
    group: str,
    name: str,
    fn: Callable[[], Any],
    *,
    loops: int = LOOPS,
    warmup: int = WARMUP,
) -> dict[str, Any]:
    try:
        for _ in range(warmup):
            fn()
        samples = []
        size = None
        for _ in range(loops):
            t0 = time.perf_counter()
            value = fn()
            samples.append((time.perf_counter() - t0) * 1e3)
            if size is None:
                size = result_size(value)
        row = {
            "group": group,
            "name": name,
            "status": "ok",
            "p50_ms": percentile(samples, 50),
            "p90_ms": percentile(samples, 90),
            "p99_ms": percentile(samples, 99),
            "mean_ms": mean(samples),
            "loops": loops,
            "size": size,
        }
        size_part = "" if size is None else f" size={size}"
        print(
            f"{group:14s} {name:34s} "
            f"p50={row['p50_ms']:9.3f} ms "
            f"p90={row['p90_ms']:9.3f} ms "
            f"mean={row['mean_ms']:9.3f} ms{size_part}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark must keep going.
        row = {
            "group": group,
            "name": name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"{group:14s} {name:34s} ERROR {row['error']}", flush=True)
    append_row(row)
    return row


def write_results() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in ROWS:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"\nWrote {len(ROWS)} rows to {OUTPUT}", flush=True)


def make_vectors(start: int, size: int, dim: int, *, binary: bool = False) -> np.ndarray:
    rng = np.random.default_rng(SEED + start + (10_000 if binary else 0))
    if binary:
        return rng.integers(0, 2, size=(size, dim), dtype=np.int8).astype(np.float32)
    return rng.random((size, dim), dtype=np.float32)


def make_fields(start: int, size: int) -> list[dict[str, Any]]:
    fields = []
    for row_id in range(start, start + size):
        bucket = row_id % 1000
        tenant = row_id % 10
        topic = row_id % 128
        fields.append(
            {
                "order": row_id,
                "bucket": bucket,
                "tenant": tenant,
                "category": f"cat{row_id % 32}",
                "title": f"topic{topic}",
                "body": f"topic{topic} tenant{tenant} bucket{bucket}",
                "active": row_id % 2 == 0,
                "score": float(bucket) / 1000.0,
            }
        )
    return fields


def make_sparse_vectors(start: int, size: int) -> list[dict[int, float]]:
    vectors = []
    for row_id in range(start, start + size):
        a = row_id % 1024
        b = (row_id * 7 + 13) % 1024
        c = (row_id * 17 + 29) % 1024
        vectors.append({a: 1.0, b: 0.5, c: 0.25})
    return vectors


def create_collection(
    root: Path,
    db_name: str,
    coll_name: str,
    *,
    n: int,
    dim: int,
    binary: bool = False,
):
    client = lynse.VectorDBClient(str(root))
    db = client.create_database(db_name, drop_if_exists=True)
    coll = db.require_collection(coll_name, dim=dim, drop_if_exists=True, default_index=None)

    inserted = 0
    while inserted < n:
        size = min(BUILD_CHUNK, n - inserted)
        ids = list(range(inserted, inserted + size))
        coll.add(
            ids=ids,
            vectors=make_vectors(inserted, size, dim, binary=binary),
            fields=make_fields(inserted, size),
            batch_size=size,
        )
        inserted += size
    coll.commit()
    return client, db, coll


def prepare_root() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)


def prepare_api_collection():
    client, db, coll = create_collection(
        ROOT / "api",
        "api_db",
        "api_vectors",
        n=API_N,
        dim=DIM,
    )
    coll.build_index("FLAT-IP")

    named_n = min(FEATURE_N, API_N)
    coll.create_vector_field("image", DIM, metric="ip", index_mode="FLAT-IP")
    coll.add_named_vectors(
        "image",
        make_vectors(100_000, named_n, DIM),
        list(range(named_n)),
    )
    coll.add_sparse_vectors(make_sparse_vectors(0, named_n), list(range(named_n)))
    coll.commit()
    return client, db, coll


def run_management_api_bench(client, db) -> None:
    print("\nManagement/public API smoke timings:", flush=True)
    timed_once("client", "list_databases", client.list_databases)
    timed_once("client", "get_database", lambda: client.get_database("api_db"))
    timed_once(
        "client",
        "create_collection",
        lambda: client.create_collection("api_db", "client_temp", dim=DIM, drop_if_exists=True),
    )
    timed_once("client", "snapshot_database", lambda: client.snapshot_database("api_db", ROOT / "api_db_snapshot"))
    timed_once(
        "client",
        "restore_database",
        lambda: client.restore_database("api_db_restored", ROOT / "api_db_snapshot", overwrite=True),
    )
    timed_once("client", "drop_database", lambda: client.drop_database("api_db_restored"))

    timed_once("database", "database_exists", db.database_exists)
    timed_once("database", "show_collections", db.show_collections)
    timed_once("database", "show_collections_details", db.show_collections_details)
    timed_once("database", "get_collection", lambda: db.get_collection("api_vectors", warm_up=False))
    timed_once("database", "require_collection existing", lambda: db.require_collection("api_vectors", dim=DIM))
    timed_once(
        "database",
        "update_collection_description",
        lambda: db.update_collection_description("api_vectors", "matrix benchmark"),
    )
    timed_once(
        "database",
        "snapshot_collection",
        lambda: db.snapshot_collection("api_vectors", ROOT / "api_collection_snapshot"),
    )
    timed_once(
        "database",
        "restore_collection",
        lambda: db.restore_collection("api_vectors_copy", ROOT / "api_collection_snapshot", overwrite=True),
    )
    timed_once(
        "database",
        "export_collection",
        lambda: db.export_collection("api_vectors", ROOT / "api_collection_export"),
    )
    timed_once(
        "database",
        "import_collection",
        lambda: db.import_collection("api_vectors_imported", ROOT / "api_collection_export", overwrite=True),
    )
    timed_once("database", "drop_collection copy", lambda: db.drop_collection("api_vectors_copy"))
    timed_once("database", "drop_collection imported", lambda: db.drop_collection("api_vectors_imported"))
    timed_once("database", "drop_collection temp", lambda: db.drop_collection("client_temp"))


def run_write_api_bench(db) -> None:
    print("\nWrite/index lifecycle API timings:", flush=True)
    coll = db.require_collection("write_api", dim=DIM, drop_if_exists=True, default_index=None)
    base = make_vectors(200_000, 256, DIM)
    fields = make_fields(200_000, 256)
    ids = list(range(200_000, 200_256))

    timed_once("write", "add explicit ids", lambda: coll.add(ids=ids, vectors=base, fields=fields, batch_size=256))
    timed_once("write", "add generated ids", lambda: coll.add(vectors=make_vectors(201_000, 64, DIM), batch_size=64))
    timed_once(
        "write",
        "upsert existing",
        lambda: coll.upsert(ids[:64], vectors=make_vectors(202_000, 64, DIM), fields=make_fields(202_000, 64)),
    )
    timed_once(
        "write",
        "upsert new",
        lambda: coll.upsert(list(range(210_000, 210_064)), vectors=make_vectors(203_000, 64, DIM)),
    )

    def session_add():
        with coll.insert_session() as session:
            session.add(
                list(range(220_000, 220_064)),
                vectors=make_vectors(204_000, 64, DIM),
                fields=make_fields(204_000, 64),
                batch_size=64,
            )

    timed_once("write", "insert_session add", session_add)
    timed_once("write", "flush", coll.flush)
    timed_once("write", "checkpoint", coll.checkpoint)
    timed_once("write", "build_index FLAT-IP", lambda: coll.build_index("FLAT-IP"))
    timed_once("write", "remove_index", coll.remove_index)
    timed_once("write", "create_vector_field", lambda: coll.create_vector_field("image", DIM, metric="ip"))
    timed_once(
        "write",
        "add_named_vectors",
        lambda: coll.add_named_vectors("image", make_vectors(205_000, 64, DIM), ids[:64]),
    )
    timed_once("write", "add_sparse_vectors", lambda: coll.add_sparse_vectors(make_sparse_vectors(0, 64), ids[:64]))
    timed_once("write", "delete", lambda: coll.delete(ids[:32]))
    timed_once("write", "list_deleted_ids", coll.list_deleted_ids)
    timed_once("write", "restore", lambda: coll.restore(ids[:32]))
    timed_once("write", "compact", coll.compact)
    timed_once("write", "snapshot_to", lambda: coll.snapshot_to(ROOT / "write_collection_snapshot"))
    timed_once("write", "export_to", lambda: coll.export_to(ROOT / "write_collection_export"))
    timed_once("write", "commit", coll.commit)
    timed_once("write", "close", coll.close)
    timed_once("database", "drop_collection write_api", lambda: db.drop_collection("write_api"))


def simple_reranker(payload: dict[str, Any]) -> dict[str, list[Any]]:
    ids = list(payload["items"][::-1])
    return {"ids": [item["id"] for item in ids]}


def run_read_api_bench(coll) -> None:
    print("\nRead/query/search public API timings:", flush=True)
    rng = np.random.default_rng(SEED + 777)
    query = rng.random(DIM, dtype=np.float32)
    batch = rng.random((BATCH_QUERIES, DIM), dtype=np.float32)
    named_query = rng.random(DIM, dtype=np.float32)
    sparse_query = {0: 1.0, 13: 0.5, 29: 0.25}
    exact_order = min(1234, API_N - 1)
    filter_ids = list(range(min(512, API_N)))

    filters = {
        "exact": f'"order" = {exact_order}',
        "1pct": '"bucket" < 10',
        "10pct": '"bucket" < 100',
        "string": '"category" = \'cat7\'',
    }

    bench_many("read", "shape property", lambda: coll.shape, loops=LOOPS * 5)
    bench_many("read", "vector_dtype property", lambda: coll.vector_dtype, loops=LOOPS * 5)
    bench_many("read", "exists", coll.exists, loops=LOOPS * 5)
    bench_many("read", "is_id_exists", lambda: coll.is_id_exists(exact_order), loops=LOOPS * 5)
    bench_many("read", "max_id property", lambda: coll.max_id, loops=LOOPS * 5)
    bench_many("read", "stats", coll.stats)
    bench_many("read", "list_fields", coll.list_fields)
    bench_many("read", "list_vector_fields", coll.list_vector_fields)
    bench_many("read", "head", lambda: coll.head(10))
    bench_many("read", "tail", lambda: coll.tail(10))

    bench_many("query", "where exact ids-only", lambda: coll.query(filters["exact"], return_ids_only=True))
    bench_many("query", "where 1pct ids-only", lambda: coll.query(filters["1pct"], return_ids_only=True))
    bench_many("query", "where 10pct ids-only", lambda: coll.query(filters["10pct"], return_ids_only=True))
    bench_many("query", "where string fields", lambda: coll.query(filters["string"], return_ids_only=False))
    bench_many("query", "filter_ids ids-only", lambda: coll.query(filter_ids=filter_ids, return_ids_only=True))
    bench_many("query", "query_vectors 1pct", lambda: coll.query_vectors(filters["1pct"]))
    bench_many("query", "query_vectors filter_ids", lambda: coll.query_vectors(filter_ids=filter_ids))

    bench_many("search", "vector", lambda: coll.search(query, k=K))
    bench_many("search", "vector return_fields", lambda: coll.search(query, k=K, return_fields=True))
    bench_many("search", "vector filter 10pct", lambda: coll.search(query, k=K, where=filters["10pct"]))
    bench_many("search", "vector approx", lambda: coll.search(query, k=K, approx=True, eps=1e-4))
    bench_many("search", "named vector field", lambda: coll.search(named_query, k=K, vector_field="image"))
    bench_many("search", "profile filter 10pct", lambda: coll.search_profile(query, k=K, where=filters["10pct"]))
    bench_many("search", "sparse", lambda: coll.search_sparse(sparse_query, k=K))
    bench_many("search", "sparse filter 10pct", lambda: coll.search_sparse(sparse_query, k=K, where=filters["10pct"]))
    bench_many("search", "bm25 topic42", lambda: coll.bm25_search("topic42", k=K))
    bench_many("search", "bm25 topic42 filter", lambda: coll.bm25_search("topic42", k=K, where=filters["10pct"]))
    bench_many(
        "search",
        "hybrid topic42",
        lambda: coll.hybrid_search(query, text="topic42", k=K, candidate_limit=100),
    )
    bench_many(
        "search",
        "hybrid topic42 filter",
        lambda: coll.hybrid_search(query, text="topic42", k=K, where=filters["10pct"], candidate_limit=100),
    )
    bench_many("search", "batch_search", lambda: coll.batch_search(batch, k=K), loops=max(2, LOOPS))
    bench_many(
        "search",
        "batch_search filter",
        lambda: coll.batch_search(batch, k=K, where=filters["10pct"]),
        loops=max(2, LOOPS),
    )
    bench_many("search", "search_range", lambda: coll.search_range(query, threshold=0.0, max_results=K))
    bench_many(
        "search",
        "reranker",
        lambda: coll.search(query, k=K, return_fields=True, reranker=simple_reranker),
    )
    bench_many("collection", "update_description", lambda: coll.update_description("read bench"))


def metric_for_mode(mode: str) -> str:
    upper = mode.upper()
    if "JACCARD" in upper:
        return "JACCARD"
    if "HAMMING" in upper:
        return "HAMMING"
    if "L2" in upper:
        return "L2"
    if "COS" in upper:
        return "COS"
    return "IP"


def exact_mode_for_metric(metric: str) -> str:
    return {
        "IP": "FLAT-IP",
        "L2": "FLAT-L2",
        "COS": "FLAT-COS",
        "JACCARD": "FLAT-JACCARD-BINARY",
        "HAMMING": "FLAT-HAMMING-BINARY",
    }[metric]


def n_clusters_for(n: int) -> int:
    return max(4, min(256, int(np.sqrt(max(1, n)))))


def recall_at_k(got: list[Any], expected: list[Any]) -> float:
    if not expected:
        return 0.0
    return len(set(got[:K]).intersection(expected[:K])) / float(min(K, len(expected)))


def build_exact_refs(coll, queries: np.ndarray, metrics: set[str]) -> dict[str, list[list[Any]]]:
    refs = {}
    for metric in sorted(metrics):
        mode = exact_mode_for_metric(metric)
        coll.build_index(mode)
        refs[metric] = [coll.search(query, k=K).ids.tolist() for query in queries]
    return refs


def run_one_index_mode(coll, mode: str, queries: np.ndarray, refs: dict[str, list[list[Any]]], *, n: int) -> None:
    metric = metric_for_mode(mode)
    where = '"bucket" < 100'
    batch_queries = queries[: min(BATCH_QUERIES, len(queries))]
    row: dict[str, Any] = {
        "group": "index_matrix",
        "name": mode,
        "mode": mode,
        "metric": metric,
        "n": n,
        "dim": DIM,
    }

    try:
        build_args = {"n_clusters": n_clusters_for(n)} if mode.upper().startswith(("IVF", "SPANN")) else {}
        t0 = time.perf_counter()
        coll.build_index(mode, **build_args)
        row["build_ms"] = (time.perf_counter() - t0) * 1e3

        for _ in range(WARMUP):
            coll.search(queries[0], k=K, nprobe=NPROBE)

        search_samples = []
        filter_samples = []
        batch_samples = []
        recalls = []
        filter_bad_ids = 0
        for i in range(LOOPS):
            query = queries[i % len(queries)]
            t0 = time.perf_counter()
            result = coll.search(query, k=K, nprobe=NPROBE)
            search_samples.append((time.perf_counter() - t0) * 1e3)
            recalls.append(recall_at_k(result.ids.tolist(), refs[metric][i % len(refs[metric])]))

            t0 = time.perf_counter()
            filtered = coll.search(query, k=K, where=where, nprobe=NPROBE)
            filter_samples.append((time.perf_counter() - t0) * 1e3)
            filter_bad_ids += sum(1 for item_id in filtered.ids.tolist() if int(item_id) % 1000 >= 100)

            t0 = time.perf_counter()
            coll.batch_search(batch_queries, k=K, nprobe=NPROBE)
            batch_samples.append((time.perf_counter() - t0) * 1e3)

        row.update(
            {
                "status": "ok",
                "search_p50_ms": percentile(search_samples, 50),
                "search_p90_ms": percentile(search_samples, 90),
                "filter_p50_ms": percentile(filter_samples, 50),
                "batch_p50_ms": percentile(batch_samples, 50),
                "recall_at_k": mean(recalls),
                "filter_bad_ids": filter_bad_ids,
                "loops": LOOPS,
            }
        )
        print(
            f"{'index_matrix':14s} {mode:34s} "
            f"build={row['build_ms']:9.3f} ms "
            f"search={row['search_p50_ms']:9.3f} ms "
            f"filter={row['filter_p50_ms']:9.3f} ms "
            f"batch={row['batch_p50_ms']:9.3f} ms "
            f"recall={row['recall_at_k']:.3f} "
            f"bad_filter={filter_bad_ids}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - matrix should expose every failure.
        row.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"{'index_matrix':14s} {mode:34s} ERROR {row['error']}", flush=True)

    append_row(row)


def run_index_matrix() -> None:
    print("\nIndex and quantization matrix:", flush=True)
    float_modes = [mode for mode in INDEX_MODES if "BINARY" not in mode.upper()]
    binary_modes = [mode for mode in INDEX_MODES if "BINARY" in mode.upper()]

    if float_modes:
        _, _, coll = create_collection(
            ROOT / "index_float",
            "index_db",
            "float_vectors",
            n=INDEX_N,
            dim=DIM,
        )
        rng = np.random.default_rng(SEED + 999)
        queries = rng.random((max(LOOPS, BATCH_QUERIES, 4), DIM), dtype=np.float32)
        refs = build_exact_refs(coll, queries, {metric_for_mode(mode) for mode in float_modes})
        for mode in float_modes:
            run_one_index_mode(coll, mode, queries, refs, n=INDEX_N)

    if INCLUDE_BINARY and binary_modes:
        _, _, coll = create_collection(
            ROOT / "index_binary",
            "index_db",
            "binary_vectors",
            n=BINARY_N,
            dim=DIM,
            binary=True,
        )
        queries = np.vstack(
            [
                make_vectors(300_000 + i, 1, DIM, binary=True)[0]
                for i in range(max(LOOPS, BATCH_QUERIES, 4))
            ]
        ).astype(np.float32)
        refs = build_exact_refs(coll, queries, {metric_for_mode(mode) for mode in binary_modes})
        for mode in binary_modes:
            run_one_index_mode(coll, mode, queries, refs, n=BINARY_N)


def main() -> None:
    prepare_root()
    print("=" * 120, flush=True)
    print(
        f"LynseDB public API + index matrix: api_n={API_N:,}, index_n={INDEX_N:,}, "
        f"binary_n={BINARY_N:,}, dim={DIM}, loops={LOOPS}, modes={len(INDEX_MODES)}",
        flush=True,
    )
    print("=" * 120, flush=True)

    client, db, coll = prepare_api_collection()
    run_management_api_bench(client, db)
    run_write_api_bench(db)
    run_read_api_bench(coll)
    run_index_matrix()
    write_results()


if __name__ == "__main__":
    main()
