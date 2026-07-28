#!/usr/bin/env python3
"""1M-scale dense/sparse index matrix for the local LynseDB performance gate.

Builds collections once per run directory, then measures every documented index
alias plus sparse / hybrid / BM25 / named-dense shared paths.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from gate_index_modes import (  # noqa: E402
    ALL_INDEX_MODES,
    collection_kind_for_mode,
    exact_mode_for_metric,
    metric_for_mode,
    n_clusters_for,
    recall_floor_for_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=int(os.environ.get("GATE_ROWS", "1000000")))
    parser.add_argument("--dim", type=int, default=int(os.environ.get("GATE_DIM", "128")))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("GATE_BATCH_SIZE", "100000")),
    )
    parser.add_argument("--k", type=int, default=int(os.environ.get("GATE_K", "10")))
    parser.add_argument("--warmups", type=int, default=int(os.environ.get("GATE_WARMUPS", "2")))
    parser.add_argument("--trials", type=int, default=int(os.environ.get("GATE_TRIALS", "5")))
    parser.add_argument(
        "--batch-queries",
        type=int,
        default=int(os.environ.get("GATE_BATCH_QUERIES", "8")),
    )
    parser.add_argument("--nprobe", type=int, default=int(os.environ.get("GATE_NPROBE", "32")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("GATE_SEED", "20260727")))
    parser.add_argument(
        "--sparse-nnz",
        type=int,
        default=int(os.environ.get("GATE_SPARSE_NNZ", "16")),
    )
    parser.add_argument(
        "--sparse-dims",
        type=int,
        default=int(os.environ.get("GATE_SPARSE_DIMS", "1024")),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("GATE_MATRIX_DIR", "/tmp/lynse_gate_matrix")),
    )
    parser.add_argument(
        "--reuse",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("GATE_MATRIX_REUSE", "0") == "1",
    )
    parser.add_argument(
        "--modes",
        default=os.environ.get("GATE_INDEX_MODES", ""),
        help="Comma-separated mode subset; default = full documented matrix",
    )
    parser.add_argument("--skip-shared", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--skip-upsert", action="store_true")
    parser.add_argument(
        "--skip-modes",
        action="store_true",
        help="skip the per-index-mode matrix (still runs upsert/shared/sparse unless skipped)",
    )
    parser.add_argument("--side", default=os.environ.get("GATE_SIDE", "self"))
    parser.add_argument("--git-ref", default=os.environ.get("GATE_GIT_REF", ""))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def median_ms(samples: list[float]) -> float:
    return float(statistics.median(samples)) if samples else 0.0


def timed_ms(fn) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000.0


def recall_at_k(got: list[Any], expected: list[Any], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(got[:k]).intersection(expected[:k])) / float(min(k, len(expected)))


def result_ids(result) -> list[int]:
    ids = result.ids
    if hasattr(ids, "tolist"):
        return [int(x) for x in ids.tolist()]
    return [int(x) for x in ids]


def make_fields(start: int, size: int) -> list[dict[str, Any]]:
    fields = []
    for row_id in range(start, start + size):
        bucket = row_id % 1000
        topic = row_id % 100
        tenant = row_id % 10
        fields.append(
            {
                "order": row_id,
                "bucket": bucket,
                "tag": "a" if row_id % 2 == 0 else "b",
                "category": f"cat{topic}",
                "title": f"topic{topic}",
                "body": f"topic{topic} tenant{tenant} bucket{bucket}",
            }
        )
    return fields


def make_dense(rng: np.random.Generator, size: int, dim: int) -> np.ndarray:
    return rng.random((size, dim), dtype=np.float32)


def make_binary(rng: np.random.Generator, size: int, dim: int) -> np.ndarray:
    return rng.integers(0, 2, size=(size, dim), dtype=np.int8).astype(np.float32)


def make_haversine(rng: np.random.Generator, size: int) -> np.ndarray:
    lon = rng.uniform(-180.0, 180.0, size=size).astype(np.float32)
    lat = rng.uniform(-90.0, 90.0, size=size).astype(np.float32)
    return np.stack([lon, lat], axis=1)


def make_distribution(rng: np.random.Generator, size: int, dim: int) -> np.ndarray:
    raw = rng.random((size, dim), dtype=np.float32)
    sums = raw.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    return raw / sums


def make_sparse_vectors(start: int, size: int, *, nnz: int, dims: int) -> list[dict[int, float]]:
    vectors = []
    for row_id in range(start, start + size):
        vec = {}
        for offset in range(nnz):
            idx = (row_id * 31 + offset * 17) % dims
            vec[idx] = 1.0 / float(offset + 1)
        vectors.append(vec)
    return vectors


def configured_modes(raw: str) -> list[str]:
    if raw.strip():
        return [item.strip().upper() for item in raw.split(",") if item.strip()]
    return list(ALL_INDEX_MODES)


def profile_marker(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "rows": args.rows,
        "dim": args.dim,
        "haversine_dim": 2,
        "batch_size": args.batch_size,
        "k": args.k,
        "warmups": args.warmups,
        "trials": args.trials,
        "batch_queries": args.batch_queries,
        "nprobe": args.nprobe,
        "n_clusters": n_clusters_for(args.rows),
        "seed": args.seed,
        "sparse_nnz": args.sparse_nnz,
        "sparse_dims": args.sparse_dims,
        "modes": configured_modes(args.modes),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
    }


def open_client(data_dir: Path):
    import lynse

    return lynse.VectorDBClient(str(data_dir))


def create_collection(client, name: str, *, n: int, dim: int, kind: str, args: argparse.Namespace):
    db = client.create_database("gate_db", drop_if_exists=False)
    coll = db.require_collection(name, dim=dim, drop_if_exists=True, default_index=None)
    rng = np.random.default_rng(args.seed + hash(kind) % 10_000)
    print(f"Writing {n:,} x {dim} ({kind}) into {name}...", flush=True)
    inserted = 0
    while inserted < n:
        size = min(args.batch_size, n - inserted)
        if kind == "dense":
            vectors = make_dense(rng, size, dim)
        elif kind == "binary":
            vectors = make_binary(rng, size, dim)
        elif kind == "haversine":
            vectors = make_haversine(rng, size)
        elif kind == "distribution":
            vectors = make_distribution(rng, size, dim)
        else:
            raise ValueError(kind)
        ids = list(range(inserted, inserted + size))
        coll.add(ids=ids, vectors=vectors, fields=make_fields(inserted, size), batch_size=size)
        inserted += size
        if inserted % max(args.batch_size, 1) == 0 or inserted == n:
            print(f"  {name}: {inserted:,}/{n:,}", flush=True)
    coll.commit()
    return coll


def ensure_collections(args: argparse.Namespace, modes: list[str]):
    import lynse  # noqa: F401

    data_dir = args.data_dir
    kinds = {collection_kind_for_mode(mode) for mode in modes}
    kinds.add("dense")  # shared / sparse / named always need dense

    marker_path = data_dir / "profile.json"
    if args.reuse and marker_path.exists():
        existing = json.loads(marker_path.read_text())
        wanted = profile_marker(args)
        # Compare without modes order noise for reuse of datasets.
        key_existing = {k: existing[k] for k in wanted if k != "modes"}
        key_wanted = {k: wanted[k] for k in wanted if k != "modes"}
        if key_existing == key_wanted:
            print(f"Reusing datasets under {data_dir}", flush=True)
            client = open_client(data_dir)
            db = client.get_database("gate_db")
            colls = {}
            for kind in kinds:
                name = f"{kind}_vectors"
                dim = 2 if kind == "haversine" else args.dim
                colls[kind] = db.require_collection(name, dim=dim, drop_if_exists=False)
            return client, colls

    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    client = open_client(data_dir)
    colls = {}
    for kind in sorted(kinds):
        dim = 2 if kind == "haversine" else args.dim
        colls[kind] = create_collection(
            client, f"{kind}_vectors", n=args.rows, dim=dim, kind=kind, args=args
        )
    marker_path.write_text(json.dumps(profile_marker(args), indent=2, sort_keys=True) + "\n")
    return client, colls


def build_exact_refs(coll, queries: np.ndarray, metrics: set[str], args: argparse.Namespace):
    refs: dict[str, list[list[int]]] = {}
    for metric in sorted(metrics):
        mode = exact_mode_for_metric(metric)
        build_args = {}
        if mode.upper().startswith(("IVF", "SPANN")):
            build_args["n_clusters"] = n_clusters_for(args.rows)
        coll.build_index(mode, **build_args)
        refs[metric] = [result_ids(coll.search(query, k=args.k, nprobe=args.nprobe)) for query in queries]
    return refs


def run_one_mode(coll, mode: str, queries: np.ndarray, refs: dict[str, list[list[int]]], args: argparse.Namespace):
    metric = metric_for_mode(mode)
    where = '"bucket" < 100'
    batch = queries[: min(args.batch_queries, len(queries))]
    row: dict[str, Any] = {
        "mode": mode,
        "metric": metric,
        "collection_kind": collection_kind_for_mode(mode),
        "min_recall": recall_floor_for_mode(mode),
    }
    try:
        build_args = {}
        if mode.upper().startswith(("IVF", "SPANN")):
            build_args["n_clusters"] = n_clusters_for(args.rows)
        t0 = time.perf_counter()
        coll.build_index(mode, **build_args)
        row["build_ms"] = (time.perf_counter() - t0) * 1000.0

        for _ in range(args.warmups):
            coll.search(queries[0], k=args.k, nprobe=args.nprobe)

        search_samples = []
        filter_samples = []
        batch_samples = []
        batch_filter_samples = []
        recalls = []
        filter_bad_ids = 0
        for i in range(args.trials):
            query = queries[i % len(queries)]
            search_samples.append(
                timed_ms(lambda q=query: coll.search(q, k=args.k, nprobe=args.nprobe))
            )
            got = result_ids(coll.search(query, k=args.k, nprobe=args.nprobe))
            recalls.append(recall_at_k(got, refs[metric][i % len(refs[metric])], args.k))

            filter_samples.append(
                timed_ms(
                    lambda q=query: coll.search(q, k=args.k, where=where, nprobe=args.nprobe)
                )
            )
            filtered = result_ids(coll.search(query, k=args.k, where=where, nprobe=args.nprobe))
            filter_bad_ids += sum(1 for item_id in filtered if int(item_id) % 1000 >= 100)

            batch_samples.append(
                timed_ms(
                    lambda: safe_batch_search(
                        coll, batch, k=args.k, nprobe=args.nprobe
                    )
                )
            )
            batch_filter_samples.append(
                timed_ms(
                    lambda: safe_batch_search(
                        coll, batch, k=args.k, where=where, nprobe=args.nprobe
                    )
                )
            )

        row.update(
            {
                "status": "ok",
                "search_p50_ms": percentile(search_samples, 50),
                "filter_p50_ms": percentile(filter_samples, 50),
                "batch_p50_ms": percentile(batch_samples, 50),
                "batch_filter_p50_ms": percentile(batch_filter_samples, 50),
                "recall_at_k": float(statistics.mean(recalls)) if recalls else 0.0,
                "filter_bad_ids": filter_bad_ids,
            }
        )
        print(
            f"[mode] {mode:28s} build={row['build_ms']:10.1f}ms "
            f"search={row['search_p50_ms']:8.3f} filter={row['filter_p50_ms']:8.3f} "
            f"recall={row['recall_at_k']:.3f} bad={filter_bad_ids}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - matrix must report every failure
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[mode] {mode:28s} ERROR {row['error']}", flush=True)
    return row


def run_sparse(coll, args: argparse.Namespace) -> dict[str, Any]:
    print("Sparse vector ingest + search...", flush=True)
    where = '"bucket" < 100'
    t0 = time.perf_counter()
    inserted = 0
    while inserted < args.rows:
        size = min(args.batch_size, args.rows - inserted)
        ids = list(range(inserted, inserted + size))
        coll.add_sparse_vectors(
            make_sparse_vectors(
                inserted, size, nnz=args.sparse_nnz, dims=args.sparse_dims
            ),
            ids,
        )
        inserted += size
    coll.commit()
    ingest_ms = (time.perf_counter() - t0) * 1000.0

    rng = np.random.default_rng(args.seed + 4242)
    queries = []
    for i in range(max(args.trials, args.batch_queries)):
        idxs = rng.choice(args.sparse_dims, size=args.sparse_nnz, replace=False)
        queries.append({int(idx): float(1.0 / (j + 1)) for j, idx in enumerate(idxs)})

    for _ in range(args.warmups):
        coll.search_sparse(queries[0], k=args.k)

    search_samples = []
    filter_samples = []
    filter_bad_ids = 0
    for i in range(args.trials):
        q = queries[i % len(queries)]
        search_samples.append(timed_ms(lambda q=q: coll.search_sparse(q, k=args.k)))
        filter_samples.append(
            timed_ms(lambda q=q: coll.search_sparse(q, k=args.k, where=where))
        )
        filtered = result_ids(coll.search_sparse(q, k=args.k, where=where))
        filter_bad_ids += sum(1 for item_id in filtered if int(item_id) % 1000 >= 100)

    row = {
        "status": "ok",
        "ingest_ms": ingest_ms,
        "search_p50_ms": percentile(search_samples, 50),
        "filter_p50_ms": percentile(filter_samples, 50),
        "filter_bad_ids": filter_bad_ids,
    }
    print(
        f"[sparse] ingest={ingest_ms:.1f}ms search={row['search_p50_ms']:.3f} "
        f"filter={row['filter_p50_ms']:.3f} bad={filter_bad_ids}",
        flush=True,
    )
    return row


def safe_batch_search(coll, queries: np.ndarray, *, k: int, where: str | None = None, nprobe: int = 10):
    """Batch workload helper that stays deadlock-safe across releases.

    Default (``GATE_SAFE_BATCH=1``): sequential ``search`` loop — comparable on
    v0.7.1 and current, and avoids the nested-Rayon hang in old ``batch_search``.
    Set ``GATE_SAFE_BATCH=0`` to call native ``batch_search``.
    """
    if os.environ.get("GATE_SAFE_BATCH", "1") == "0":
        return coll.batch_search(queries, k=k, where=where, nprobe=nprobe)
    results = []
    for query in queries:
        results.append(coll.search(query, k=k, where=where, nprobe=nprobe))
    return results


def run_shared(coll, args: argparse.Namespace) -> dict[str, Any]:
    print("Shared dense query paths (FLAT-IP)...", flush=True)
    coll.build_index("FLAT-IP")
    rng = np.random.default_rng(args.seed + 7)
    query = rng.random(args.dim, dtype=np.float32)
    batch = rng.random((args.batch_queries, args.dim), dtype=np.float32)
    exact_order = min(123_456, args.rows - 1)
    filters = {
        "exact": f'"order" = {exact_order}',
        "1pct": '"bucket" < 10',
        "10pct": '"bucket" < 100',
    }
    use_safe_batch = os.environ.get("GATE_SAFE_BATCH", "1") != "0"

    def batch_fn(where: str | None = None):
        if use_safe_batch:
            # Prefer sequential single-search loop for cross-version fairness:
            # native batch_search deadlocks on v0.7.1 at 1M with nested Rayon.
            for q in batch:
                coll.search(q, k=args.k, where=where, nprobe=args.nprobe)
        else:
            coll.batch_search(batch, k=args.k, where=where, nprobe=args.nprobe)

    def bench(label: str, fn, loops: int | None = None) -> float:
        print(f"[shared] starting {label}...", flush=True)
        loops = args.trials if loops is None else loops
        for _ in range(args.warmups):
            fn()
        samples = [timed_ms(fn) for _ in range(loops)]
        value = percentile(samples, 50)
        print(f"[shared] {label:32s} p50={value:9.3f} ms", flush=True)
        return value

    shared: dict[str, Any] = {
        "batch_impl": "sequential_search_loop" if use_safe_batch else "native_batch_search",
        "query_exact_p50_ms": bench(
            "query exact",
            lambda: coll.query(filters["exact"], return_ids_only=True),
            loops=args.trials * 2,
        ),
        "query_1pct_p50_ms": bench(
            "query 1pct", lambda: coll.query(filters["1pct"], return_ids_only=True)
        ),
        "query_10pct_p50_ms": bench(
            "query 10pct", lambda: coll.query(filters["10pct"], return_ids_only=True)
        ),
        "search_p50_ms": bench("dense search", lambda: coll.search(query, k=args.k)),
        "search_filter_p50_ms": bench(
            "dense search 10pct",
            lambda: coll.search(query, k=args.k, where=filters["10pct"]),
        ),
        "approx_p50_ms": bench(
            "approx search",
            lambda: coll.search(query, k=args.k, approx=True, eps=1e-4),
        ),
        "bm25_p50_ms": bench("bm25", lambda: coll.bm25_search("topic42", k=args.k)),
        "bm25_filter_p50_ms": bench(
            "bm25 10pct",
            lambda: coll.bm25_search("topic42", k=args.k, where=filters["10pct"]),
        ),
        "hybrid_p50_ms": bench(
            "hybrid",
            lambda: coll.hybrid_search(
                query, text="topic42", k=args.k, candidate_limit=100
            ),
        ),
        "hybrid_filter_p50_ms": bench(
            "hybrid 10pct",
            lambda: coll.hybrid_search(
                query,
                text="topic42",
                k=args.k,
                where=filters["10pct"],
                candidate_limit=100,
            ),
        ),
        "batch_p50_ms": bench(
            "batch search",
            lambda: batch_fn(None),
            loops=max(2, args.trials),
        ),
        "batch_filter_p50_ms": bench(
            "batch search 10pct",
            lambda: batch_fn(filters["10pct"]),
            loops=max(2, args.trials),
        ),
    }

    # Named dense field on a measurable subset (full 1M duplicate is prohibitive).
    named_n = min(args.rows, max(args.batch_size, 100_000))
    coll.create_vector_field("image", args.dim, metric="ip", index_mode="FLAT-IP")
    named_rng = np.random.default_rng(args.seed + 99)
    inserted = 0
    while inserted < named_n:
        size = min(args.batch_size, named_n - inserted)
        coll.add_named_vectors(
            "image",
            named_rng.random((size, args.dim), dtype=np.float32),
            list(range(inserted, inserted + size)),
        )
        inserted += size
    coll.commit()
    named_query = named_rng.random(args.dim, dtype=np.float32)
    shared["named_dense_n"] = named_n
    shared["named_dense_p50_ms"] = bench(
        "named dense search",
        lambda: coll.search(named_query, k=args.k, vector_field="image"),
    )

    commit_samples = []
    for _ in range(args.warmups):
        coll.commit()
    for _ in range(args.trials):
        commit_samples.append(timed_ms(coll.commit))
    shared["commit_median_ms"] = median_ms(commit_samples)

    delete_ids = list(range(0, args.rows, 20))
    coll.delete(delete_ids)
    tombstone_samples = []
    for _ in range(args.warmups):
        coll.search(query, k=args.k)
    for _ in range(args.trials):
        tombstone_samples.append(timed_ms(lambda: coll.search(query, k=args.k)))
        hit_ids = result_ids(coll.search(query, k=args.k))
        assert all(i not in delete_ids for i in hit_ids), hit_ids
    shared["tombstone_search_median_ms"] = median_ms(tombstone_samples)
    # Restore for later mode builds on the same collection.
    coll.restore(delete_ids)
    coll.commit()
    return shared


def run_upsert(args: argparse.Namespace) -> dict[str, Any]:
    import lynse._core as rb

    test_dir = args.data_dir / "upsert"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 11)
    mgr = rb.DatabaseManager(str(test_dir))
    mgr.create_database("bench_db")
    mgr.require_collection("bench_db", "bench_vectors", args.dim)
    coll = mgr.get_collection("bench_db", "bench_vectors", args.dim)
    for start in range(0, args.rows, args.batch_size):
        end = min(start + args.batch_size, args.rows)
        coll.add_items(rng.random((end - start, args.dim), dtype=np.float32), list(range(start, end)))
    coll.commit()

    updates = {}
    for update_size in (1, 100):
        samples = []
        for trial in range(max(1, args.trials // 2)):
            ids = list(range(trial, trial + update_size))
            vectors = rng.random((update_size, args.dim), dtype=np.float32)
            samples.append(
                timed_ms(lambda ids=ids, vectors=vectors: coll.update_items(ids, vectors, None))
            )
        updates[str(update_size)] = {"median_ms": median_ms(samples)}
    return updates


def make_queries(kind: str, args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed + 999 + hash(kind) % 1000)
    nq = max(args.trials, args.batch_queries, 4)
    if kind == "binary":
        return make_binary(rng, nq, args.dim)
    if kind == "haversine":
        return make_haversine(rng, nq)
    if kind == "distribution":
        return make_distribution(rng, nq, args.dim)
    return make_dense(rng, nq, args.dim)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("RAYON_NUM_THREADS", "4"))
    modes = [] if args.skip_modes else configured_modes(args.modes)
    print("=" * 100, flush=True)
    print(
        f"gate_matrix_bench side={args.side} rows={args.rows:,} dim={args.dim} "
        f"modes={len(modes)} data={args.data_dir}",
        flush=True,
    )
    print("=" * 100, flush=True)

    _, colls = ensure_collections(args, modes or ["FLAT-IP"])
    payload: dict[str, Any] = {
        "schema_version": 3,
        "side": args.side,
        "git_ref": args.git_ref,
        "profile": profile_marker(args),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "modes": {},
    }

    if not args.skip_upsert:
        print("Upsert bench...", flush=True)
        payload["upsert"] = run_upsert(args)

    if not args.skip_shared:
        payload["shared"] = run_shared(colls["dense"], args)

    if not args.skip_sparse:
        try:
            payload["sparse"] = run_sparse(colls["dense"], args)
        except Exception as exc:  # noqa: BLE001
            payload["sparse"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[sparse] ERROR {payload['sparse']['error']}", flush=True)

    if args.skip_modes:
        print("Index mode matrix skipped (--skip-modes)", flush=True)
    else:
        by_kind: dict[str, list[str]] = {}
        for mode in modes:
            by_kind.setdefault(collection_kind_for_mode(mode), []).append(mode)

        for kind, kind_modes in by_kind.items():
            coll = colls[kind]
            queries = make_queries(kind, args)
            metrics = {metric_for_mode(mode) for mode in kind_modes}
            print(f"Building exact refs for {kind}: {sorted(metrics)}", flush=True)
            refs = build_exact_refs(coll, queries, metrics, args)
            for mode in kind_modes:
                payload["modes"][mode] = run_one_mode(coll, mode, queries, refs, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
