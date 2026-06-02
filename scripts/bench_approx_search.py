#!/usr/bin/env python3
"""Benchmark exact vs approximate flat search.

Default target is 1M x 128 vectors with k=10 and eps=1e-4. The script reports
latency, top-k recall against exact search, and distance error for the returned
approximate IDs.
"""

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import lynse


DATA_DIR = Path(os.environ.get("LYNSE_BENCH_APPROX_DIR", "/tmp/lynse_bench_approx"))
REUSE_DIR = Path(os.environ.get("LYNSE_BENCH_REUSE_DIR", "/tmp/lynse_bench_query"))
N = int(os.environ.get("LYNSE_BENCH_N", "1000000"))
DIM = int(os.environ.get("LYNSE_BENCH_DIM", "128"))
K = int(os.environ.get("LYNSE_BENCH_K", "10"))
EPS_VALUES = [
    float(x) for x in os.environ.get("LYNSE_BENCH_EPS", "1e-4").split(",") if x.strip()
]
WARMUP = int(os.environ.get("LYNSE_BENCH_WARMUP", "20"))
LOOPS = int(os.environ.get("LYNSE_BENCH_LOOPS", "60"))
QUALITY_QUERIES = int(os.environ.get("LYNSE_BENCH_QUALITY_QUERIES", "20"))
BUILD_CHUNK = int(os.environ.get("LYNSE_BENCH_BUILD_CHUNK", "50000"))
SEED = int(os.environ.get("LYNSE_BENCH_SEED", "20260601"))
MIN_SPEEDUP = float(os.environ.get("LYNSE_BENCH_MIN_SPEEDUP", "1.5"))
MIN_RECALL = float(os.environ.get("LYNSE_BENCH_MIN_RECALL", "0.97"))
INDEX_MODE = os.environ.get("LYNSE_BENCH_INDEX_MODE", "").strip()


def percentile(values, pct):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def time_once(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def bench(fn, loops=LOOPS):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(loops):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return {
        "p50_ms": percentile(times, 50),
        "p90_ms": percentile(times, 90),
        "p99_ms": percentile(times, 99),
    }


def exact_distances(collection, query, ids, metric):
    if len(ids) == 0:
        return np.empty(0, dtype=np.float32)
    vectors = collection.query_vectors(filter_ids=[int(i) for i in ids]).vectors.astype(np.float32)
    q = np.asarray(query, dtype=np.float32)
    metric = (metric or "IP").lower()
    if metric == "ip":
        return vectors @ q
    if metric == "l2":
        diff = vectors - q
        return np.einsum("ij,ij->i", diff, diff)
    if metric == "cosine":
        denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)
        return 1.0 - (vectors @ q) / np.maximum(denom, 1e-30)
    raise ValueError(f"unsupported metric for benchmark: {metric}")


def eval_approx(collection, query, approx_result, exact_result):
    approx_ids = np.asarray(approx_result.ids, dtype=np.int64)
    exact_ids = set(int(x) for x in exact_result.ids)
    recall = len(set(int(x) for x in approx_ids) & exact_ids) / max(1, min(K, len(exact_ids)))

    exact_for_returned = exact_distances(
        collection,
        query,
        approx_ids,
        approx_result.distance_metric or exact_result.distance_metric,
    )
    approx_dists = np.asarray(approx_result.distances, dtype=np.float32)
    errs = np.abs(approx_dists - exact_for_returned)
    return {
        "recall": float(recall),
        "mean_abs_err": float(np.mean(errs)) if len(errs) else float("nan"),
        "max_abs_err": float(np.max(errs)) if len(errs) else float("nan"),
    }


def prepare_collection():
    if REUSE_DIR.exists() and os.environ.get("LYNSE_BENCH_FORCE_REBUILD") != "1":
        print(f"Using dataset from {REUSE_DIR}", flush=True)
        client = lynse.VectorDBClient(str(REUSE_DIR))
        db = client.get_database("bench_db")
        collection = db.require_collection("vectors", dim=DIM)
        if collection.shape[0] >= N and collection.shape[1] == DIM:
            return collection
        print("Reuse dataset shape mismatch; rebuilding benchmark dataset.", flush=True)

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    client = lynse.VectorDBClient(str(DATA_DIR))
    db = client.create_database("bench_db", drop_if_exists=True)
    collection = db.require_collection("vectors", dim=DIM, drop_if_exists=True)

    rng = np.random.default_rng(SEED)
    print(f"Inserting {N:,} rows in chunks of {BUILD_CHUNK:,}...", flush=True)
    t0 = time.perf_counter()
    inserted = 0
    while inserted < N:
        size = min(BUILD_CHUNK, N - inserted)
        vectors = rng.random((size, DIM), dtype=np.float32)
        collection.bulk_add_binary(vectors, batch_size=size, enable_progress_bar=False)
        inserted += size
        if inserted % max(BUILD_CHUNK * 5, 1) == 0 or inserted == N:
            print(f"  inserted {inserted:,}/{N:,}", flush=True)
    collection.commit()
    print(f"Insert done in {time.perf_counter() - t0:.1f}s", flush=True)
    return collection


def main():
    collection = prepare_collection()
    if INDEX_MODE:
        collection.build_index(INDEX_MODE)
    rng = np.random.default_rng(SEED + 1)
    query = rng.random(DIM, dtype=np.float32)

    print("Measuring latency...", flush=True)
    first_approx_latency = {
        eps: time_once(lambda eps=eps: collection.search(query, k=K, approx=True, eps=eps))
        for eps in EPS_VALUES
    }
    exact_latency = bench(lambda: collection.search(query, k=K, approx=False))
    approx_latency = {
        eps: bench(lambda eps=eps: collection.search(query, k=K, approx=True, eps=eps))
        for eps in EPS_VALUES
    }

    quality = {eps: [] for eps in EPS_VALUES}
    for seed in range(QUALITY_QUERIES):
        q = np.random.default_rng(SEED + 1000 + seed).random(DIM, dtype=np.float32)
        exact_q = collection.search(q, k=K, approx=False)
        for eps in EPS_VALUES:
            approx_q = collection.search(q, k=K, approx=True, eps=eps)
            quality[eps].append(eval_approx(collection, q, approx_q, exact_q))

    print("=" * 96, flush=True)
    print(
        f"Benchmark: {N:,} vectors x {DIM}d, k={K}, "
        f"quality_queries={QUALITY_QUERIES}, index={INDEX_MODE or collection.index_mode}",
        flush=True,
    )
    print("=" * 96, flush=True)
    for eps in EPS_VALUES:
        print(
            f"approx first-call init eps={eps:g}: {first_approx_latency[eps]:.3f} ms "
            "(raw-f32 row-order/shortlist cache, no quantization)",
            flush=True,
        )
    print(
        f"{'mode':<14} {'eps':>10} {'p50 ms':>10} {'p90 ms':>10} {'p99 ms':>10} "
        f"{'speedup':>9} "
        f"{'recall@10':>10} {'mean err':>12} {'max err':>12}",
        flush=True,
    )
    print(
        f"{'exact':<14} {'-':>10} {exact_latency['p50_ms']:10.3f} "
        f"{exact_latency['p90_ms']:10.3f} {exact_latency['p99_ms']:10.3f} "
        f"{1.0:9.2f}x {1.0:10.3f} {0.0:12.6f} {0.0:12.6f}",
        flush=True,
    )
    failed = []
    for eps in EPS_VALUES:
        rows = quality[eps]
        recall = float(np.mean([r["recall"] for r in rows]))
        mean_err = float(np.mean([r["mean_abs_err"] for r in rows]))
        max_err = float(np.max([r["max_abs_err"] for r in rows]))
        lat = approx_latency[eps]
        speedup = exact_latency["p50_ms"] / max(lat["p50_ms"], 1e-9)
        print(
            f"{'approx':<14} {eps:10.1e} {lat['p50_ms']:10.3f} "
            f"{lat['p90_ms']:10.3f} {lat['p99_ms']:10.3f} "
            f"{speedup:9.2f}x "
            f"{recall:10.3f} {mean_err:12.6f} {max_err:12.6f}",
            flush=True,
        )
        if MIN_SPEEDUP > 0 and speedup < MIN_SPEEDUP:
            failed.append(f"eps={eps:g}: speedup {speedup:.2f}x < {MIN_SPEEDUP:.2f}x")
        if MIN_RECALL > 0 and recall < MIN_RECALL:
            failed.append(f"eps={eps:g}: recall@10 {recall:.3f} < {MIN_RECALL:.3f}")

    if failed:
        print(
            "FAIL: " + "; ".join(failed),
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
