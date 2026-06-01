#!/usr/bin/env python3
"""Benchmark exact vs approximate flat search on 1M x 128 vectors.

Measures latency, top-10 recall, and distance error vs exact f32 search.
Reuses /tmp/lynse_bench_query if present (from bench_lynse_query.py).
"""
import shutil
import time
from pathlib import Path

import numpy as np

import lynse

DATA_DIR = Path("/tmp/lynse_bench_approx")
REUSE_DIR = Path("/tmp/lynse_bench_query")
N = 1_000_000
DIM = 128
K = 10
EPS = 1e-4
WARMUP = 30
LOOPS = 80
QUALITY_QUERIES = 20


def bench(fn, loops=LOOPS):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(loops):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.1)]


def eval_approx_with_gt(result, gt_ids, gt_dists):
    ids = [int(x) for x in result.ids]
    recall = len(set(ids) & gt_ids) / K
    errs = []
    for i, d in zip(result.ids, result.distances):
        uid = int(i)
        if uid in gt_dists:
            errs.append(abs(float(d) - gt_dists[uid]))
    mean_err = float(np.mean(errs)) if errs else float("nan")
    max_err = float(np.max(errs)) if errs else float("nan")
    within_eps = sum(1 for e in errs if e <= EPS + 1e-9) / len(errs) if errs else 0.0
    return recall, mean_err, max_err, within_eps


def prepare_collection():
    if REUSE_DIR.exists():
        print(f"Using dataset from {REUSE_DIR}", flush=True)
        client = lynse.VectorDBClient(str(REUSE_DIR))
        db = client.get_database("bench_db")
        return db.require_collection("vectors", dim=DIM)

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    client = lynse.VectorDBClient(str(DATA_DIR))
    db = client.create_database("bench_db", drop_if_exists=True)
    collection = db.require_collection("vectors", dim=DIM, drop_if_exists=True)

    query = np.random.random(DIM).astype(np.float32)
    print(f"Inserting {N:,} rows...", flush=True)
    t0 = time.perf_counter()
    with collection.insert_session() as session:
        for i in range(N):
            vec = query if i == 0 else np.random.random(DIM).astype(np.float32)
            session.add_item(vec, id=i, field={"order": i})
    print(f"Insert done in {time.perf_counter() - t0:.1f}s", flush=True)
    return collection


def main():
    collection = prepare_collection()
    query = np.random.random(DIM).astype(np.float32)

    print("Warming up...", flush=True)
    med_ex, p10_ex = bench(lambda: collection.search(query, k=K, approx=False))
    med_ap, p10_ap = bench(lambda: collection.search(query, k=K, approx=True, eps=EPS))

    recalls, mean_errs, max_errs, within = [], [], [], []
    for seed in range(QUALITY_QUERIES):
        rng = np.random.default_rng(seed + 1000)
        q = rng.random(DIM, dtype=np.float32)
        exact_q = collection.search(q, k=K, approx=False)
        gt_ids = set(int(x) for x in exact_q.ids)
        gt_dists = {int(i): float(d) for i, d in zip(exact_q.ids, exact_q.distances)}
        approx_q = collection.search(q, k=K, approx=True, eps=EPS)
        r, me, mx, we = eval_approx_with_gt(approx_q, gt_ids, gt_dists)
        recalls.append(r)
        mean_errs.append(me)
        max_errs.append(mx)
        within.append(we)

    print("=" * 60, flush=True)
    print(f"Benchmark: {N:,} vectors x {DIM}d, k={K}, eps={EPS}", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Mode':<12} {'median us':>12} {'p10 us':>12}", flush=True)
    print(f"{'exact':<12} {med_ex:12.1f} {p10_ex:12.1f}", flush=True)
    print(f"{'approx':<12} {med_ap:12.1f} {p10_ap:12.1f}", flush=True)
    speedup = med_ex / med_ap if med_ap > 0 else float("nan")
    print(f"\nSpeedup (median): {speedup:.2f}x", flush=True)
    print(f"\nApprox quality ({QUALITY_QUERIES} random queries):", flush=True)
    print(f"  top-{K} recall (mean): {float(np.mean(recalls)) * 100:.1f}%", flush=True)
    print(f"  top-{K} recall (min):  {float(np.min(recalls)) * 100:.1f}%", flush=True)
    print(f"  mean |dist err|:       {float(np.mean(mean_errs)):.6f}", flush=True)
    print(f"  max  |dist err| (worst): {float(np.max(max_errs)):.6f}", flush=True)
    print(f"  frac err <= eps (mean): {float(np.mean(within)) * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
