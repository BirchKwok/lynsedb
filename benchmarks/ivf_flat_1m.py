#!/usr/bin/env python3
"""
IVF_FLAT (mmap) benchmark: 1M × 128 search with various nprobe values.

Compares:
  - FlatIndex (brute force) baseline
  - IvfFlatIndex with nprobe=1,5,10,20,50
  - Recall @ each nprobe level

Usage:
    python benchmarks/ivf_flat_1m.py
"""

import gc
import os
import shutil
import sys
import time
import tempfile

import numpy as np

try:
    import lynse_core
except ImportError:
    print("ERROR: Build with: cd rust/lynse-core && maturin develop --release")
    sys.exit(1)


def timer(fn, warmup=2, repeat=10):
    """Run fn warmup+repeat times, return (median_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        gc.disable()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        gc.enable()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


def recall_at_k(ivf_ids, bf_ids, k):
    """Compute recall@k: fraction of brute-force top-k found in IVF results."""
    bf_set = set(bf_ids[:k].tolist())
    ivf_set = set(ivf_ids[:k].tolist())
    return len(bf_set & ivf_set) / k


def main():
    print("=" * 70)
    print("  IVF_FLAT Benchmark: 1M × 128 Search")
    print(f"  Platform: {sys.platform} | NumPy {np.__version__}")
    print("=" * 70)

    DIM = 128
    K = 10
    N = 1_000_000
    N_PARTITIONS = 256
    N_ITERS = 15

    print(f"\n  N = {N:,} vectors × dim={DIM}, top-{K}")
    print(f"  Data size: {N * DIM * 4 / 1024 / 1024:.0f} MB")
    print(f"  IVF partitions: {N_PARTITIONS}, KMeans iters: {N_ITERS}")
    print(f"  Avg vectors/partition: {N // N_PARTITIONS:,}")

    # Generate random data
    np.random.seed(42)
    data = np.random.randn(N, DIM).astype(np.float32)
    queries = np.random.randn(10, DIM).astype(np.float32)

    tmpdir = tempfile.mkdtemp(prefix="ivf_bench_")

    # ── Build IVF_FLAT index ──
    ivf_path = os.path.join(tmpdir, "ivf_vectors.bin")
    print(f"\n  Building IVF_FLAT index ({N_PARTITIONS} partitions, {N_ITERS} iters)...")
    t0 = time.perf_counter()
    ivf_idx = lynse_core.IvfFlatIndex.build(ivf_path, data, DIM, N_PARTITIONS, N_ITERS)
    t_build = time.perf_counter() - t0
    print(f"  Build time: {t_build:.2f}s ({N / t_build:.0f} vectors/s)")
    assert len(ivf_idx) == N

    # ── Build FlatIndex (brute force baseline) ──
    flat_path = os.path.join(tmpdir, "flat_vectors.bin")
    flat_idx = lynse_core.FlatIndex(flat_path, DIM)
    flat_idx.write(data)

    # ── Benchmark ──
    for metric, label in [("ip", "IP"), ("l2", "L2")]:
        print(f"\n  {'─' * 60}")
        print(f"  Metric: {label}")
        print(f"  {'─' * 60}")

        # Build metric-aware IVF index
        ivf_path_m = os.path.join(tmpdir, f"ivf_{metric}.bin")
        print(f"\n  Building IVF_FLAT (metric={metric})...")
        t0 = time.perf_counter()
        ivf_idx_m = lynse_core.IvfFlatIndex.build(
            ivf_path_m, data, DIM, N_PARTITIONS, N_ITERS, metric=metric
        )
        t_build = time.perf_counter() - t0
        print(f"  Build time: {t_build:.2f}s")

        # Brute force baseline
        query = queries[0]
        med_bf, mn_bf, mx_bf = timer(lambda: flat_idx.search(query, k=K, metric=metric), warmup=3, repeat=15)
        bf_ids, bf_dists = flat_idx.search(query, k=K, metric=metric)
        print(f"\n  FlatIndex (brute force):")
        print(f"    Median: {med_bf:.3f}ms  (min={mn_bf:.3f}, max={mx_bf:.3f})")

        # IVF with various nprobe
        for nprobe in [1, 5, 10, 20, 50, N_PARTITIONS]:
            med, mn, mx = timer(
                lambda np=nprobe: ivf_idx_m.search(query, k=K, nprobe=np, metric=metric),
                warmup=3, repeat=15,
            )
            ivf_ids, ivf_dists = ivf_idx_m.search(query, k=K, nprobe=nprobe, metric=metric)
            rec = recall_at_k(ivf_ids, bf_ids, K)

            # Multi-query average recall
            recalls = []
            for q in queries:
                _ivf_ids, _ = ivf_idx_m.search(q, k=K, nprobe=nprobe, metric=metric)
                _bf_ids, _ = flat_idx.search(q, k=K, metric=metric)
                recalls.append(recall_at_k(_ivf_ids, _bf_ids, K))
            avg_recall = np.mean(recalls)

            nprobe_label = f"nprobe={nprobe}" if nprobe < N_PARTITIONS else "nprobe=ALL"
            speedup = med_bf / med if med > 0 else float('inf')
            print(f"\n  IVF_FLAT {nprobe_label}:")
            print(f"    Median: {med:.3f}ms  (min={mn:.3f}, max={mx:.3f})")
            print(f"    Recall@{K}: {avg_recall:.1%}  |  Speedup: {speedup:.1f}x vs brute force")

        del ivf_idx_m

    # Cleanup
    del flat_idx
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
