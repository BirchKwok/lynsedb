#!/usr/bin/env python3
"""
FlatIndex (mmap) benchmark: 1M × 128 brute-force search.

Measures:
  - Write throughput (raw binary)
  - Search latency: IP, L2, Cosine at 1M scale
  - Warmup vs hot-cache performance
  - Python baseline comparison (simsimd + argpartition)

Usage:
    python benchmarks/flat_mmap_1m.py
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

try:
    import simsimd
    HAS_SIMSIMD = True
except ImportError:
    HAS_SIMSIMD = False


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


def py_ip_search(query, candidates, k):
    """Python baseline: simsimd IP + argpartition top-k."""
    dists = np.asarray(simsimd.inner(query.squeeze(), candidates)).squeeze()
    if k >= len(dists):
        idx = np.argsort(-dists)
        return idx, dists[idx]
    part = np.argpartition(dists, len(dists) - k)[-k:]
    top_idx = part[np.argsort(-dists[part])]
    return top_idx, dists[top_idx]


def py_l2_search(query, candidates, k):
    """Python baseline: simsimd L2 + argpartition top-k."""
    dists = np.asarray(simsimd.sqeuclidean(query.squeeze(), candidates)).squeeze()
    if k >= len(dists):
        idx = np.argsort(dists)
        return idx, dists[idx]
    part = np.argpartition(dists, k)[:k]
    top_idx = part[np.argsort(dists[part])]
    return top_idx, dists[top_idx]


def main():
    print("=" * 70)
    print("  FlatIndex (mmap) Benchmark: Brute-Force Search")
    print(f"  Platform: {sys.platform} | NumPy {np.__version__}")
    print(f"  SimSIMD: {'yes' if HAS_SIMSIMD else 'no'}")
    print("=" * 70)

    DIM = 128
    K = 10

    for N in [100_000, 500_000, 1_000_000, 2_000_000]:
        print(f"\n{'=' * 70}")
        print(f"  N = {N:,} vectors × dim={DIM}, top-{K}")
        print(f"  Data size: {N * DIM * 4 / 1024 / 1024:.0f} MB")
        print(f"{'=' * 70}")

        # Generate random data
        np.random.seed(42)
        data = np.random.randn(N, DIM).astype(np.float32)
        query = np.random.randn(DIM).astype(np.float32)

        # ── Write ──
        tmpdir = tempfile.mkdtemp(prefix="flat_mmap_bench_")
        path = os.path.join(tmpdir, "vectors.bin")

        idx = lynse_core.FlatIndex(path, DIM)
        t0 = time.perf_counter()
        idx.write(data)
        t_write = (time.perf_counter() - t0) * 1000
        print(f"\n  Write {N:,}×{DIM}: {t_write:.1f}ms ({N * DIM * 4 / t_write / 1e6:.0f} GB/s)")
        assert len(idx) == N

        # ── Rust FlatIndex search ──
        for metric, label in [("ip", "IP"), ("l2", "L2"), ("cosine", "Cosine")]:
            med, mn, mx = timer(lambda m=metric: idx.search(query, k=K, metric=m), warmup=5, repeat=20)
            print(f"\n  Rust FlatIndex {label}:")
            print(f"    Median: {med:.3f}ms  (min={mn:.3f}, max={mx:.3f})")

        # ── Python baseline (simsimd) ──
        if HAS_SIMSIMD:
            med_py, mn_py, mx_py = timer(lambda: py_ip_search(query, data, K), warmup=3, repeat=10)
            print(f"\n  Python simsimd IP:")
            print(f"    Median: {med_py:.3f}ms  (min={mn_py:.3f}, max={mx_py:.3f})")

            # Speedup
            med_rust, _, _ = timer(lambda: idx.search(query, k=K, metric="ip"), warmup=3, repeat=10)
            if med_py > 0:
                ratio = med_py / med_rust
                winner = f"Rust {ratio:.1f}x faster" if ratio > 1.05 else (
                    f"Python {1/ratio:.1f}x faster" if ratio < 0.95 else "~Tie"
                )
                print(f"    → {winner}")

            med_py_l2, _, _ = timer(lambda: py_l2_search(query, data, K), warmup=3, repeat=10)
            med_rust_l2, _, _ = timer(lambda: idx.search(query, k=K, metric="l2"), warmup=3, repeat=10)
            print(f"\n  Python simsimd L2:")
            print(f"    Median: {med_py_l2:.3f}ms")
            if med_py_l2 > 0:
                ratio = med_py_l2 / med_rust_l2
                winner = f"Rust {ratio:.1f}x faster" if ratio > 1.05 else (
                    f"Python {1/ratio:.1f}x faster" if ratio < 0.95 else "~Tie"
                )
                print(f"    → {winner}")

        # ── Correctness check ──
        if HAS_SIMSIMD:
            rust_ids, rust_dists = idx.search(query, k=K, metric="ip")
            py_ids, py_dists = py_ip_search(query, data, K)
            # Top-1 should match
            if rust_ids[0] == py_ids[0]:
                print(f"\n  ✓ Correctness: Rust top-1 matches Python (id={rust_ids[0]})")
            else:
                print(f"\n  ✗ Mismatch: Rust={rust_ids[0]}, Python={py_ids[0]}")
                print(f"    Rust dists: {rust_dists[:5]}")
                print(f"    Python dists: {py_dists[:5]}")

        # Cleanup
        del idx
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
