#!/usr/bin/env python3
"""
SIFT1M benchmark: 1M × 128 with ground-truth recall evaluation.

Dataset: http://corpus-texmex.irisa.fr/ (ANN_SIFT1M)
  - sift_base.fvecs:  1,000,000 × 128  (base vectors)
  - sift_query.fvecs:     10,000 × 128  (query vectors)
  - sift_groundtruth.ivecs: 10,000 × 100 (ground-truth NN ids, L2)

Usage:
    python benchmarks/sift1m_bench.py
"""

import gc
import os
import shutil
import struct
import sys
import time
import tempfile

import numpy as np

try:
    import lynse_core
except ImportError:
    print("ERROR: Build with: cd rust/lynse-core && maturin develop --release")
    sys.exit(1)


SIFT_DIR = "/Users/guobingming/Downloads/sift"


# ── fvecs / ivecs loaders ──────────────────────────────────────────────────

def read_fvecs(path: str) -> np.ndarray:
    """Read .fvecs file → (n, dim) float32 array."""
    with open(path, "rb") as f:
        buf = f.read()
    offset = 0
    vectors = []
    while offset < len(buf):
        dim = struct.unpack_from("<i", buf, offset)[0]
        offset += 4
        vec = np.frombuffer(buf, dtype=np.float32, count=dim, offset=offset)
        vectors.append(vec)
        offset += dim * 4
    return np.array(vectors, dtype=np.float32)


def read_fvecs_fast(path: str) -> np.ndarray:
    """Fast .fvecs reader using numpy (avoids Python loop)."""
    data = np.fromfile(path, dtype=np.float32)
    # First 4 bytes of first vector encode dim as int32
    dim = int(np.frombuffer(data[:1].tobytes(), dtype=np.int32)[0])
    # Each row: 1 int (dim) + dim floats = (dim+1) floats
    stride = dim + 1
    n = data.shape[0] // stride
    # Reshape and drop the first column (dim prefix)
    return data.reshape(n, stride)[:, 1:].copy()


def read_ivecs_fast(path: str) -> np.ndarray:
    """Fast .ivecs reader using numpy."""
    data = np.fromfile(path, dtype=np.int32)
    dim = data[0]
    stride = dim + 1
    n = data.shape[0] // stride
    return data.reshape(n, stride)[:, 1:].copy()


# ── Timer ──────────────────────────────────────────────────────────────────

def timer(fn, warmup=3, repeat=20):
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


def recall_at_k(result_ids, gt_ids, k):
    """Recall@k: fraction of ground-truth top-k found in result."""
    gt_set = set(gt_ids[:k].tolist())
    res_set = set(result_ids[:k].tolist())
    return len(gt_set & res_set) / k


def batch_recall(index, queries, gt, k, nprobe=None, metric="l2", n_queries=100):
    """Average recall@k over n_queries."""
    recalls = []
    for i in range(min(n_queries, len(queries))):
        q = queries[i]
        if nprobe is not None:
            ids, _ = index.search(q, k=k, nprobe=nprobe, metric=metric)
        else:
            ids, _ = index.search(q, k=k, metric=metric)
        recalls.append(recall_at_k(ids, gt[i], k))
    return np.mean(recalls)


def main():
    print("=" * 70)
    print("  SIFT1M Benchmark — Ground-Truth Recall")
    print(f"  Platform: {sys.platform} | NumPy {np.__version__}")
    print("=" * 70)

    # ── Load SIFT data ──
    print("\n  Loading SIFT1M dataset...")
    t0 = time.perf_counter()
    base = read_fvecs_fast(os.path.join(SIFT_DIR, "sift_base.fvecs"))
    queries = read_fvecs_fast(os.path.join(SIFT_DIR, "sift_query.fvecs"))
    gt = read_ivecs_fast(os.path.join(SIFT_DIR, "sift_groundtruth.ivecs"))
    t_load = time.perf_counter() - t0

    N, DIM = base.shape
    NQ = queries.shape[0]
    N_GT = gt.shape[1]  # 100 nearest neighbors per query
    K = 10

    print(f"  Base:    {N:>10,} × {DIM} ({base.nbytes / 1024 / 1024:.0f} MB)")
    print(f"  Queries: {NQ:>10,} × {DIM}")
    print(f"  GT:      {NQ:>10,} × {N_GT}")
    print(f"  Load time: {t_load:.2f}s")

    tmpdir = tempfile.mkdtemp(prefix="sift_bench_")

    # ── Build FlatIndex (brute force) ──
    print(f"\n  Building FlatIndex (brute force baseline)...")
    flat_path = os.path.join(tmpdir, "flat.bin")
    t0 = time.perf_counter()
    flat_idx = lynse_core.FlatIndex(flat_path, DIM)
    flat_idx.write(base)
    t_flat = time.perf_counter() - t0
    print(f"  FlatIndex built: {t_flat:.2f}s")

    # ── Brute-force L2 baseline ──
    # SIFT ground truth is computed with L2 distance
    print(f"\n  {'─' * 60}")
    print(f"  Brute Force (L2)")
    print(f"  {'─' * 60}")

    q0 = queries[0]
    med_bf, mn_bf, mx_bf = timer(lambda: flat_idx.search(q0, k=K, metric="l2"))
    print(f"  Single-query latency: {med_bf:.3f}ms (min={mn_bf:.3f}, max={mx_bf:.3f})")

    bf_recall = batch_recall(flat_idx, queries, gt, K, metric="l2", n_queries=100)
    print(f"  Recall@{K} vs GT (100 queries): {bf_recall:.1%}")

    # ── Build IVF_FLAT indexes with different partition counts ──
    for n_parts in [256, 512, 1024]:
        print(f"\n  {'─' * 60}")
        print(f"  IVF_FLAT (L2, {n_parts} partitions)")
        print(f"  {'─' * 60}")

        ivf_path = os.path.join(tmpdir, f"ivf_l2_{n_parts}.bin")
        t0 = time.perf_counter()
        ivf_idx = lynse_core.IvfFlatIndex.build(
            ivf_path, base, DIM, n_parts, 20, metric="l2"
        )
        t_build = time.perf_counter() - t0
        print(f"  Build: {t_build:.1f}s ({N / t_build:.0f} vec/s)")

        for nprobe in [1, 5, 10, 20, 50, 100]:
            med, mn, mx = timer(
                lambda np=nprobe: ivf_idx.search(q0, k=K, nprobe=np, metric="l2"),
                warmup=3, repeat=20,
            )
            rec = batch_recall(ivf_idx, queries, gt, K, nprobe=nprobe, metric="l2", n_queries=100)
            speedup = med_bf / med if med > 0 else float("inf")
            marker = " ◀" if med < 1.0 and rec > 0.9 else ""
            print(f"  nprobe={nprobe:>3}: {med:>7.3f}ms  recall@{K}={rec:.1%}  ({speedup:.1f}x){marker}")

        del ivf_idx

    # ── Also test IP metric (after L2-normalizing SIFT vectors) ──
    print(f"\n  {'─' * 60}")
    print(f"  IVF_FLAT (IP on L2-normalized SIFT, 256 partitions)")
    print(f"  {'─' * 60}")

    # Normalize base and queries for IP search
    base_norm = base / np.linalg.norm(base, axis=1, keepdims=True)
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    ivf_ip_path = os.path.join(tmpdir, "ivf_ip_256.bin")
    t0 = time.perf_counter()
    ivf_ip = lynse_core.IvfFlatIndex.build(
        ivf_ip_path, base_norm.astype(np.float32), DIM, 256, 20, metric="ip"
    )
    t_build = time.perf_counter() - t0
    print(f"  Build: {t_build:.1f}s")

    # For normalized vectors, max IP ≈ min L2, so GT is still valid
    q0_norm = queries_norm[0]
    for nprobe in [1, 5, 10, 20, 50]:
        med, mn, mx = timer(
            lambda np=nprobe: ivf_ip.search(q0_norm, k=K, nprobe=np, metric="ip"),
            warmup=3, repeat=20,
        )
        # Compare against L2 ground truth (valid for normalized vectors)
        rec = batch_recall(ivf_ip, queries_norm, gt, K, nprobe=nprobe, metric="ip", n_queries=100)
        print(f"  nprobe={nprobe:>3}: {med:>7.3f}ms  recall@{K}={rec:.1%}")

    del ivf_ip

    # ── Cleanup ──
    del flat_idx
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
