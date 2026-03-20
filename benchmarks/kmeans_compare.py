#!/usr/bin/env python3
"""
KMeans speed comparison: Rust IVF_FLAT build vs sklearn MiniBatchKMeans.

Uses SIFT1M dataset for realistic comparison.
"""

import os
import shutil
import sys
import tempfile
import time

import numpy as np

SIFT_DIR = "/Users/guobingming/Downloads/sift"


def read_fvecs_fast(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    dim = int(np.frombuffer(data[:1].tobytes(), dtype=np.int32)[0])
    stride = dim + 1
    n = data.shape[0] // stride
    return data.reshape(n, stride)[:, 1:].copy()


def read_ivecs_fast(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    dim = data[0]
    stride = dim + 1
    n = data.shape[0] // stride
    return data.reshape(n, stride)[:, 1:].copy()


def recall_at_k(result_ids, gt_ids, k):
    return len(set(result_ids[:k].tolist()) & set(gt_ids[:k].tolist())) / k


def main():
    print("=" * 70, flush=True)
    print("  Rust IVF_FLAT vs sklearn MiniBatchKMeans", flush=True)
    print("=" * 70, flush=True)

    # Load SIFT
    print("\n  Loading SIFT1M...", flush=True)
    base = read_fvecs_fast(os.path.join(SIFT_DIR, "sift_base.fvecs"))
    queries = read_fvecs_fast(os.path.join(SIFT_DIR, "sift_query.fvecs"))
    gt = read_ivecs_fast(os.path.join(SIFT_DIR, "sift_groundtruth.ivecs"))
    N, DIM = base.shape
    print(f"  {N:,} × {DIM}\n", flush=True)

    K = 10
    NPROBES = [5, 10, 20, 40]

    from sklearn.cluster import MiniBatchKMeans
    import lynse_core

    # ── Part 1: Build speed comparison ──
    print("  ┌─ Build Speed ─────────────────────────────────────────────┐", flush=True)
    print(f"  │ {'k':>6}  {'MiniBatch':>10}  {'Rust':>10}  {'Ratio':>8}             │", flush=True)
    print(f"  │ {'':─>6}  {'':─>10}  {'':─>10}  {'':─>8}             │", flush=True)

    rust_indices = {}
    tmpdirs = []

    for n_clusters in [256, 512, 1024]:
        # sklearn MiniBatchKMeans
        t0 = time.perf_counter()
        mbkm = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=4096, max_iter=20,
            init="k-means++", random_state=42, n_init=1
        )
        mbkm.fit(base)
        t_mb = time.perf_counter() - t0

        # Rust IVF_FLAT build
        tmpdir = tempfile.mkdtemp(prefix="km_bench_")
        tmpdirs.append(tmpdir)
        ivf_path = os.path.join(tmpdir, f"ivf_{n_clusters}.bin")
        t0 = time.perf_counter()
        ivf_idx = lynse_core.IvfFlatIndex.build(
            ivf_path, base, DIM, n_clusters, 20, metric="l2"
        )
        t_rust = time.perf_counter() - t0
        rust_indices[n_clusters] = (ivf_idx, mbkm)

        ratio = t_rust / t_mb
        print(f"  │ {n_clusters:>6}  {t_mb:>9.2f}s  {t_rust:>9.2f}s  {ratio:>7.2f}x             │", flush=True)

    print("  └──────────────────────────────────────────────────────────┘\n", flush=True)

    # ── Part 2: Recall vs nprobe sweep ──
    print("  ┌─ Recall@10 vs nprobe ─────────────────────────────────────┐", flush=True)
    header_probes = "  ".join(f"np={p}" for p in NPROBES)
    print(f"  │ {'k':>6}  {header_probes:>40}  {'sklearn':>8} │", flush=True)
    print(f"  │ {'':─>6}  {'':─>40}  {'':─>8} │", flush=True)

    for n_clusters in [256, 512, 1024]:
        ivf_idx, mbkm = rust_indices[n_clusters]

        # Rust recall at each nprobe
        rust_parts = []
        for nprobe in NPROBES:
            recalls = []
            for qi in range(min(100, len(queries))):
                ids, _ = ivf_idx.search(queries[qi], k=K, nprobe=nprobe, metric="l2")
                recalls.append(recall_at_k(ids, gt[qi], K))
            r = np.mean(recalls)
            rust_parts.append(f"{r:>5.1%}")

        # sklearn recall (nprobe=10 equivalent)
        mb_centroids = mbkm.cluster_centers_
        mb_labels = mbkm.labels_
        mb_recalls = []
        for qi in range(min(100, len(queries))):
            q = queries[qi]
            cdists = np.sum((mb_centroids - q) ** 2, axis=1)
            top_c = np.argsort(cdists)[:10]
            cand_ids = np.concatenate([np.where(mb_labels == c)[0] for c in top_c])
            if len(cand_ids) == 0:
                mb_recalls.append(0.0)
                continue
            vecs = base[cand_ids]
            dists = np.sum((vecs - q) ** 2, axis=1)
            topk = cand_ids[np.argsort(dists)[:K]]
            mb_recalls.append(recall_at_k(topk, gt[qi], K))
        mb_r = np.mean(mb_recalls)

        probes_str = "  ".join(f"{p:>6}" for p in rust_parts)
        print(f"  │ {n_clusters:>6}  {probes_str:>40}  {mb_r:>7.1%} │", flush=True)

    print("  └──────────────────────────────────────────────────────────┘", flush=True)

    # Cleanup
    for ivf_idx, _ in rust_indices.values():
        del ivf_idx
    for d in tmpdirs:
        shutil.rmtree(d, ignore_errors=True)

    print(f"\n{'=' * 70}", flush=True)
    print("  Done.", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
