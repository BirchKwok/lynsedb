#!/usr/bin/env python3
"""End-to-end baseline for current IVF MiniBatchKMeans build and query speed.

Usage:
    python benchmarks/ivf_kmeans_baseline.py

Environment variables:
    LYNSE_BENCH_N=100000
    LYNSE_BENCH_DIM=128
    LYNSE_BENCH_K=256
    LYNSE_BENCH_INDEX_K=256
    LYNSE_BENCH_QUERIES=2000
    LYNSE_BENCH_BATCH=256
    LYNSE_BENCH_NPROBE=1
    LYNSE_BENCH_DTYPES=float32,float16

Note:
    `LYNSE_BENCH_K` controls the synthetic data cluster count. `LYNSE_BENCH_INDEX_K`
    controls the IVF centroid count passed to the Rust public build path.
"""

from __future__ import annotations

import os
import platform
import shutil
import statistics
import tempfile
import time

import numpy as np

from lynse._backend import RustEngine


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_list(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [part.strip() for part in value.split(",") if part.strip()]


def generate_clustered_unit_vectors(n: int, dim: int, k: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, dim), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12

    assignments = np.arange(n) % k
    noise = rng.normal(0.0, 0.03, size=(n, dim)).astype(np.float32)
    data = centers[assignments] + noise
    data /= np.linalg.norm(data, axis=1, keepdims=True) + 1e-12
    return np.ascontiguousarray(data, dtype=np.float32)


def percentile_ms(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    idx = min(len(samples) - 1, max(0, int(round((len(samples) - 1) * p))))
    return sorted(samples)[idx] * 1_000.0


def main() -> None:
    n = env_int("LYNSE_BENCH_N", 100_000)
    dim = env_int("LYNSE_BENCH_DIM", 128)
    k = env_int("LYNSE_BENCH_K", 256)
    index_k = env_int("LYNSE_BENCH_INDEX_K", 256)
    n_queries = min(env_int("LYNSE_BENCH_QUERIES", 2_000), n)
    batch_size = max(1, env_int("LYNSE_BENCH_BATCH", 256))
    nprobe = max(1, min(env_int("LYNSE_BENCH_NPROBE", 1), index_k))
    dtypes = env_list("LYNSE_BENCH_DTYPES", "float32,float16")

    print(
        f"IVF_BASELINE_CONFIG n={n} dim={dim} data_k={k} index_k={index_k} n_queries={n_queries} "
        f"batch_size={batch_size} nprobe={nprobe} dtypes={','.join(dtypes)} machine={platform.machine()} "
        f"python={platform.python_version()} platform={platform.platform()}"
    )

    vectors = generate_clustered_unit_vectors(n, dim, k)
    queries = vectors[:n_queries]
    ids = list(range(n))

    for dtype in dtypes:
        root = tempfile.mkdtemp(prefix=f"lynse_ivf_baseline_{dtype}_")
        try:
            engine = RustEngine(root)
            collection = engine.create_collection("bench", dim, dtypes=dtype)

            t0 = time.perf_counter()
            collection.add_items(vectors, ids)
            ingest_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            collection.build_index("IVF", n_clusters=index_k)
            build_s = time.perf_counter() - t1

            single_samples = []
            for query in queries:
                t = time.perf_counter()
                result = collection.search(query, k=10, nprobe=nprobe)
                single_samples.append(time.perf_counter() - t)
                if len(result.ids) == 0:
                    raise RuntimeError("empty IVF result during baseline")

            t2 = time.perf_counter()
            batch_result_count = 0
            for start in range(0, n_queries, batch_size):
                batch = queries[start : start + batch_size]
                results = collection.batch_search(batch, k=10, nprobe=nprobe)
                batch_result_count += len(results)
            batch_s = time.perf_counter() - t2

            print(
                "IVF_BASELINE_RESULT "
                f"dtype={dtype} "
                f"ingest_ms={ingest_s * 1_000.0:.2f} "
                f"build_ms={build_s * 1_000.0:.2f} "
                f"build_vec_per_s={n / max(build_s, 1e-9):.0f} "
                f"single_avg_ms={statistics.fmean(single_samples) * 1_000.0:.4f} "
                f"single_p50_ms={percentile_ms(single_samples, 0.50):.4f} "
                f"single_p95_ms={percentile_ms(single_samples, 0.95):.4f} "
                f"single_p99_ms={percentile_ms(single_samples, 0.99):.4f} "
                f"single_qps={n_queries / max(sum(single_samples), 1e-9):.0f} "
                f"batch_ms={batch_s * 1_000.0:.2f} "
                f"batch_qps={batch_result_count / max(batch_s, 1e-9):.0f}"
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
