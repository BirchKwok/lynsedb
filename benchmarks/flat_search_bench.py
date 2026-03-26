"""Benchmark flat/brute-force search on 1M × 128 vectors.

Uses Rust backend directly for fast bulk writes.
"""
import numpy as np
import time
import shutil
import os

def bench():
    # Use Rust backend directly for fast writes
    import lynse._core as rb

    test_dir = '/tmp/lynse_flat_bench'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    d = 128
    n = 1_000_000

    mgr = rb.DatabaseManager(test_dir)
    mgr.create_database("bench_db")
    mgr.require_collection("bench_db", "bench_vectors", d)
    coll = mgr.get_collection("bench_db", "bench_vectors", d)

    print(f"Writing {n} vectors (dim={d})...")
    query = np.random.random(d).astype(np.float32)

    # Bulk write via Rust (fast — no Python per-item overhead)
    batch_size = 100_000
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        vecs = np.random.random((end - start, d)).astype(np.float32)
        if start == 0:
            vecs[0] = query
        coll.add_items(vecs, None)
        print(f"  Written {end}/{n}")

    coll.commit()
    shape = coll.shape()
    print(f"Shape: {shape}")

    # Warmup (first search triggers mmap creation + page faults)
    # 20 warmups to ensure all 512MB is resident in page cache
    print("\nWarmup searches...")
    for _ in range(20):
        coll.search(query, 10, None, 10)

    # Benchmark
    n_trials = 30
    times = []
    print(f"\nBenchmarking {n_trials} searches (k=10)...")
    for i in range(n_trials):
        t0 = time.perf_counter()
        result = coll.search(query, 10, None, 10)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        times.append(ms)

    times.sort()
    median = times[len(times) // 2]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    mean = sum(times) / len(times)

    n_threads = os.environ.get('RAYON_NUM_THREADS', 'default')
    print(f"\n=== Results ({n} × {d}, top-10, threads={n_threads}) ===")
    print(f"  Median: {median:.2f} ms")
    print(f"  Mean:   {mean:.2f} ms")
    print(f"  P10:    {p10:.2f} ms")
    print(f"  P90:    {p90:.2f} ms")
    print(f"  Min:    {min(times):.2f} ms")
    print(f"  Max:    {max(times):.2f} ms")

    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    bench()
