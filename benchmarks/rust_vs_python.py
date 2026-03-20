#!/usr/bin/env python3
"""
Rust vs Python Backend Performance Benchmark for LynseDB.

Self-contained: uses simsimd/usearch/numpy directly as Python baseline,
does NOT import the lynse package (avoids logger init issues).

Compares:
  1. Distance computation (IP, L2, Cosine)
  2. Top-k search (brute-force)
  3. Vector write throughput
  4. Full collection search (Flat index)
  5. HNSW index build + search
  6. Batch search (Rust only vs Python loop)

Usage:
    python benchmarks/rust_vs_python.py
"""

import gc
import os
import shutil
import sys
import time
import tempfile
from contextlib import contextmanager

import numpy as np
import simsimd
from usearch.compiled import exact_search
from usearch.index import MetricKind

# Import Rust backend
try:
    import lynse_core
except ImportError:
    print("ERROR: Rust backend not available. Build with: cd rust/lynse-core && maturin develop --release")
    sys.exit(1)


# ── Python baseline functions (extracted from lynse.computational_layer.engines) ──

def py_ip_search(query, candidates, k):
    """Python IP search: simsimd.inner + argpartition top-k."""
    dists = simsimd.inner(query.squeeze(), candidates)
    dists = np.asarray(dists).squeeze()
    if k >= len(dists):
        idx = np.argsort(-dists)
        return idx, dists[idx]
    part = np.argpartition(dists, len(dists) - k)[-k:]
    top_dists = dists[part]
    order = np.argsort(-top_dists)
    return part[order], top_dists[order]


def py_l2_search(query, candidates, k):
    """Python L2 search: simsimd.sqeuclidean + argpartition top-k."""
    dists = simsimd.sqeuclidean(query.squeeze(), candidates)
    dists = np.asarray(dists).squeeze()
    if k >= len(dists):
        idx = np.argsort(dists)
        return idx, dists[idx]
    part = np.argpartition(dists, k)[:k]
    top_dists = dists[part]
    order = np.argsort(top_dists)
    return part[order], top_dists[order]


def py_cosine_search(query, candidates, k):
    """Python cosine search: simsimd.cosine + argpartition top-k."""
    dists = simsimd.cosine(query.squeeze(), candidates)
    dists = np.asarray(dists).squeeze()
    if k >= len(dists):
        idx = np.argsort(dists)
        return idx, dists[idx]
    part = np.argpartition(dists, k)[:k]
    top_dists = dists[part]
    order = np.argsort(top_dists)
    return part[order], top_dists[order]

# ── Helpers ──

@contextmanager
def timer(label=""):
    gc.disable()
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    gc.enable()
    return elapsed

def bench(fn, warmup=2, runs=10, label=""):
    """Run fn() multiple times and return (median_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    
    times = []
    for _ in range(runs):
        gc.disable()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        gc.enable()
        times.append(elapsed * 1000)  # ms
    
    times.sort()
    median = times[len(times) // 2]
    return median, times[0], times[-1]


def fmt(median, mn, mx):
    """Format benchmark result."""
    return f"{median:>10.3f}ms  (min={mn:.3f}, max={mx:.3f})"


def winner_str(rust_ms, py_ms):
    if rust_ms < py_ms * 0.95:
        ratio = py_ms / rust_ms
        return f"Rust {ratio:.1f}x faster"
    elif py_ms < rust_ms * 0.95:
        ratio = rust_ms / py_ms
        return f"Python {ratio:.1f}x faster"
    else:
        return "~Tie"


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_row(test_name, rust_result, py_result):
    rust_med, rust_min, rust_max = rust_result
    py_med, py_min, py_max = py_result
    w = winner_str(rust_med, py_med)
    print(f"\n  {test_name}")
    print(f"    Rust:   {fmt(rust_med, rust_min, rust_max)}")
    print(f"    Python: {fmt(py_med, py_min, py_max)}")
    print(f"    Winner: {w}")


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

results_table = []

def record(name, rust_ms, py_ms):
    results_table.append((name, rust_ms, py_ms))


# ── 1. Distance Computation ──────────────────────────────────────────────────

def _bench_single_distance(a, b, metric, py_fn):
    """Benchmark a single distance metric pair (avoids lambda closure issues)."""
    rust_res = bench(
        lambda: lynse_core.py_compute_distance(a, b, metric),
        warmup=100, runs=1000,
    )
    py_res = bench(
        lambda: py_fn(a, b),
        warmup=100, runs=1000,
    )
    return rust_res, py_res


def bench_distance():
    print_header("1. Distance Computation (single pair)")
    
    for dim in [128, 768]:
        a = np.random.randn(dim).astype(np.float32)
        b = np.random.randn(dim).astype(np.float32)
        
        tests = [
            ("ip", "IP", simsimd.inner),
            ("l2", "L2", simsimd.sqeuclidean),
            ("cosine", "COSINE", simsimd.cosine),
        ]
        for metric, label, py_fn in tests:
            rust_res, py_res = _bench_single_distance(a, b, metric, py_fn)
            name = f"Distance {label} dim={dim}"
            print_row(name, rust_res, py_res)
            record(name, rust_res[0], py_res[0])


# ── 2. Top-k Search ──────────────────────────────────────────────────────────

def bench_topk():
    print_header("2. Top-k Search (brute-force, k=10)")
    
    for n in [10_000, 50_000, 100_000]:
        dim = 128
        query = np.random.randn(dim).astype(np.float32)
        candidates = np.random.randn(n, dim).astype(np.float32)
        k = 10
        
        # Rust
        rust_res = bench(
            lambda: lynse_core.py_top_k_search(query, candidates, "ip", k),
            warmup=3, runs=20,
        )
        
        # Python (SimSIMD + argpartition)
        py_res = bench(
            lambda: py_ip_search(query, candidates, k),
            warmup=3, runs=20,
        )
        
        name = f"Top-10 IP n={n:,} dim={dim}"
        print_row(name, rust_res, py_res)
        record(name, rust_res[0], py_res[0])
    
    # L2 variant
    for n in [50_000]:
        dim = 128
        query = np.random.randn(dim).astype(np.float32)
        candidates = np.random.randn(n, dim).astype(np.float32)
        k = 10
        
        rust_res = bench(
            lambda: lynse_core.py_top_k_search(query, candidates, "l2", k),
            warmup=3, runs=20,
        )
        
        py_res = bench(
            lambda: py_l2_search(query, candidates, k),
            warmup=3, runs=20,
        )
        
        name = f"Top-10 L2 n={n:,} dim={dim}"
        print_row(name, rust_res, py_res)
        record(name, rust_res[0], py_res[0])


# ── 3. Write Throughput ──────────────────────────────────────────────────────

def bench_write():
    print_header("3. Write Throughput (vector storage)")
    
    dim = 128
    
    for n in [10_000, 50_000]:
        data = np.random.randn(n, dim).astype(np.float32)
        
        # Rust write
        def rust_write():
            tmp = tempfile.mkdtemp()
            try:
                engine = lynse_core.DatabaseEngine(tmp)
                coll = engine.get_collection("bench", dim, 100_000)
                coll.add_items(data, None)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        
        rust_res = bench(rust_write, warmup=1, runs=5)
        
        # Python write baseline: np.save (raw numpy I/O)
        def py_write():
            tmp = tempfile.mkdtemp()
            try:
                np.save(os.path.join(tmp, "data.npy"), data)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        py_res = bench(py_write, warmup=1, runs=5)
        
        name = f"Write {n:,}x{dim}"
        print_row(name, rust_res, py_res)
        record(name, rust_res[0], py_res[0])


# ── 4. Full Collection Search (Flat) ─────────────────────────────────────────

def bench_collection_search():
    print_header("4. Collection Search (Flat brute-force)")
    
    dim = 128
    n = 50_000
    k = 10
    
    data = np.random.randn(n, dim).astype(np.float32)
    query = np.random.randn(dim).astype(np.float32)
    
    # Rust: full engine search
    tmp_rust = tempfile.mkdtemp()
    try:
        engine = lynse_core.DatabaseEngine(tmp_rust)
        coll = engine.get_collection("bench", dim, 100_000)
        coll.add_items(data, None)
        
        rust_res = bench(
            lambda: coll.search(query, k, None, 10),
            warmup=3, runs=20,
        )
    finally:
        shutil.rmtree(tmp_rust, ignore_errors=True)
    
    # Python: simsimd + argpartition
    py_res = bench(
        lambda: py_ip_search(query, data, k),
        warmup=3, runs=20,
    )
    
    name = f"Search {n:,}x{dim} k={k}"
    print_row(name, rust_res, py_res)
    record(name, rust_res[0], py_res[0])


# ── 5. HNSW Index Build + Search ─────────────────────────────────────────────

def bench_hnsw():
    print_header("5. HNSW Index (Rust only — no Python HNSW)")
    
    dim = 128
    
    for n in [5_000, 10_000]:
        data = np.random.randn(n, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)
        
        tmp = tempfile.mkdtemp()
        try:
            engine = lynse_core.DatabaseEngine(tmp)
            coll = engine.get_collection("bench", dim, 100_000)
            coll.add_items(data, None)
            
            # Build
            gc.disable()
            t0 = time.perf_counter()
            coll.build_index("HNSW-IP")
            build_ms = (time.perf_counter() - t0) * 1000
            gc.enable()
            
            # Search
            search_res = bench(
                lambda: coll.search(query, 10, None, 10),
                warmup=5, runs=50,
            )
            
            print(f"\n  HNSW-IP n={n:,} dim={dim}")
            print(f"    Build:  {build_ms:>10.1f}ms")
            print(f"    Search: {fmt(*search_res)}")
            record(f"HNSW build {n:,}", build_ms, float('nan'))
            record(f"HNSW search {n:,}", search_res[0], float('nan'))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── 6. Batch Search ──────────────────────────────────────────────────────────

def bench_batch_search():
    print_header("6. Batch Search (10 queries)")
    
    dim = 128
    n = 50_000
    n_queries = 10
    k = 10
    
    data = np.random.randn(n, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Rust: batch_search (parallel via rayon)
    tmp = tempfile.mkdtemp()
    try:
        engine = lynse_core.DatabaseEngine(tmp)
        coll = engine.get_collection("bench", dim, 100_000)
        coll.add_items(data, None)
        
        rust_res = bench(
            lambda: coll.batch_search(queries, k, None, 10),
            warmup=3, runs=20,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    
    # Python: sequential loop (simsimd + argpartition)
    def py_batch():
        results = []
        for i in range(n_queries):
            results.append(py_ip_search(queries[i], data, k))
        return results
    
    py_res = bench(py_batch, warmup=3, runs=20)
    
    name = f"Batch search {n_queries}x{n:,} k={k}"
    print_row(name, rust_res, py_res)
    record(name, rust_res[0], py_res[0])


# ── 7. BitSet Performance ────────────────────────────────────────────────────

def bench_bitset():
    print_header("7. BitSet Operations (Python only — Rust BitSet not yet exposed to PyO3)")
    
    try:
        from bitarray import bitarray
    except ImportError:
        print(f"  [SKIP] bitarray not installed")
        return
    
    # Inline Python BitSet for benchmark
    class PyBitSet:
        def __init__(self, size, fill=0):
            self._bits = bitarray(size)
            self._bits.setall(fill)
        def set_bit(self, i):
            self._bits[i] = 1
        def count(self):
            return self._bits.count()
    
    size = 1_000_000
    
    py_bs = PyBitSet(size=size, fill=0)
    for i in range(0, size, 100):
        py_bs.set_bit(i)
    
    py_res = bench(
        lambda: py_bs.count(),
        warmup=10, runs=100,
    )
    
    print(f"\n  Python BitSet count (n={size:,}, 10k set bits)")
    print(f"    {fmt(*py_res)}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print_header("SUMMARY TABLE")
    print(f"  {'Benchmark':<40} {'Rust (ms)':>12} {'Python (ms)':>12} {'Winner':>20}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*20}")
    for name, rust_ms, py_ms in results_table:
        if np.isnan(py_ms):
            py_str = "N/A"
            w = "Rust only"
        else:
            py_str = f"{py_ms:.3f}"
            w = winner_str(rust_ms, py_ms)
        rust_str = f"{rust_ms:.3f}"
        print(f"  {name:<40} {rust_str:>12} {py_str:>12} {w:>20}")


if __name__ == "__main__":
    print("=" * 70)
    print("  LynseDB: Rust vs Python Backend Benchmark")
    print(f"  Platform: {sys.platform} | NumPy {np.__version__}")
    print(f"  Rust backend: lynse_core loaded")
    print("=" * 70)
    
    np.random.seed(42)
    
    bench_distance()
    bench_topk()
    bench_write()
    bench_collection_search()
    bench_hnsw()
    bench_batch_search()
    
    # BitSet benchmark — skip if Rust BitSet not exposed to Python yet
    try:
        bench_bitset()
    except Exception:
        pass
    
    print_summary()
