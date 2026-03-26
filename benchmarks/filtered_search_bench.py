"""
Benchmark: Pre-filter vs Post-filter strategies for conditional search.

Compares three approaches at various filter selectivities:
1. prefilter_copy:  field query → copy matching vectors → top_k on subset
2. prefilter_fused: field query → scan mmap in-place, skip non-matching (no copy)
3. postfilter:      full optimized FlatMmap scan (SIMD, parallel) → filter results

Usage:
    python benchmarks/filtered_search_bench.py
"""

import numpy as np
import time
import shutil
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynse import _core as lynse_core

# ─── Configuration ────────────────────────────────────────────────────────────

N_VECTORS = 500_000       # Total vectors
DIM = 128                 # Vector dimension
K = 10                    # Top-k results
WARMUP = 3                # Warmup iterations
ITERATIONS = 20           # Benchmark iterations
OVERSAMPLE = 100          # Oversample factor for post-filter

# Selectivity levels: what fraction of vectors match the filter
SELECTIVITIES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9]

DB_PATH = "/tmp/lynse_filter_bench"

# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_collection():
    """Create collection with vectors and field metadata."""
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    engine = lynse_core.DatabaseEngine(DB_PATH)
    coll = engine.get_collection("bench", DIM)

    print(f"Inserting {N_VECTORS:,} × {DIM} vectors (single batch for minimal ApexBase segments)...")
    t0 = time.time()

    # Generate random vectors
    np.random.seed(42)
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    # Normalize for IP metric
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Generate field metadata with an "order" column for filtering
    # "order" ranges from 0 to 999, so we can control selectivity
    fields = []
    for i in range(N_VECTORS):
        fields.append({"order": i % 1000, "category": f"cat_{i % 50}"})

    # Single batch insert → 1 ApexBase segment → fast field queries
    coll.add_items(vectors, fields)
    print(f"  Inserted {N_VECTORS:,} vectors in single batch")

    coll.commit()

    # Build flat index
    coll.build_index("FLAT-IP")

    elapsed = time.time() - t0
    print(f"Setup complete in {elapsed:.1f}s")
    print(f"Shape: {coll.shape()}")
    print()

    return engine, coll, vectors


def run_benchmark(coll, vectors):
    """Run benchmark across all selectivities."""
    query = vectors[0].copy()  # Use first vector as query

    print(f"{'Selectivity':>12} {'Matches':>8} | "
          f"{'field_query':>12} {'prefilter_copy':>15} {'prefilter_fused':>16} {'postfilter':>12} {'unfiltered':>12} | "
          f"{'Winner':>18}")
    print("-" * 140)

    for sel in SELECTIVITIES:
        # Build filter expression that matches ~sel fraction of vectors
        # "order" ranges 0..999, so order < X matches X/1000 of vectors
        threshold = int(sel * 1000)
        if threshold < 1:
            threshold = 1
        where_expr = f"\"order\" < {threshold}"

        expected_matches = int(N_VECTORS * threshold / 1000)

        try:
            results = coll.benchmark_filtered_search(
                query,
                k=K,
                where_expr=where_expr,
                oversample=OVERSAMPLE,
                warmup=WARMUP,
                iterations=ITERATIONS,
            )
        except Exception as e:
            print(f"{sel:>12.3f} {expected_matches:>8,} | ERROR: {e}")
            continue

        # Parse results
        timings = {}
        counts = {}
        for name, us, count in results:
            timings[name] = us
            counts[name] = count

        actual_matches = counts.get("field_query", 0)

        # Find winner (fastest search strategy)
        search_strategies = ["prefilter_copy", "prefilter_fused", "postfilter"]
        search_times = {s: timings.get(s, float('inf')) for s in search_strategies}
        winner = min(search_times, key=search_times.get)

        # Format output
        fq = timings.get("field_query", 0)
        s1 = timings.get("prefilter_copy", 0)
        s2 = timings.get("prefilter_fused", 0)
        s3 = timings.get("postfilter", 0)
        s4 = timings.get("unfiltered_mmap", 0)

        # Add result count check
        r1 = counts.get("prefilter_copy", 0)
        r2 = counts.get("prefilter_fused", 0)
        r3 = counts.get("postfilter", 0)

        print(f"{sel:>12.3f} {actual_matches:>8,} | "
              f"{fq:>10.0f}µs {s1:>13.0f}µs {s2:>14.0f}µs {s3:>10.0f}µs {s4:>10.0f}µs | "
              f"{'→ ' + winner:>18} "
              f"[{r1}/{r2}/{r3}]")

    print()
    print("Legend: prefilter_copy = copy matching vectors then search")
    print("        prefilter_fused = scan mmap in-place, skip non-matching")
    print("        postfilter = full optimized scan then filter results")
    print(f"        [r1/r2/r3] = result counts for each strategy (should all be {K})")
    print()

    # Also test with different oversample factors for post-filter
    print("\n=== Post-filter oversample sensitivity (selectivity=0.01) ===")
    where_expr = '"order" < 10'
    print(f"{'oversample':>12} {'postfilter_µs':>14} {'results':>8}")
    print("-" * 40)
    for os_factor in [10, 20, 50, 100, 200, 500, 1000]:
        try:
            results = coll.benchmark_filtered_search(
                query, k=K, where_expr=where_expr,
                oversample=os_factor, warmup=WARMUP, iterations=ITERATIONS,
            )
            for name, us, count in results:
                if name == "postfilter":
                    print(f"{os_factor:>12} {us:>12.0f}µs {count:>8}")
        except Exception as e:
            print(f"{os_factor:>12} ERROR: {e}")


def cleanup():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)


if __name__ == "__main__":
    print(f"=== Filtered Search Strategy Benchmark ===")
    print(f"N={N_VECTORS:,}, dim={DIM}, k={K}, warmup={WARMUP}, iterations={ITERATIONS}")
    print(f"Oversample factor for post-filter: {OVERSAMPLE}")
    print()

    try:
        engine, coll, vectors = setup_collection()
        run_benchmark(coll, vectors)
    finally:
        cleanup()
