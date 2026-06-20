"""Reproducible flat-search performance benchmark.

The benchmark uses the Rust backend directly so Python collection buffering does
not affect either ingest or search measurements.
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--index-mode", default="FLAT-IP")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.05)
    parser.add_argument("--keep-data", action="store_true")
    return parser.parse_args()


def percentile(sorted_values, fraction):
    index = min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    return sorted_values[index]


def bench(args):
    import lynse._core as rb

    test_dir = Path(tempfile.gettempdir()) / "lynse_flat_bench"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    rng = np.random.default_rng(args.seed)
    binary_metric = any(
        token in args.index_mode.upper()
        for token in ("HAMMING", "JACCARD", "TANIMOTO", "DICE")
    )
    query = (
        rng.integers(0, 2, size=args.dim).astype(np.float32)
        if binary_metric
        else rng.random(args.dim, dtype=np.float32)
    )

    mgr = rb.DatabaseManager(str(test_dir))
    mgr.create_database("bench_db")
    mgr.require_collection("bench_db", "bench_vectors", args.dim)
    coll = mgr.get_collection("bench_db", "bench_vectors", args.dim)

    print(f"Writing {args.rows} vectors (dim={args.dim})...")
    ingest_started = time.perf_counter()
    for start in range(0, args.rows, args.batch_size):
        end = min(start + args.batch_size, args.rows)
        vectors = (
            rng.integers(0, 2, size=(end - start, args.dim)).astype(np.float32)
            if binary_metric
            else rng.random((end - start, args.dim), dtype=np.float32)
        )
        if start == 0:
            vectors[0] = query
        coll.add_items(vectors, list(range(start, end)))
        print(f"  Written {end}/{args.rows}")

    coll.commit()
    coll.build_index(args.index_mode, None)
    ingest_seconds = time.perf_counter() - ingest_started
    print(f"Shape: {coll.shape()}")

    print(f"\nWarmup searches ({args.warmups})...")
    for _ in range(args.warmups):
        coll.search(query, args.k, None, 10)

    print(f"Benchmarking {args.trials} searches (k={args.k})...")
    times_ms = []
    for _ in range(args.trials):
        started = time.perf_counter()
        coll.search(query, args.k, None, 10)
        times_ms.append((time.perf_counter() - started) * 1000)

    times_ms.sort()
    collection_dir = test_dir / "bench_db" / "bench_vectors"
    vector_manifest = collection_dir / "vector_manifest.json"
    if vector_manifest.exists():
        segment_count = len(json.loads(vector_manifest.read_text())["segments"])
    else:
        segment_count = 1 if args.rows else 0
    result = {
        "schema_version": 1,
        "rows": args.rows,
        "dimension": args.dim,
        "k": args.k,
        "index_mode": args.index_mode.upper(),
        "seed": args.seed,
        "warmups": args.warmups,
        "trials": args.trials,
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
        "segment_count": segment_count,
        "source_vector_bytes": args.rows * args.dim * 4,
        "estimated_hot_scan_bytes": (
            args.rows * ((args.dim + 63) // 64) * 8
            if binary_metric
            else args.rows * args.dim * 4
        ),
        "compatibility_file_created": (collection_dir / "vectors.compat.bin").exists(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ingest_seconds": ingest_seconds,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "p10_ms": percentile(times_ms, 0.10),
        "p90_ms": percentile(times_ms, 0.90),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }

    print("\n=== Results ===")
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    exit_code = 0
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        search_ratio = result["median_ms"] / baseline["median_ms"] - 1.0
        ingest_ratio = result["ingest_seconds"] / baseline["ingest_seconds"] - 1.0
        result["median_regression"] = search_ratio
        result["ingest_regression"] = ingest_ratio
        print(f"Median change from baseline: {search_ratio:+.2%}")
        print(f"Ingest change from baseline: {ingest_ratio:+.2%}")
        if search_ratio > args.max_regression:
            print(
                f"Regression exceeds allowed {args.max_regression:.2%}: "
                f"{result['median_ms']:.3f} ms vs {baseline['median_ms']:.3f} ms"
            )
            exit_code = 1
        if ingest_ratio > args.max_regression:
            print(
                f"Ingest regression exceeds allowed {args.max_regression:.2%}: "
                f"{result['ingest_seconds']:.3f} s vs "
                f"{baseline['ingest_seconds']:.3f} s"
            )
            exit_code = 1

    if not args.keep_data:
        shutil.rmtree(test_dir, ignore_errors=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(bench(parse_args()))
