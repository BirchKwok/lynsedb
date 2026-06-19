"""Reproducible existing-vector upsert benchmark.

The benchmark uses the Rust backend directly and measures journaled positional
updates after a committed initial load. It covers both single-row and small
batch updates, the two cases that previously triggered a full vector rewrite.
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
    parser.add_argument("--ingest-batch-size", type=int, default=100_000)
    parser.add_argument("--update-sizes", default="1,100")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.10)
    parser.add_argument("--keep-data", action="store_true")
    return parser.parse_args()


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def bench(args):
    import lynse._core as rb

    update_sizes = [int(value) for value in args.update_sizes.split(",")]
    if not update_sizes or any(size <= 0 or size > args.rows for size in update_sizes):
        raise ValueError("update sizes must be between 1 and the row count")

    test_dir = Path(tempfile.gettempdir()) / "lynse_upsert_bench"
    shutil.rmtree(test_dir, ignore_errors=True)

    rng = np.random.default_rng(args.seed)
    manager = rb.DatabaseManager(str(test_dir))
    manager.create_database("bench_db")
    manager.require_collection("bench_db", "bench_vectors", args.dim)
    collection = manager.get_collection("bench_db", "bench_vectors", args.dim)

    print(f"Writing {args.rows} vectors (dim={args.dim})...")
    for start in range(0, args.rows, args.ingest_batch_size):
        end = min(start + args.ingest_batch_size, args.rows)
        vectors = rng.random((end - start, args.dim), dtype=np.float32)
        collection.add_items(vectors, list(range(start, end)))
    collection.commit()

    updates = {}
    for update_size in update_sizes:
        samples = []
        for trial in range(args.trials):
            start = (trial * 7919) % (args.rows - update_size + 1)
            ids = list(range(start, start + update_size))
            vectors = np.full(
                (update_size, args.dim), (trial + 1) / 10, dtype=np.float32
            )
            started = time.perf_counter()
            collection.update_items(ids, vectors, None)
            samples.append((time.perf_counter() - started) * 1000)
        updates[str(update_size)] = summarize(samples)

    result = {
        "schema_version": 1,
        "rows": args.rows,
        "dimension": args.dim,
        "trials": args.trials,
        "seed": args.seed,
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "updates": updates,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    exit_code = 0
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        for update_size, current in updates.items():
            previous = baseline["updates"].get(update_size)
            if previous is None:
                continue
            change = current["median_ms"] / previous["median_ms"] - 1.0
            print(f"Update size {update_size} median change: {change:+.2%}")
            if change > args.max_regression:
                exit_code = 1

    if not args.keep_data:
        shutil.rmtree(test_dir, ignore_errors=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(bench(parse_args()))
