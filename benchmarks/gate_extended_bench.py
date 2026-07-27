#!/usr/bin/env python3
"""Extended local-gate scenarios for LynseDB.

Covers paths the core flat/upsert gate misses:
  - filtered FLAT-IP search (~10% selectivity)
  - batch search
  - commit flush latency
  - HNSW-IP search latency + recall@k vs FLAT-IP
  - delete/tombstone + search latency

Intended for machine-local gates only (not GitHub Actions).
"""

from __future__ import annotations

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-queries", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nprobe", type=int, default=10)
    parser.add_argument("--min-hnsw-recall", type=float, default=0.90)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def median_ms(samples: list[float]) -> float:
    return float(statistics.median(samples))


def timed_ms(fn) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000.0


def recall_at_k(got: list[int], expected: list[int], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(got[:k]).intersection(expected[:k])) / float(min(k, len(expected)))


def result_ids(result) -> list[int]:
    return [int(x) for x in result.ids()]


def bench(args: argparse.Namespace) -> dict:
    import lynse._core as rb

    test_dir = Path(tempfile.mkdtemp(prefix="lynse_gate_ext_"))
    try:
        rng = np.random.default_rng(args.seed)
        mgr = rb.DatabaseManager(str(test_dir))
        mgr.create_database("bench_db")
        mgr.require_collection("bench_db", "bench_vectors", args.dim)
        coll = mgr.get_collection("bench_db", "bench_vectors", args.dim)

        query = rng.random(args.dim, dtype=np.float32)
        queries = rng.random((args.batch_queries, args.dim), dtype=np.float32)
        queries[0] = query

        print(f"Writing {args.rows} vectors (dim={args.dim}) with fields...")
        for start in range(0, args.rows, args.batch_size):
            end = min(start + args.batch_size, args.rows)
            vectors = rng.random((end - start, args.dim), dtype=np.float32)
            if start == 0:
                vectors[0] = query
            ids = list(range(start, end))
            fields = [{"bucket": i % 1000, "tag": "a" if i % 2 == 0 else "b"} for i in ids]
            coll.add_items(vectors, ids, fields)

        commit_samples = []
        for _ in range(args.warmups):
            coll.commit()
        for _ in range(args.trials):
            commit_samples.append(timed_ms(coll.commit))

        coll.build_index("FLAT-IP", None)

        where = '"bucket" < 100'
        for _ in range(args.warmups):
            coll.search(query, args.k, where, args.nprobe)
            coll.batch_search(queries, args.k, None, args.nprobe)

        filter_samples = []
        batch_samples = []
        for trial in range(args.trials):
            q = queries[trial % len(queries)]
            filter_samples.append(
                timed_ms(lambda q=q: coll.search(q, args.k, where, args.nprobe))
            )
            batch_samples.append(
                timed_ms(lambda: coll.batch_search(queries, args.k, None, args.nprobe))
            )

        delete_ids = list(range(0, args.rows, 20))
        coll.delete_items(delete_ids)
        tombstone_samples = []
        for _ in range(args.warmups):
            coll.search(query, args.k, None, args.nprobe)
        for _ in range(args.trials):
            tombstone_samples.append(
                timed_ms(lambda: coll.search(query, args.k, None, args.nprobe))
            )
            hit_ids = result_ids(coll.search(query, args.k, None, args.nprobe))
            assert all(i not in delete_ids for i in hit_ids), hit_ids

        # Fresh collection for HNSW so tombstones do not pollute recall.
        mgr2 = rb.DatabaseManager(str(test_dir / "hnsw"))
        mgr2.create_database("bench_db")
        mgr2.require_collection("bench_db", "bench_vectors", args.dim)
        hnsw = mgr2.get_collection("bench_db", "bench_vectors", args.dim)
        for start in range(0, args.rows, args.batch_size):
            end = min(start + args.batch_size, args.rows)
            vectors = rng.random((end - start, args.dim), dtype=np.float32)
            if start == 0:
                vectors[0] = query
            hnsw.add_items(vectors, list(range(start, end)))
        hnsw.commit()
        hnsw.build_index("FLAT-IP", None)
        exact_refs = [
            result_ids(hnsw.search(queries[i], args.k, None, args.nprobe))
            for i in range(min(args.trials, len(queries)))
        ]
        build_started = time.perf_counter()
        hnsw.build_index("HNSW-IP", None)
        hnsw_build_ms = (time.perf_counter() - build_started) * 1000.0

        for _ in range(args.warmups):
            hnsw.search(query, args.k, None, args.nprobe)
        hnsw_samples = []
        recalls = []
        for trial in range(min(args.trials, len(queries))):
            q = queries[trial]
            hnsw_samples.append(
                timed_ms(lambda q=q: hnsw.search(q, args.k, None, args.nprobe))
            )
            got = result_ids(hnsw.search(q, args.k, None, args.nprobe))
            recalls.append(recall_at_k(got, exact_refs[trial], args.k))

        recall = float(statistics.mean(recalls)) if recalls else 0.0
        result = {
            "schema_version": 1,
            "rows": args.rows,
            "dimension": args.dim,
            "k": args.k,
            "batch_queries": args.batch_queries,
            "seed": args.seed,
            "warmups": args.warmups,
            "trials": args.trials,
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "filtered_median_ms": median_ms(filter_samples),
            "batch_median_ms": median_ms(batch_samples),
            "commit_median_ms": median_ms(commit_samples),
            "tombstone_search_median_ms": median_ms(tombstone_samples),
            "hnsw_build_ms": hnsw_build_ms,
            "hnsw_search_median_ms": median_ms(hnsw_samples),
            "hnsw_recall_at_k": recall,
            "min_hnsw_recall": args.min_hnsw_recall,
            "hnsw_recall_ok": recall >= args.min_hnsw_recall,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if not result["hnsw_recall_ok"]:
            raise SystemExit(
                f"HNSW recall {recall:.3f} below floor {args.min_hnsw_recall:.3f}"
            )
        return result
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    bench(parse_args())
