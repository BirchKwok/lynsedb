#!/usr/bin/env python3
"""Run DISKANN-IP on SIFT 100k three times with a fixed build seed; report variance.

Usage:
  LYNSE_DISKANN_SEED=42 python benchmarks/diskann_ip_seed_variance.py
  LYNSE_DISKANN_SEED=42 RAYON_NUM_THREADS=4 python benchmarks/diskann_ip_seed_variance.py --runs 3
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from sift_io import load_sift_dataset  # noqa: E402


def recall_at_k(got: list[int], expected: list[int], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(got[:k]).intersection(expected[:k])) / float(min(k, len(expected)))


def result_ids(result) -> list[int]:
    ids = result.ids
    if hasattr(ids, "tolist"):
        return [int(x) for x in ids.tolist()]
    return [int(x) for x in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sift-dir", type=Path, default=Path(os.environ.get("GATE_SIFT_DIR", "sift")))
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--mode", default="DISKANN-IP")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("LYNSE_DISKANN_SEED", "42")))
    args = ap.parse_args()

    os.environ["LYNSE_DISKANN_SEED"] = str(args.seed)
    print(f"LYNSE_DISKANN_SEED={os.environ['LYNSE_DISKANN_SEED']} RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', '')}")

    from lynse import VectorDBClient

    ds = load_sift_dataset(args.sift_dir, rows=args.rows, query_limit=args.queries)
    base = ds["base"]
    queries = ds["queries"][: args.queries]
    dim = int(ds["dim"])
    print(f"data={ds['base_path']} n={base.shape[0]} dim={dim} queries={len(queries)}")

    # Exact IP refs once (FLAT-IP).
    root_ref = Path(tempfile.mkdtemp(prefix="diskann_ip_var_ref_"))
    try:
        client = VectorDBClient(uri=str(root_ref))
        db = client.create_database("ref", drop_if_exists=True)
        coll = db.require_collection("c", dim=dim, drop_if_exists=True)
        coll.add(ids=list(range(len(base))), vectors=base, batch_size=50_000)
        coll.commit()
        coll.build_index("FLAT-IP")
        refs = [result_ids(coll.search(vector=q, k=args.k, nprobe=args.nprobe)) for q in queries]
        client.close()
    finally:
        shutil.rmtree(root_ref, ignore_errors=True)

    recalls = []
    p50s = []
    builds = []
    for run in range(args.runs):
        root = Path(tempfile.mkdtemp(prefix=f"diskann_ip_var_{run}_"))
        try:
            client = VectorDBClient(uri=str(root))
            db = client.create_database("db", drop_if_exists=True)
            coll = db.require_collection("c", dim=dim, drop_if_exists=True)
            coll.add(ids=list(range(len(base))), vectors=base, batch_size=50_000)
            coll.commit()

            t0 = time.perf_counter()
            coll.build_index(args.mode)
            build_ms = (time.perf_counter() - t0) * 1000.0

            for _ in range(args.warmups):
                coll.search(vector=queries[0], k=args.k, nprobe=args.nprobe)

            samples = []
            run_recalls = []
            for i in range(args.trials):
                q = queries[i % len(queries)]
                t0 = time.perf_counter()
                res = coll.search(vector=q, k=args.k, nprobe=args.nprobe)
                samples.append((time.perf_counter() - t0) * 1000.0)
                run_recalls.append(recall_at_k(result_ids(res), refs[i % len(queries)], args.k))

            # Full-query mean recall (all queries once).
            full = [
                recall_at_k(
                    result_ids(coll.search(vector=q, k=args.k, nprobe=args.nprobe)),
                    refs[i],
                    args.k,
                )
                for i, q in enumerate(queries)
            ]
            recall = float(np.mean(full))
            p50 = float(np.percentile(samples, 50))
            recalls.append(recall)
            p50s.append(p50)
            builds.append(build_ms)
            print(
                f"run={run} seed={args.seed} recall@10={recall:.4f} "
                f"p50={p50:.3f}ms build={build_ms:.0f}ms",
                flush=True,
            )
            client.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def summarize(name: str, xs: list[float]) -> None:
        mean = statistics.mean(xs)
        stdev = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(
            f"{name}: values={[round(x, 4) if name.startswith('recall') else round(x, 3) for x in xs]} "
            f"mean={mean:.4f} stdev={stdev:.4f} min={min(xs):.4f} max={max(xs):.4f}"
        )

    print("--- summary ---")
    summarize("recall@10", recalls)
    summarize("search_p50_ms", p50s)
    summarize("build_ms", builds)


if __name__ == "__main__":
    main()
