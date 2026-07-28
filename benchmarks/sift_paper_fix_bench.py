#!/usr/bin/env python3
"""Paper-fix matrix on ANN_SIFT1M (or sift_learn) with aligned search budgets.

Loads vectors from a SIFT directory (fvecs/ivecs), builds LynseDB collections,
and reports build/search latency + recall@k for ``SIFT_PAPER_FIX_INDEX_MODES``.

Official ``sift_groundtruth.ivecs`` is used for L2 only when the full 1M base is
loaded; otherwise exact refs are computed via FLAT-* rebuilds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from gate_index_modes import (  # noqa: E402
    SIFT_PAPER_FIX_INDEX_MODES,
    exact_mode_for_metric,
    metric_for_mode,
    n_clusters_for,
    recall_floor_for_mode,
)
from sift_io import load_sift_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sift-dir",
        type=Path,
        default=Path(os.environ.get("GATE_SIFT_DIR", "sift")),
    )
    p.add_argument("--rows", type=int, default=int(os.environ.get("GATE_ROWS", "100000")))
    p.add_argument("--k", type=int, default=int(os.environ.get("GATE_K", "10")))
    p.add_argument("--warmups", type=int, default=int(os.environ.get("GATE_WARMUPS", "2")))
    p.add_argument("--trials", type=int, default=int(os.environ.get("GATE_TRIALS", "20")))
    p.add_argument(
        "--batch-queries",
        type=int,
        default=int(os.environ.get("GATE_BATCH_QUERIES", "8")),
    )
    p.add_argument("--nprobe", type=int, default=int(os.environ.get("GATE_NPROBE", "64")))
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("GATE_BATCH_SIZE", "50000")),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("GATE_MATRIX_DIR", "/tmp/lynse_sift_paper_fix")),
    )
    p.add_argument(
        "--modes",
        default=os.environ.get("GATE_INDEX_MODES", ""),
        help="Comma-separated subset; default = SIFT_PAPER_FIX_INDEX_MODES",
    )
    p.add_argument("--side", default=os.environ.get("GATE_SIDE", "self"))
    p.add_argument("--git-ref", default=os.environ.get("GATE_GIT_REF", "local"))
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def timed_ms(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def recall_at_k(got: list[int], expected: list[int], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(got[:k]).intersection(expected[:k])) / float(min(k, len(expected)))


def result_ids(result) -> list[int]:
    ids = result.ids
    if hasattr(ids, "tolist"):
        return [int(x) for x in ids.tolist()]
    return [int(x) for x in ids]


def make_fields(start: int, size: int) -> list[dict[str, Any]]:
    return [
        {
            "order": row_id,
            "bucket": row_id % 1000,
            "tag": "a" if row_id % 2 == 0 else "b",
        }
        for row_id in range(start, start + size)
    ]


def configured_modes(raw: str) -> list[str]:
    if raw.strip():
        return [m.strip().upper() for m in raw.split(",") if m.strip()]
    return list(SIFT_PAPER_FIX_INDEX_MODES)


def ingest_dense(coll, base: np.ndarray, batch_size: int) -> None:
    n, _dim = base.shape
    inserted = 0
    while inserted < n:
        end = min(inserted + batch_size, n)
        ids = list(range(inserted, end))
        coll.add(
            ids=ids,
            vectors=base[inserted:end],
            fields=make_fields(inserted, end - inserted),
            batch_size=end - inserted,
        )
        inserted = end
        if inserted % max(batch_size, 1) == 0 or inserted == n:
            print(f"  dense ingest {inserted:,}/{n:,}", flush=True)
    coll.commit()


def build_refs(
    coll,
    queries: np.ndarray,
    metrics: set[str],
    *,
    k: int,
    nprobe: int,
    n_clusters: int,
    official_l2_gt: np.ndarray | None,
) -> dict[str, list[list[int]]]:
    refs: dict[str, list[list[int]]] = {}
    for metric in sorted(metrics):
        if metric == "L2" and official_l2_gt is not None:
            refs[metric] = [official_l2_gt[i, :k].astype(int).tolist() for i in range(len(queries))]
            print(f"  refs[{metric}]: official SIFT groundtruth", flush=True)
            continue
        mode = exact_mode_for_metric(metric)
        build_args = {}
        if mode.upper().startswith(("IVF", "SPANN")):
            build_args["n_clusters"] = n_clusters
        coll.build_index(mode, **build_args)
        refs[metric] = [
            result_ids(coll.search(query, k=k, nprobe=nprobe)) for query in queries
        ]
        print(f"  refs[{metric}]: via {mode}", flush=True)
    return refs


def run_one_mode(
    coll,
    mode: str,
    queries: np.ndarray,
    refs: dict[str, list[list[int]]],
    args: argparse.Namespace,
    n_clusters: int,
) -> dict[str, Any]:
    metric = metric_for_mode(mode)
    where = '"bucket" < 100'
    batch = queries[: min(args.batch_queries, len(queries))]
    row: dict[str, Any] = {
        "mode": mode,
        "metric": metric,
        "min_recall": recall_floor_for_mode(mode),
    }
    try:
        build_args: dict[str, Any] = {}
        if mode.upper().startswith(("IVF", "SPANN")):
            build_args["n_clusters"] = n_clusters
        t0 = time.perf_counter()
        coll.build_index(mode, **build_args)
        row["build_ms"] = (time.perf_counter() - t0) * 1000.0

        for _ in range(args.warmups):
            coll.search(queries[0], k=args.k, nprobe=args.nprobe)

        search_samples = []
        filter_samples = []
        batch_samples = []
        recalls = []
        filter_bad = 0
        for i in range(args.trials):
            q = queries[i % len(queries)]
            search_samples.append(
                timed_ms(lambda q=q: coll.search(q, k=args.k, nprobe=args.nprobe))
            )
            got = result_ids(coll.search(q, k=args.k, nprobe=args.nprobe))
            recalls.append(recall_at_k(got, refs[metric][i % len(refs[metric])], args.k))
            filter_samples.append(
                timed_ms(
                    lambda q=q: coll.search(q, k=args.k, where=where, nprobe=args.nprobe)
                )
            )
            filtered = result_ids(
                coll.search(q, k=args.k, where=where, nprobe=args.nprobe)
            )
            filter_bad += sum(1 for item_id in filtered if int(item_id) % 1000 >= 100)
            batch_samples.append(
                timed_ms(
                    lambda: [
                        coll.search(qq, k=args.k, nprobe=args.nprobe) for qq in batch
                    ]
                )
            )

        row.update(
            {
                "status": "ok",
                "search_p50_ms": percentile(search_samples, 50),
                "filter_p50_ms": percentile(filter_samples, 50),
                "batch_p50_ms": percentile(batch_samples, 50),
                "recall_at_k": float(statistics.mean(recalls)) if recalls else 0.0,
                "filter_bad_ids": filter_bad,
            }
        )
        print(
            f"[mode] {mode:28s} build={row['build_ms']:10.1f}ms "
            f"search={row['search_p50_ms']:8.3f} recall={row['recall_at_k']:.3f}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[mode] {mode:28s} ERROR {row['error']}", flush=True)
    return row


def main() -> int:
    args = parse_args()
    os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("RAYON_NUM_THREADS", "4"))
    modes = configured_modes(args.modes)

    print("=" * 100, flush=True)
    print(
        f"sift_paper_fix_bench side={args.side} rows={args.rows:,} "
        f"modes={len(modes)} sift={args.sift_dir}",
        flush=True,
    )
    print("=" * 100, flush=True)

    ds = load_sift_dataset(
        args.sift_dir,
        rows=args.rows,
        query_limit=max(args.trials, args.batch_queries, 50),
    )
    base: np.ndarray = ds["base"]
    queries: np.ndarray = ds["queries"]
    official_gt = ds["groundtruth"]
    dim = ds["dim"]
    n = ds["n_base"]
    n_clusters = n_clusters_for(n)

    if args.data_dir.exists():
        shutil.rmtree(args.data_dir)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    import lynse

    client = lynse.VectorDBClient(str(args.data_dir))
    db = client.create_database("sift_db", drop_if_exists=True)
    coll = db.require_collection("dense_vectors", dim=dim, drop_if_exists=True, default_index=None)
    print(f"Ingesting SIFT base from {ds['base_path']} ({n:,} x {dim})...", flush=True)
    ingest_dense(coll, base, args.batch_size)

    metrics = {metric_for_mode(m) for m in modes}
    print(f"Building exact refs for metrics={sorted(metrics)}", flush=True)
    refs = build_refs(
        coll,
        queries,
        metrics,
        k=args.k,
        nprobe=args.nprobe,
        n_clusters=n_clusters,
        official_l2_gt=official_gt,
    )

    payload: dict[str, Any] = {
        "schema_version": 3,
        "side": args.side,
        "git_ref": args.git_ref,
        "dataset": {
            "name": "sift",
            "sift_dir": str(args.sift_dir),
            "base_path": ds["base_path"],
            "n_base": n,
            "dim": dim,
            "n_queries": int(queries.shape[0]),
            "official_l2_gt": official_gt is not None,
        },
        "profile": {
            "rows": n,
            "dim": dim,
            "k": args.k,
            "warmups": args.warmups,
            "trials": args.trials,
            "batch_queries": args.batch_queries,
            "nprobe": args.nprobe,
            "n_clusters": n_clusters,
            "modes": modes,
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
            "dataset": "sift",
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "modes": {},
    }

    for mode in modes:
        payload["modes"][mode] = run_one_mode(coll, mode, queries, refs, args, n_clusters)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
