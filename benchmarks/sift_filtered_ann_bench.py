#!/usr/bin/env python3
"""Validate filtered HNSW / DiskANN in-graph search on ANN_SIFT1M-style data.

Compares FLAT (exact filtered reference) vs HNSW / DiskANN under several
metadata selectivities. Reports latency and filtered recall@k.

Paths come only from ``--sift-dir`` / ``GATE_SIFT_DIR`` (and related env vars).
Do not hard-code machine-local dataset paths in this file.

Example:
    GATE_SIFT_DIR=/path/to/sift GATE_ROWS=100000 \\
      python benchmarks/sift_filtered_ann_bench.py --output /tmp/sift_filtered.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from sift_io import load_sift_dataset  # noqa: E402


DEFAULT_MODES = ("FLAT-L2", "HNSW-L2", "DISKANN-L2")
DEFAULT_SELECTIVITIES = (0.0, 0.01, 0.05, 0.10, 0.50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sift-dir",
        type=Path,
        default=Path(os.environ.get("GATE_SIFT_DIR", "sift")),
        help="Directory with sift_*.fvecs / sift_groundtruth.ivecs",
    )
    p.add_argument("--rows", type=int, default=int(os.environ.get("GATE_ROWS", "100000")))
    p.add_argument("--queries", type=int, default=int(os.environ.get("GATE_QUERIES", "100")))
    p.add_argument("--k", type=int, default=int(os.environ.get("GATE_K", "10")))
    p.add_argument("--warmups", type=int, default=int(os.environ.get("GATE_WARMUPS", "3")))
    p.add_argument("--trials", type=int, default=int(os.environ.get("GATE_TRIALS", "20")))
    p.add_argument(
        "--nprobe",
        type=int,
        default=int(os.environ.get("GATE_NPROBE", "64")),
        help="Query-time HNSW ef / DiskANN beam override (0 = index default)",
    )
    p.add_argument("--diskann-r", type=int, default=None)
    p.add_argument("--diskann-l", type=int, default=None)
    p.add_argument("--diskann-alpha", type=float, default=None)
    p.add_argument("--diskann-max-degree", type=int, default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("GATE_BATCH_SIZE", "50000")),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("GATE_MATRIX_DIR", "/tmp/lynse_sift_filtered_ann")),
    )
    p.add_argument(
        "--modes",
        default=os.environ.get("GATE_INDEX_MODES", ",".join(DEFAULT_MODES)),
        help="Comma-separated index modes",
    )
    p.add_argument(
        "--selectivities",
        default=os.environ.get(
            "GATE_SELECTIVITIES",
            ",".join(str(x) for x in DEFAULT_SELECTIVITIES),
        ),
        help="Comma-separated filter fractions; 0 means unfiltered",
    )
    p.add_argument(
        "--min-ann-recall",
        type=float,
        default=None,
        help="Fail when any non-FLAT row has recall@k below this threshold",
    )
    p.add_argument(
        "--max-short-results",
        type=int,
        default=0,
        help="Maximum total queries allowed to return fewer than k results",
    )
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def recall_at_k(got: list[int], expected: list[int], k: int) -> float:
    if not expected:
        return 1.0 if not got else 0.0
    return len(set(got[:k]).intersection(expected[:k])) / float(min(k, len(expected)))


def result_ids(result) -> list[int]:
    ids = result.ids
    if hasattr(ids, "tolist"):
        return [int(x) for x in ids.tolist()]
    return [int(x) for x in ids]


def make_fields(n: int) -> list[dict[str, Any]]:
    # bucket in [0, 999] → where "bucket" < T matches about T/1000 of rows.
    return [{"bucket": i % 1000, "order": i} for i in range(n)]


def where_for_selectivity(sel: float) -> str | None:
    if sel <= 0.0:
        return None
    threshold = max(1, int(round(sel * 1000)))
    return f'"bucket" < {threshold}'


def expected_match_frac(sel: float) -> float:
    if sel <= 0.0:
        return 1.0
    return max(1, int(round(sel * 1000))) / 1000.0


def ingest(coll, base: np.ndarray, batch_size: int) -> None:
    n, _ = base.shape
    fields = make_fields(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        coll.add(
            vectors=base[start:end],
            ids=list(range(start, end)),
            fields=fields[start:end],
            batch_size=end - start,
        )
    coll.commit()


def search_once(coll, query: np.ndarray, k: int, where: str | None, nprobe: int):
    kwargs: dict[str, Any] = {"k": k}
    if where is not None:
        kwargs["where"] = where
    if nprobe > 0:
        kwargs["nprobe"] = nprobe
    return coll.search(query, **kwargs)


def filter_ok(ids: list[int], where: str | None, n: int) -> bool:
    if where is None:
        return True
    # where is always '"bucket" < T'
    threshold = int(where.rsplit("<", 1)[1].strip())
    return all((i % 1000) < threshold for i in ids if 0 <= i < n)


def measure_mode(
    coll,
    *,
    queries: np.ndarray,
    k: int,
    where: str | None,
    nprobe: int,
    warmups: int,
    trials: int,
    refs: list[list[int]],
    n_rows: int,
) -> dict[str, Any]:
    for i in range(min(warmups, len(queries))):
        search_once(coll, queries[i], k, where, nprobe)

    latencies_ms: list[float] = []
    recalls: list[float] = []
    leaks = 0
    shortfalls = 0

    n_eval = min(trials, len(queries), len(refs))
    for i in range(n_eval):
        t0 = time.perf_counter()
        result = search_once(coll, queries[i], k, where, nprobe)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        got = result_ids(result)
        if len(got) < min(k, len(refs[i])):
            shortfalls += 1
        if not filter_ok(got, where, n_rows):
            leaks += 1
        recalls.append(recall_at_k(got, refs[i], k))

    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    return {
        "n_eval": n_eval,
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "qps": (1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        "recall_at_k_mean": float(np.mean(recalls)) if recalls else 0.0,
        "recall_at_k_p50": percentile(recalls, 50),
        "filter_leaks": leaks,
        "result_shortfalls": shortfalls,
    }


def build_exact_refs(
    ref_coll,
    queries: np.ndarray,
    k: int,
    where: str | None,
    nprobe: int,
) -> list[list[int]]:
    refs: list[list[int]] = []
    for i in range(len(queries)):
        refs.append(result_ids(search_once(ref_coll, queries[i], k, where, nprobe)))
    return refs


def format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "index",
        "selectivity",
        "where",
        "p50_ms",
        "p95_ms",
        "qps",
        "recall@k",
        "leaks",
        "short",
        "build_s",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        sel = r["selectivity"]
        sel_s = "none" if sel <= 0 else f"{sel:.0%}"
        where = r["where"] or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    r["index_mode"],
                    sel_s,
                    where,
                    f"{r['latency_p50_ms']:.3f}",
                    f"{r['latency_p95_ms']:.3f}",
                    f"{r['qps']:.1f}",
                    f"{r['recall_at_k_mean']:.4f}",
                    str(r["filter_leaks"]),
                    str(r["result_shortfalls"]),
                    f"{r['build_s']:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    sift_dir = args.sift_dir.expanduser().resolve()
    if not sift_dir.is_dir():
        print(
            f"SIFT dir not found: {sift_dir} (set --sift-dir or GATE_SIFT_DIR)",
            file=sys.stderr,
        )
        return 2

    modes = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    selectivities = [float(x) for x in args.selectivities.split(",") if x.strip()]

    print(
        f"Loading SIFT rows={args.rows} queries={args.queries} "
        f"from env/CLI sift-dir (resolved exists={sift_dir.is_dir()})",
        flush=True,
    )
    ds = load_sift_dataset(sift_dir, rows=args.rows, query_limit=args.queries)
    base = ds["base"]
    queries = ds["queries"]
    n, dim = base.shape
    print(
        f"Loaded base={n}x{dim} queries={queries.shape[0]} "
        f"source={Path(ds['base_path']).name}",
        flush=True,
    )

    from lynse import VectorDBClient

    if args.data_dir.exists():
        shutil.rmtree(args.data_dir)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    client = VectorDBClient(uri=str(args.data_dir))
    db = client.create_database("sift_filtered", drop_if_exists=True)

    # Shared collection for FLAT reference + each ANN rebuild.
    coll = db.require_collection(
        "sift",
        dim=dim,
        drop_if_exists=True,
        default_index=None,
    )
    t_ingest0 = time.perf_counter()
    ingest(coll, base, args.batch_size)
    ingest_s = time.perf_counter() - t_ingest0
    print(f"Ingest done in {ingest_s:.2f}s", flush=True)

    # Exact filtered / unfiltered references from FLAT-L2.
    print("Building FLAT-L2 reference index...", flush=True)
    t0 = time.perf_counter()
    coll.build_index("FLAT-L2")
    flat_build_s = time.perf_counter() - t0
    print(f"FLAT-L2 build {flat_build_s:.2f}s", flush=True)

    ref_cache: dict[float, list[list[int]]] = {}
    for sel in selectivities:
        where = where_for_selectivity(sel)
        print(f"Computing FLAT refs selectivity={sel} where={where!r}...", flush=True)
        ref_cache[sel] = build_exact_refs(
            coll, queries[: args.trials], args.k, where, args.nprobe
        )

    table_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for mode in modes:
        if mode != "FLAT-L2":
            print(f"Building {mode}...", flush=True)
            t0 = time.perf_counter()
            build_kwargs: dict[str, Any] = {}
            if mode.startswith("DISKANN"):
                for key, value in (
                    ("r", args.diskann_r),
                    ("l", args.diskann_l),
                    ("alpha", args.diskann_alpha),
                    ("max_degree", args.diskann_max_degree),
                ):
                    if value is not None:
                        build_kwargs[key] = value
            coll.build_index(mode, **build_kwargs)
            build_s = time.perf_counter() - t0
            print(f"{mode} build {build_s:.2f}s", flush=True)
        else:
            build_s = flat_build_s

        for sel in selectivities:
            where = where_for_selectivity(sel)
            stats = measure_mode(
                coll,
                queries=queries,
                k=args.k,
                where=where,
                nprobe=args.nprobe,
                warmups=args.warmups,
                trials=args.trials,
                refs=ref_cache[sel],
                n_rows=n,
            )
            row = {
                "index_mode": mode,
                "selectivity": sel,
                "expected_match_frac": expected_match_frac(sel),
                "where": where,
                "build_s": build_s,
                "k": args.k,
                "nprobe": args.nprobe,
                **stats,
            }
            table_rows.append(row)
            results.append(row)
            print(
                f"  {mode:12s} sel={sel:4.2f} p50={stats['latency_p50_ms']:.3f}ms "
                f"recall={stats['recall_at_k_mean']:.4f} "
                f"leaks={stats['filter_leaks']} short={stats['result_shortfalls']}",
                flush=True,
            )

    table = format_table(table_rows)
    print("\n=== SIFT filtered ANN results ===\n")
    print(table)
    print()

    payload = {
        "schema_version": 1,
        "dataset": {
            "n_base": n,
            "dim": dim,
            "n_queries_loaded": int(queries.shape[0]),
            "base_file": Path(ds["base_path"]).name,
            "rows_requested": args.rows,
        },
        "config": {
            "k": args.k,
            "warmups": args.warmups,
            "trials": args.trials,
            "nprobe": args.nprobe,
            "diskann_build": {
                "r": args.diskann_r,
                "l": args.diskann_l,
                "alpha": args.diskann_alpha,
                "max_degree": args.diskann_max_degree,
            },
            "modes": modes,
            "selectivities": selectivities,
            "ingest_s": ingest_s,
            "acceptance": {
                "min_ann_recall": args.min_ann_recall,
                "max_short_results": args.max_short_results,
                "max_filter_leaks": 0,
            },
        },
        "rows": results,
        "markdown_table": table,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.output}", flush=True)

    leaks = sum(r["filter_leaks"] for r in results)
    if leaks:
        print(f"FAIL: filter leaks={leaks}", file=sys.stderr)
        return 1
    short_results = sum(r["result_shortfalls"] for r in results)
    if short_results > args.max_short_results:
        print(
            f"FAIL: short results={short_results} (max {args.max_short_results})",
            file=sys.stderr,
        )
        return 1
    if args.min_ann_recall is not None:
        failed_recall = [
            r
            for r in results
            if not r["index_mode"].upper().startswith("FLAT-")
            and r["recall_at_k_mean"] < args.min_ann_recall
        ]
        if failed_recall:
            details = ", ".join(
                f"{r['index_mode']}@{r['selectivity']:.0%}={r['recall_at_k_mean']:.4f}"
                for r in failed_recall
            )
            print(
                f"FAIL: ANN recall below {args.min_ann_recall:.4f}: {details}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
