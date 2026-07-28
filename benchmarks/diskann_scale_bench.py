#!/usr/bin/env python3
"""Scale bench for DiskANN (and IVF/HNSW baselines) at large N.

Usage:
  python3 benchmarks/diskann_scale_bench.py --rows 10000000 --dim 128
  python3 benchmarks/diskann_scale_bench.py --rows 1000000 --modes DISKANN-IP,IVF-IP,HNSW-IP
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _median_ms(fn, *, warm: int = 3, trials: int = 20) -> dict[str, float]:
    for _ in range(warm):
        fn()
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "p50_ms": statistics.median(samples),
        "p90_ms": samples[max(0, int(len(samples) * 0.9) - 1)],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def _ingest(coll, n: int, dim: int, seed: int, batch: int) -> None:
    rng = np.random.default_rng(seed)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        ids = list(range(start, end))
        vecs = rng.random((end - start, dim), dtype=np.float32)
        coll.add(ids=ids, vectors=vecs, batch_size=end - start)
        if start == 0 or end == n or (end // batch) % 20 == 0:
            print(f"  ingest {end}/{n} ({100.0 * end / n:.1f}%)", flush=True)
    coll.commit()
    print("  commit done", flush=True)


def _exact_topk(store_search, queries: list[np.ndarray], k: int) -> list[set[int]]:
    out = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        ids = set(int(x) for x in store_search(q, k))
        out.append(ids)
        print(
            f"  GT query {i + 1}/{len(queries)} in {(time.perf_counter() - t0) * 1000:.0f}ms",
            flush=True,
        )
    return out


def main() -> int:
    import lynse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--batch", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--modes",
        type=str,
        default="DISKANN-IP,IVF-IP,HNSW-IP",
        help="Comma-separated index modes",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Persistent data dir (default: temp under /tmp)",
    )
    ap.add_argument(
        "--skip-gt",
        action="store_true",
        help="Skip FLAT ground-truth recall (much faster at 10M)",
    )
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    data_root = Path(args.data_dir) if args.data_dir else Path(
        tempfile.mkdtemp(prefix=f"lynse_diskann_{args.rows}_")
    )
    data_root.mkdir(parents=True, exist_ok=True)
    print(
        f"rows={args.rows} dim={args.dim} modes={modes} data_dir={data_root}",
        flush=True,
    )

    client = lynse.VectorDBClient(str(data_root))
    db = client.create_database("scale", drop_if_exists=True)
    coll = db.require_collection(
        "v", dim=args.dim, drop_if_exists=True, default_index=None
    )

    t_ing = time.perf_counter()
    _ingest(coll, args.rows, args.dim, args.seed, args.batch)
    ingest_s = time.perf_counter() - t_ing
    print(f"ingest_s={ingest_s:.1f}", flush=True)

    rng = np.random.default_rng(args.seed + 1)
    queries = [rng.random(args.dim, dtype=np.float32) for _ in range(args.queries)]

    exact: list[set[int]] | None = None
    if not args.skip_gt:
        print("building FLAT-IP for ground truth...", flush=True)
        coll.build_index("FLAT-IP")
        exact = _exact_topk(
            lambda q, k: coll.search(q, k=k).ids.tolist(),
            queries,
            args.k,
        )

    results = {
        "rows": args.rows,
        "dim": args.dim,
        "k": args.k,
        "nprobe": args.nprobe,
        "queries": args.queries,
        "ingest_s": ingest_s,
        "data_dir": str(data_root),
        "modes": {},
    }

    for mode in modes:
        print(f"\n=== build {mode} ===", flush=True)
        kw = {"n_clusters": 1024} if mode.startswith("IVF") else {}
        t0 = time.perf_counter()
        coll.build_index(mode, **kw)
        build_s = time.perf_counter() - t0
        print(f"build_s={build_s:.1f}", flush=True)

        diskann_dir = data_root / "scale" / "v" / "diskann"
        sidecar = {}
        if diskann_dir.exists():
            for p in sorted(diskann_dir.iterdir()):
                sidecar[p.name] = p.stat().st_size
            print(f"diskann sidecars: {sidecar}", flush=True)

        search = _median_ms(
            lambda: coll.search(queries[0], k=args.k, nprobe=args.nprobe),
            warm=5,
            trials=30,
        )
        print(
            f"search_p50={search['p50_ms']:.3f}ms p90={search['p90_ms']:.3f}ms",
            flush=True,
        )

        recall = None
        if exact is not None:
            recs = []
            for q, ex in zip(queries, exact):
                got = set(int(x) for x in coll.search(q, k=args.k, nprobe=args.nprobe).ids)
                recs.append(len(got & ex) / float(args.k))
            recall = float(statistics.mean(recs))
            print(f"recall@{args.k}={recall:.3f}", flush=True)

        results["modes"][mode] = {
            "build_s": build_s,
            "search": search,
            "recall_at_k": recall,
            "sidecar_bytes": sidecar,
            "kwargs": kw,
        }

    text = json.dumps(results, indent=2)
    print("\n=== SUMMARY ===", flush=True)
    print(text, flush=True)
    out = Path(args.out) if args.out else data_root / "diskann_scale_result.json"
    out.write_text(text)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
