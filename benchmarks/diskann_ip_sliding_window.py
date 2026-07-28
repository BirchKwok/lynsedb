#!/usr/bin/env python3
"""Small SlidingWindow-style stress for DiskANN IP in-place updates.

Builds a DiskANN index, then repeatedly inserts a batch and deletes an older
batch, measuring search recall@10 against brute force on the live set.

Usage:
  python benchmarks/diskann_ip_sliding_window.py
  python benchmarks/diskann_ip_sliding_window.py --n 2000 --dim 32 --steps 8
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


def recall_at_k(approx: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    if len(approx) == 0 or len(truth) == 0:
        return 0.0
    a = set(approx[:k].tolist())
    t = set(truth[:k].tolist())
    return len(a & t) / float(min(k, len(t)))


def brute_topk(
    data: np.ndarray, query: np.ndarray, ids: np.ndarray, k: int, metric: str
) -> np.ndarray:
    if metric == "l2":
        d = np.sum((data - query) ** 2, axis=1)
        order = np.argsort(d)[:k]
    else:
        scores = data @ query
        order = np.argsort(-scores)[:k]
    return ids[order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="vectors per window half")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", default="DISKANN-L2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lynse import VectorDBClient

    rng = np.random.default_rng(args.seed)
    root = Path(tempfile.mkdtemp(prefix="diskann_ip_sw_"))
    try:
        client = VectorDBClient(uri=str(root))
        db = client.create_database("db", drop_if_exists=True)
        coll = db.require_collection("sw", dim=args.dim, drop_if_exists=True)

        part = args.n
        data = rng.standard_normal((part * 2, args.dim)).astype(np.float32)
        ids = np.arange(part * 2, dtype=np.int64)
        coll.add(ids=ids[:part].tolist(), vectors=data[:part])
        coll.commit()
        coll.build_index(args.mode)

        metric = "ip" if "IP" in args.mode.upper() else "l2"
        live_mask = np.zeros(part * (args.steps + 2), dtype=bool)
        live_mask[:part] = True
        all_data = [data[:part]]
        next_id = part

        recalls = []
        t0 = time.perf_counter()
        for step in range(args.steps):
            batch = rng.standard_normal((part, args.dim)).astype(np.float32)
            batch_ids = np.arange(next_id, next_id + part, dtype=np.int64)
            next_id += part
            all_data.append(batch)
            coll.add(ids=batch_ids.tolist(), vectors=batch)
            coll.commit()
            live_mask[batch_ids] = True

            if step >= 1:
                expire_from = (step - 1) * part
                expire_ids = list(range(expire_from, expire_from + part))
                coll.delete(expire_ids)
                live_mask[expire_ids] = False

            corpus = np.vstack(all_data)
            live_ids = np.nonzero(live_mask[: corpus.shape[0]])[0]
            live_data = corpus[live_ids]
            q = rng.standard_normal(args.dim).astype(np.float32)
            truth = brute_topk(live_data, q, live_ids, args.k, metric)
            res = coll.search(vector=q, k=args.k)
            got = np.asarray(res.ids, dtype=np.int64)
            r = recall_at_k(got, truth, args.k)
            recalls.append(r)
            print(
                f"step={step} live={live_ids.size} recall@{args.k}={r:.3f} "
                f"top={got[:5].tolist()}"
            )

        elapsed = time.perf_counter() - t0
        print(
            f"done mode={args.mode} mean_recall={float(np.mean(recalls)):.3f} "
            f"min_recall={float(np.min(recalls)):.3f} elapsed_s={elapsed:.2f} path={root}"
        )
        client.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
