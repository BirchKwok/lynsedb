#!/usr/bin/env python3
"""Benchmark LynseDB query/search paths on a 1M x 128 dataset.

The default target matches the large local benchmark requested for query,
filtered vector search, batch search, BM25, and hybrid vector+text search.

Environment knobs:
  LYNSE_BENCH_N=1000000
  LYNSE_BENCH_DIM=128
  LYNSE_BENCH_LOOPS=30
  LYNSE_BENCH_BUILD_CHUNK=100000
  LYNSE_BENCH_FORCE_REBUILD=1
  LYNSE_BENCH_REUSE=0
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from statistics import mean

import numpy as np

import lynse


DATA_DIR = Path(os.environ.get("LYNSE_BENCH_QUERY_DIR", "/tmp/lynse_bench_query"))
N = int(os.environ.get("LYNSE_BENCH_N", "1000000"))
DIM = int(os.environ.get("LYNSE_BENCH_DIM", "128"))
K = int(os.environ.get("LYNSE_BENCH_K", "10"))
BUILD_CHUNK = int(os.environ.get("LYNSE_BENCH_BUILD_CHUNK", "100000"))
WARMUP = int(os.environ.get("LYNSE_BENCH_WARMUP", "8"))
LOOPS = int(os.environ.get("LYNSE_BENCH_LOOPS", "30"))
BATCH_QUERIES = int(os.environ.get("LYNSE_BENCH_BATCH_QUERIES", "16"))
SEED = int(os.environ.get("LYNSE_BENCH_SEED", "20260618"))
REUSE = os.environ.get("LYNSE_BENCH_REUSE", "1") != "0"
FORCE_REBUILD = os.environ.get("LYNSE_BENCH_FORCE_REBUILD", "0") == "1"


def percentile(values: list[float], pct: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def bench(label: str, fn, *, loops: int = LOOPS, warmup: int = WARMUP) -> dict[str, float]:
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(loops):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)

    row = {
        "p50_ms": percentile(samples, 50),
        "p90_ms": percentile(samples, 90),
        "p99_ms": percentile(samples, 99),
        "mean_ms": mean(samples),
        "loops": float(loops),
    }
    print(
        f"{label:36s} "
        f"p50={row['p50_ms']:9.3f} ms "
        f"p90={row['p90_ms']:9.3f} ms "
        f"p99={row['p99_ms']:9.3f} ms "
        f"mean={row['mean_ms']:9.3f} ms",
        flush=True,
    )
    return row


def make_fields(start: int, size: int) -> list[dict]:
    fields = []
    for row_id in range(start, start + size):
        bucket = row_id % 1000
        topic = row_id % 100
        tenant = row_id % 10
        fields.append(
            {
                "order": row_id,
                "bucket": bucket,
                "category": f"cat{topic}",
                "title": f"topic{topic}",
                "body": f"topic{topic} tenant{tenant} bucket{bucket}",
            }
        )
    return fields


def open_collection():
    client = lynse.VectorDBClient(str(DATA_DIR))
    db = client.create_database("bench_db", drop_if_exists=False)
    return db.require_collection(
        "vectors",
        dim=DIM,
        drop_if_exists=False,
        default_index=None,
    )


def prepare_collection():
    if DATA_DIR.exists() and REUSE and not FORCE_REBUILD:
        collection = open_collection()
        if collection.shape[0] >= N and collection.shape[1] == DIM:
            print(f"Using existing dataset at {DATA_DIR}", flush=True)
            return collection
        print("Existing dataset shape mismatch; rebuilding.", flush=True)

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    client = lynse.VectorDBClient(str(DATA_DIR))
    db = client.create_database("bench_db", drop_if_exists=True)
    collection = db.require_collection(
        "vectors",
        dim=DIM,
        drop_if_exists=True,
        default_index=None,
    )

    rng = np.random.default_rng(SEED)
    print(
        f"Building dataset: {N:,} vectors x {DIM}d, chunk={BUILD_CHUNK:,}",
        flush=True,
    )
    t0 = time.perf_counter()
    inserted = 0
    while inserted < N:
        size = min(BUILD_CHUNK, N - inserted)
        vectors = rng.random((size, DIM), dtype=np.float32)
        fields = make_fields(inserted, size)
        collection.add(vectors=vectors, fields=fields, batch_size=size)
        inserted += size
        print(f"  inserted {inserted:,}/{N:,}", flush=True)

    collection.commit()
    collection.build_index("FLAT-IP")
    print(f"Build done in {time.perf_counter() - t0:.1f}s", flush=True)
    return collection


def main():
    collection = prepare_collection()
    rng = np.random.default_rng(SEED + 1)
    query = rng.random(DIM, dtype=np.float32)
    batch_queries = rng.random((BATCH_QUERIES, DIM), dtype=np.float32)

    exact_order = min(123_456, N - 1)
    filters = {
        "1 row": f'"order" = {exact_order}',
        "1%": '"bucket" < 10',
        "10%": '"bucket" < 100',
        "50%": '"bucket" < 500',
    }

    print("=" * 104, flush=True)
    print(
        f"Benchmark: {N:,} vectors x {DIM}d, k={K}, loops={LOOPS}, "
        f"batch_queries={BATCH_QUERIES}, index={collection.index_mode}",
        flush=True,
    )
    print("=" * 104, flush=True)

    print("\nMetadata query:", flush=True)
    bench("query_fields exact backend", lambda: collection._rust_coll.query_fields(filters["1 row"]), loops=LOOPS * 4)
    bench(
        "query exact public ids-only",
        lambda: collection.query(filters["1 row"], return_ids_only=True),
        loops=LOOPS * 4,
    )
    bench("query bucket < 10 (1%)", lambda: collection._rust_coll.query_fields(filters["1%"]))
    bench("query bucket < 100 (10%)", lambda: collection._rust_coll.query_fields(filters["10%"]))
    bench("query bucket < 500 (50%)", lambda: collection._rust_coll.query_fields(filters["50%"]))

    print("\nVector search:", flush=True)
    bench("search backend unfiltered", lambda: collection._rust_coll.search(query, k=K))
    bench("search public unfiltered", lambda: collection.search(query, k=K))
    for label, where in filters.items():
        bench(f"search filtered {label}", lambda where=where: collection._rust_coll.search(query, k=K, where=where))

    print("\nBatch vector search:", flush=True)
    bench(
        f"batch_search {BATCH_QUERIES}q backend",
        lambda: collection._rust_coll.batch_search(batch_queries, k=K),
        loops=max(5, LOOPS // 2),
    )
    bench(
        f"batch_search {BATCH_QUERIES}q 10% filter",
        lambda: collection._rust_coll.batch_search(batch_queries, k=K, where=filters["10%"]),
        loops=max(5, LOOPS // 2),
    )

    print("\nText and hybrid search:", flush=True)
    bench("bm25 topic42", lambda: collection._rust_coll.text_search("topic42", k=K))
    bench(
        "bm25 topic42 10% filter",
        lambda: collection._rust_coll.text_search("topic42", k=K, where=filters["10%"]),
    )
    bench(
        "hybrid topic42+vector",
        lambda: collection._rust_coll.hybrid_search(
            vector=query,
            text="topic42",
            k=K,
            candidate_limit=100,
        ),
    )
    bench(
        "hybrid topic42+vector 10% filter",
        lambda: collection._rust_coll.hybrid_search(
            vector=query,
            text="topic42",
            k=K,
            where=filters["10%"],
            candidate_limit=100,
        ),
    )

    print("\nProfile path:", flush=True)
    bench(
        "search_profile 10% filter",
        lambda: collection._rust_coll.search_profile(query, k=K, where=filters["10%"]),
    )


if __name__ == "__main__":
    main()
