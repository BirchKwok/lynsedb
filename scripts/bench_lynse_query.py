#!/usr/bin/env python3
"""Benchmark LynseDB metadata query paths after performance fixes."""
import shutil
import time
from pathlib import Path

import numpy as np

import lynse

DATA_DIR = Path("/tmp/lynse_bench_query")
N = 1_000_000
LOOPS_FAST = 500
LOOPS_SLOW = 100


def bench(label, fn, loops):
    for _ in range(20):
        fn()
    t0 = time.perf_counter()
    for _ in range(loops):
        fn()
    us = (time.perf_counter() - t0) / loops * 1e6
    print(f"  {label:40s} {us:8.1f} us")
    return us


def main():
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    client = lynse.VectorDBClient(str(DATA_DIR))
    db = client.create_database("bench_db", drop_if_exists=True)
    collection = db.require_collection("vectors", dim=128, drop_if_exists=True)

    query = np.random.random(128)
    print(f"Inserting {N:,} rows...")
    t0 = time.perf_counter()
    with collection.insert_session() as session:
        for i in range(N):
            vec = query if i == 0 else np.random.random(128)
            session.add_item(vec, id=i, field={"test": f"test_{i // 1000}", "order": i})
    print(f"Insert done in {time.perf_counter() - t0:.1f}s\n")

    print("LynseDB benchmarks (lower is better):")
    bench('query(\'"order"=1\')', lambda: collection.query('"order"=1'), LOOPS_FAST)
    bench(
        'query(\'"order"=1\', return_ids_only=True)',
        lambda: collection.query('"order"=1', return_ids_only=True),
        LOOPS_FAST,
    )
    bench('query(\'"order" IN (1,2)\')', lambda: collection.query('"order" IN (1, 2)'), LOOPS_SLOW)
    bench(
        'query(\'"order"=1 OR "order"=2\')',
        lambda: collection.query('"order" = 1 OR "order" = 2'),
        LOOPS_SLOW,
    )
    bench('search(k=10)', lambda: collection.search(query, k=10), LOOPS_SLOW)
    bench(
        'search(k=10, where=\'"order"=1\')',
        lambda: collection.search(query, k=10, where='"order"=1'),
        LOOPS_SLOW,
    )
    bench(
        'search(k=10, where IN)',
        lambda: collection.search(query, k=10, where='"order" IN (1, 2)'),
        LOOPS_SLOW,
    )


if __name__ == "__main__":
    main()
