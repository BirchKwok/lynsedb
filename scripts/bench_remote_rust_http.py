#!/usr/bin/env python3
"""Benchmark remote Rust RemoteHttpClient throughput.

Environment knobs:
  LYNSE_REMOTE_BENCH_N=10000
  LYNSE_REMOTE_BENCH_RECORD_N=5000
  LYNSE_REMOTE_BENCH_DIM=64
  LYNSE_REMOTE_BENCH_BATCH=1000
  LYNSE_REMOTE_BENCH_LOOPS=50
  LYNSE_REMOTE_BENCH_WARMUP=5
  LYNSE_REMOTE_BENCH_BATCH_QUERIES=16
  LYNSE_REMOTE_BENCH_WRITE_REPEATS=3
  LYNSE_REMOTE_BENCH_KEEP_ROOT=0
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

import lynse


N = int(os.environ.get("LYNSE_REMOTE_BENCH_N", "10000"))
RECORD_N = int(os.environ.get("LYNSE_REMOTE_BENCH_RECORD_N", str(min(N, 5000))))
DIM = int(os.environ.get("LYNSE_REMOTE_BENCH_DIM", "64"))
BATCH = int(os.environ.get("LYNSE_REMOTE_BENCH_BATCH", "1000"))
LOOPS = int(os.environ.get("LYNSE_REMOTE_BENCH_LOOPS", "50"))
WARMUP = int(os.environ.get("LYNSE_REMOTE_BENCH_WARMUP", "5"))
BATCH_QUERIES = int(os.environ.get("LYNSE_REMOTE_BENCH_BATCH_QUERIES", "16"))
WRITE_REPEATS = int(os.environ.get("LYNSE_REMOTE_BENCH_WRITE_REPEATS", "3"))
SEED = int(os.environ.get("LYNSE_REMOTE_BENCH_SEED", "20260619"))
KEEP_ROOT = os.environ.get("LYNSE_REMOTE_BENCH_KEEP_ROOT", "0") == "1"


def percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def bench(label: str, fn, *, loops: int = LOOPS, warmup: int = WARMUP) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(loops):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    row = {
        "p50_ms": percentile(samples, 50),
        "p90_ms": percentile(samples, 90),
        "mean_ms": mean(samples),
        "loops": float(loops),
    }
    print(
        f"{label:34s} p50={row['p50_ms']:9.3f} ms "
        f"p90={row['p90_ms']:9.3f} ms mean={row['mean_ms']:9.3f} ms",
        flush=True,
    )
    return row


def time_once(label: str, fn, *, rows: int | None = None) -> dict[str, float]:
    start = time.perf_counter()
    value = fn()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    rate = None if rows is None or elapsed_ms <= 0 else rows / (elapsed_ms / 1e3)
    suffix = "" if rate is None else f" throughput={rate:,.0f} rows/s"
    print(f"{label:34s} elapsed={elapsed_ms:9.3f} ms{suffix}", flush=True)
    return {"elapsed_ms": elapsed_ms, "rows_per_sec": rate or 0.0, "value": value}


def time_repeated(label: str, fn, *, rows: int | None = None, repeats: int = WRITE_REPEATS) -> dict[str, float]:
    samples = []
    value = None
    for index in range(max(1, repeats)):
        start = time.perf_counter()
        value = fn(index)
        samples.append((time.perf_counter() - start) * 1e3)
    elapsed_ms = median(samples)
    rate = None if rows is None or elapsed_ms <= 0 else rows / (elapsed_ms / 1e3)
    suffix = "" if rate is None else f" throughput={rate:,.0f} rows/s"
    print(
        f"{label:34s} median={elapsed_ms:8.3f} ms "
        f"p90={percentile(samples, 90):8.3f} ms repeats={len(samples)}{suffix}",
        flush=True,
    )
    return {
        "elapsed_ms": elapsed_ms,
        "p90_ms": percentile(samples, 90),
        "rows_per_sec": rate or 0.0,
        "repeats": float(len(samples)),
        "value": value,
    }


def time_repeated_with_setup(
    label: str,
    setup,
    fn,
    *,
    rows: int | None = None,
    repeats: int = WRITE_REPEATS,
) -> dict[str, float]:
    samples = []
    value = None
    for index in range(max(1, repeats)):
        context = setup(index)
        start = time.perf_counter()
        value = fn(context)
        samples.append((time.perf_counter() - start) * 1e3)
    elapsed_ms = median(samples)
    rate = None if rows is None or elapsed_ms <= 0 else rows / (elapsed_ms / 1e3)
    suffix = "" if rate is None else f" throughput={rate:,.0f} rows/s"
    print(
        f"{label:34s} median={elapsed_ms:8.3f} ms "
        f"p90={percentile(samples, 90):8.3f} ms repeats={len(samples)}{suffix}",
        flush=True,
    )
    return {
        "elapsed_ms": elapsed_ms,
        "p90_ms": percentile(samples, 90),
        "rows_per_sec": rate or 0.0,
        "repeats": float(len(samples)),
        "value": value,
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(root: Path) -> tuple[str, subprocess.Popen]:
    port = free_port()
    cmd = [
        sys.executable,
        "-m",
        "lynse.server",
        "run",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--root",
        str(root),
        "--workers",
        "2",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urlopen(base_url, timeout=0.2) as response:
                if response.status == 200:
                    return base_url, process
        except (OSError, URLError):
            time.sleep(0.02)
    process.kill()
    output, _ = process.communicate(timeout=2)
    raise RuntimeError(f"server did not start\n{output}")


def make_fields(size: int) -> list[dict]:
    return [
        {
            "rank": i,
            "bucket": i % 100,
            "tag": f"tag{i % 17}",
            "active": i % 2 == 0,
        }
        for i in range(size)
    ]


def run_mode(base_url: str, mode: str, vectors: np.ndarray, record_vectors: np.ndarray) -> dict[str, dict]:
    client = lynse.VectorDBClient(base_url)
    db = client.create_database(f"bench_{mode.replace('-', '_')}", drop_if_exists=True)

    auto_rows = vectors.shape[0]
    auto_add = time_repeated(
        f"{mode} auto-id binary add",
        lambda index: db.require_collection(
            f"auto_vectors_write_{index}",
            dim=DIM,
            drop_if_exists=True,
            default_index=None,
        ).add(ids=None, vectors=vectors, batch_size=BATCH),
        rows=auto_rows,
    )
    auto = db.require_collection("auto_vectors_search", dim=DIM, drop_if_exists=True, default_index=None)
    auto.add(ids=None, vectors=vectors, batch_size=BATCH)
    query = vectors[0].copy()
    batch_queries = vectors[:BATCH_QUERIES].copy()
    search = bench(f"{mode} binary search", lambda: auto.search(query, k=10))
    batch_search = bench(
        f"{mode} binary batch_search",
        lambda: auto.batch_search(batch_queries, k=10),
        loops=max(10, LOOPS // 2),
    )

    record_ids = list(range(record_vectors.shape[0]))
    fields = make_fields(record_vectors.shape[0])
    record_add = time_repeated(
        f"{mode} int-id+fields add",
        lambda index: db.require_collection(
            f"record_vectors_add_{index}",
            dim=DIM,
            drop_if_exists=True,
            default_index=None,
        ).add(
            ids=record_ids,
            vectors=record_vectors,
            fields=fields,
            batch_size=BATCH,
        ),
        rows=record_vectors.shape[0],
    )
    upsert_fields = make_fields(min(BATCH, record_vectors.shape[0]))
    for item in upsert_fields:
        item["tag"] = "updated"

    def upsert_setup(index: int):
        records = db.require_collection(
            f"record_vectors_upsert_{index}",
            dim=DIM,
            drop_if_exists=True,
            default_index=None,
        )
        records.add(
            ids=record_ids,
            vectors=record_vectors,
            fields=fields,
            batch_size=BATCH,
        )
        return records

    def upsert_once(records):
        return records.upsert(
            ids=record_ids[: len(upsert_fields)],
            vectors=record_vectors[: len(upsert_fields)],
            fields=upsert_fields,
            batch_size=BATCH,
        )

    upsert = time_repeated_with_setup(
        f"{mode} int-id+fields upsert",
        upsert_setup,
        upsert_once,
        rows=len(upsert_fields),
    )

    client.close()
    return {
        "auto_add": auto_add,
        "search": search,
        "batch_search": batch_search,
        "record_add": record_add,
        "upsert": upsert,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    vectors = rng.random((N, DIM), dtype=np.float32)
    record_vectors = rng.random((RECORD_N, DIM), dtype=np.float32)

    mode = "rust-remote"
    root = Path(tempfile.mkdtemp(prefix=f"lynse-remote-bench-{mode}-"))
    base_url, process = start_server(root)
    print(
        f"\nBenchmark server: {base_url} mode={mode} root={root} n={N} "
        f"record_n={RECORD_N} dim={DIM} batch={BATCH} loops={LOOPS} "
        f"write_repeats={WRITE_REPEATS}",
        flush=True,
    )
    try:
        run_mode(base_url, mode, vectors, record_vectors)
    finally:
        process.terminate()
        process.wait(timeout=5)
        if KEEP_ROOT:
            print(f"Kept benchmark root: {root}", flush=True)
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
