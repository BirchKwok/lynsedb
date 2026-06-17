from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import httpx
import numpy as np


def _is_port_free(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _derive_rpc_port(http_port: int) -> int:
    return http_port + 10000 if http_port <= 55535 else http_port - 10000


def _pick_ports(count: int) -> list[int]:
    ports: list[int] = []
    for port in range(19000, 26000):
        if len(ports) == count:
            break
        rpc_port = _derive_rpc_port(port)
        if port in ports or rpc_port in ports:
            continue
        if _is_port_free(port) and _is_port_free(rpc_port):
            ports.append(port)
    if len(ports) != count:
        raise RuntimeError(f"could not find {count} free ports")
    return ports


def _wait_http(url: str, timeout: float = 20.0) -> None:
    deadline = time.perf_counter() + timeout
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f"server did not become ready: {url}") from last_error


def _terminate(processes: list[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.perf_counter() + 5.0
    for proc in processes:
        remaining = max(0.1, deadline - time.perf_counter())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
    for proc in processes:
        with contextlib.suppress(Exception):
            proc.wait(timeout=1.0)


def _timed(fn, repeat: int = 1) -> list[float]:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _stats(times: list[float], *, scale: float = 1000.0) -> dict[str, float]:
    values = sorted(t * scale for t in times)
    if not values:
        return {"median": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "median": median(values),
        "mean": mean(values),
        "min": values[0],
        "max": values[-1],
    }


def _record_metric(metrics: dict[str, Any], name: str, times: list[float], **extra: Any) -> None:
    metrics[name] = {**_stats(times), **extra}


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import lynse

    root = Path(args.work_dir or tempfile.mkdtemp(prefix="lynsedb_cluster_bench_"))
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    ports = _pick_ports(args.shards + 1)
    coord_port = ports[0]
    shard_ports = ports[1:]
    processes: list[subprocess.Popen] = []
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "python"))
    env.setdefault("RUST_LOG", "error")

    try:
        for idx, port in enumerate(shard_ports):
            data_dir = root / f"shard_{idx}"
            cmd = [
                sys.executable,
                "-m",
                "lynse.server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--data-dir",
                str(data_dir),
                "--workers",
                str(args.workers),
                "--no-audit-log",
            ]
            processes.append(
                subprocess.Popen(
                    cmd,
                    cwd=Path(__file__).resolve().parents[1],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        for port in shard_ports:
            _wait_http(f"http://127.0.0.1:{port}/")

        config = {
            "bucket_count": args.bucket_count,
            "write_mirror_replicas": False,
            "shard_groups": [
                {
                    "name": f"sg{idx}",
                    "primary": f"http://127.0.0.1:{port}",
                    "replicas": [],
                }
                for idx, port in enumerate(shard_ports)
            ],
        }
        config_path = root / "cluster.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        state_path = root / "cluster_state.json"
        coord_cmd = [
            sys.executable,
            "-m",
            "lynse.server",
            "--role",
            "coordinator",
            "--host",
            "127.0.0.1",
            "--port",
            str(coord_port),
            "--cluster-config",
            str(config_path),
            "--cluster-state",
            str(state_path),
            "--health-interval-secs",
            "30",
        ]
        processes.append(
            subprocess.Popen(
                coord_cmd,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        coord_uri = f"http://127.0.0.1:{coord_port}"
        _wait_http(f"{coord_uri}/")

        rng = np.random.default_rng(args.seed)
        vectors = np.ascontiguousarray(
            rng.random((args.rows, args.dim), dtype=np.float32)
        )
        queries = np.ascontiguousarray(
            rng.random((args.queries, args.dim), dtype=np.float32)
        )
        explicit_ids = list(range(args.rows))

        client = lynse.VectorDBClient(coord_uri)
        coll = client.create_collection(
            "bench_db",
            "vectors",
            dim=args.dim,
            drop_database_if_exists=True,
            default_index=None,
        )

        metrics: dict[str, Any] = {}

        def add_json() -> None:
            coll.add(
                ids=explicit_ids,
                vectors=vectors,
                batch_size=args.batch_size,
                wire_dtype=args.wire_dtype,
            )

        add_times = _timed(add_json)
        _record_metric(
            metrics,
            "storage.add_explicit_ids_public",
            add_times,
            rows=args.rows,
            rows_per_second=args.rows / add_times[0],
        )

        commit_times = _timed(coll.commit, repeat=args.commit_repeats)
        _record_metric(metrics, "storage.commit", commit_times)

        # Warm search paths so the mmap/page-cache side is not the whole result.
        for query in queries[: min(5, len(queries))]:
            coll.search(query, k=args.k)

        json_search_times = _timed(
            lambda: [coll.search(query, k=args.k) for query in queries],
            repeat=args.search_repeats,
        )
        _record_metric(
            metrics,
            "communication.search_public",
            json_search_times,
            queries=args.queries,
            queries_per_second=args.queries / median(json_search_times),
        )

        binary_search_available = hasattr(coll, "_search")
        if binary_search_available:
            binary_search_times = _timed(
                lambda: [
                    coll._search(query, args.k, None, return_fields=False)
                    for query in queries
                ],
                repeat=args.search_repeats,
            )
            _record_metric(
                metrics,
                "communication.search_binary_private",
                binary_search_times,
                queries=args.queries,
                queries_per_second=args.queries / median(binary_search_times),
            )

        try:
            batch_json_times = _timed(
                lambda: coll.batch_search(queries, k=args.k),
                repeat=args.batch_search_repeats,
            )
            _record_metric(
                metrics,
                "compute.batch_search_public",
                batch_json_times,
                queries=args.queries,
                queries_per_second=args.queries / median(batch_json_times),
                available=True,
            )
        except Exception as exc:
            metrics["compute.batch_search_public"] = {
                "available": False,
                "error": str(exc),
            }

        if binary_search_available:
            def batch_search_binary() -> None:
                params = {
                    "database_name": coll._database_name,
                    "collection_name": coll._collection_name,
                    "dim": args.dim,
                    "n_queries": args.queries,
                    "k": args.k,
                    "return_fields": "false",
                    "vector_encoding": args.wire_dtype,
                    "nprobe": 10,
                }
                body = (
                    np.ascontiguousarray(queries, dtype="<f2").tobytes()
                    if args.wire_dtype == "float16"
                    else queries.tobytes()
                )
                response = coll._session.post(
                    f"{coll._uri}/batch_search_binary",
                    params=params,
                    content=body,
                    headers={"Content-Type": "application/octet-stream"},
                )
                if response.status_code != 200:
                    raise RuntimeError(response.text)

            batch_binary_times = _timed(
                batch_search_binary,
                repeat=args.batch_search_repeats,
            )
            _record_metric(
                metrics,
                "compute.batch_search_binary_raw",
                batch_binary_times,
                queries=args.queries,
                queries_per_second=args.queries / median(batch_binary_times),
            )

        checkpoint_times = _timed(coll.checkpoint, repeat=args.checkpoint_repeats)
        _record_metric(metrics, "storage.checkpoint", checkpoint_times)

        bulk_error = None
        try:
            bulk_client = lynse.VectorDBClient(coord_uri)
            bulk_coll = bulk_client.create_collection(
                "bulk_db",
                "vectors",
                dim=args.dim,
                drop_database_if_exists=True,
                default_index=None,
            )
            bulk_vectors = vectors[: min(args.batch_size, args.rows)]
            t0 = time.perf_counter()
            bulk_coll.add(
                ids=None,
                vectors=bulk_vectors,
                batch_size=len(bulk_vectors),
                wire_dtype=args.wire_dtype,
            )
            elapsed = time.perf_counter() - t0
            metrics["storage.bulk_auto_ids_binary"] = {
                **_stats([elapsed]),
                "rows": len(bulk_vectors),
                "rows_per_second": len(bulk_vectors) / elapsed,
                "available": True,
            }
        except Exception as exc:
            bulk_error = str(exc)
            metrics["storage.bulk_auto_ids_binary"] = {
                "available": False,
                "error": bulk_error,
            }

        return {
            "config": {
                "rows": args.rows,
                "dim": args.dim,
                "queries": args.queries,
                "k": args.k,
                "batch_size": args.batch_size,
                "shards": args.shards,
                "wire_dtype": args.wire_dtype,
                "root": str(root),
                "coordinator": coord_uri,
            },
            "metrics": metrics,
        }
    finally:
        _terminate(processes)
        if args.work_dir is None and not args.keep_data:
            shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LynseDB cluster data paths")
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bucket-count", type=int, default=4096)
    parser.add_argument("--search-repeats", type=int, default=5)
    parser.add_argument("--batch-search-repeats", type=int, default=5)
    parser.add_argument("--commit-repeats", type=int, default=3)
    parser.add_argument("--checkpoint-repeats", type=int, default=3)
    parser.add_argument("--wire-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = run_benchmark(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
