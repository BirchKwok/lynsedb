from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np


def _free_port() -> int:
    candidates = list(range(20000, 50000))
    random.shuffle(candidates)
    for port in candidates:
        if _port_available(port) and _port_available(port + 10000):
            return port
    raise RuntimeError("could not find a free HTTP/RPC port pair")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_http(uri: str, timeout: float = 20.0) -> None:
    deadline = time.perf_counter() + timeout
    last_exc: Exception | None = None
    with httpx.Client(timeout=1.0) as client:
        while time.perf_counter() < deadline:
            try:
                response = client.get(uri)
                if response.status_code == 200:
                    return
            except Exception as exc:
                last_exc = exc
            time.sleep(0.1)
    raise RuntimeError(f"service did not become ready at {uri}: {last_exc}")


def _start_server(port: int, data_dir: Path, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("LYNSE_LOG_LEVEL", "warn")
    log_file = log_path.open("wb")
    return subprocess.Popen(
        [
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
            "4",
            "--no-audit-log",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )


def _start_coordinator(
    port: int,
    config_path: Path,
    state_path: Path,
    log_path: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("LYNSE_LOG_LEVEL", "warn")
    log_file = log_path.open("wb")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lynse.server",
            "--role",
            "coordinator",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--cluster-config",
            str(config_path),
            "--cluster-state",
            str(state_path),
            "--request-timeout-secs",
            "120",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )


def _stop_processes(processes: list[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.perf_counter() + 5
    for proc in processes:
        remaining = max(0.1, deadline - time.perf_counter())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


def _print_log_tails(log_dir: Path) -> None:
    for path in sorted(log_dir.glob("*.log")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        print(f"\n--- {path.name} ---", file=sys.stderr)
        for line in lines[-40:]:
            print(line, file=sys.stderr)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _require_collection(
    client: httpx.Client,
    uri: str,
    db: str,
    coll: str,
    dim: int,
    dtypes: str = "float32",
) -> None:
    response = client.post(
        f"{uri}/required_collection",
        json={
            "database_name": db,
            "collection_name": coll,
            "dim": dim,
            "drop_if_exists": True,
            "dtypes": dtypes,
        },
    )
    response.raise_for_status()


def _create_database(client: httpx.Client, uri: str, db: str) -> None:
    response = client.post(f"{uri}/create_database", json={"database_name": db})
    response.raise_for_status()


def _range_params(db: str, coll: str, n_vectors: int, dim: int, start_id: int) -> dict[str, Any]:
    return {
        "database_name": db,
        "collection_name": coll,
        "dim": dim,
        "n_vectors": n_vectors,
        "ids_encoding": "range",
        "ids_start": start_id,
        "return_ids": "false",
    }


def _vector_body(vectors: np.ndarray, wire_dtype: str) -> tuple[bytes, str]:
    encoding = wire_dtype.lower()
    if encoding in {"float16", "f16", "fp16"}:
        return np.ascontiguousarray(vectors, dtype="<f2").tobytes(), "float16"
    if encoding in {"float32", "f32"}:
        return np.ascontiguousarray(vectors, dtype=np.float32).tobytes(), "float32"
    raise ValueError(f"unsupported wire dtype: {wire_dtype}")


def _bench_json_add(
    client: httpx.Client,
    uri: str,
    db: str,
    coll: str,
    vectors: np.ndarray,
    start_id: int,
    _wire_dtype: str = "float32",
) -> dict[str, Any]:
    items = [
        {"vector": row.tolist(), "id": start_id + idx, "field": {}}
        for idx, row in enumerate(vectors)
    ]
    payload = {
        "database_name": db,
        "collection_name": coll,
        "items": items,
    }
    body = _json_bytes(payload)
    started = time.perf_counter()
    response = client.post(
        f"{uri}/bulk_add_items",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"JSON add failed: {response.status_code} {response.text}")
    return {
        "elapsed_s": elapsed,
        "request_bytes": len(body),
        "response_bytes": len(response.content),
    }


def _bench_binary_add(
    client: httpx.Client,
    uri: str,
    db: str,
    coll: str,
    vectors: np.ndarray,
    start_id: int,
    wire_dtype: str = "float32",
) -> dict[str, Any]:
    body, vector_encoding = _vector_body(vectors, wire_dtype)
    params = _range_params(db, coll, vectors.shape[0], vectors.shape[1], start_id)
    params["vector_encoding"] = vector_encoding
    started = time.perf_counter()
    response = client.post(
        f"{uri}/bulk_add_items_binary",
        params=params,
        content=body,
        headers={"Content-Type": "application/octet-stream"},
    )
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"binary add failed: {response.status_code} {response.text}")
    return {
        "elapsed_s": elapsed,
        "request_bytes": len(body),
        "response_bytes": len(response.content),
    }


def _bench_batch_search(
    client: httpx.Client,
    uri: str,
    db: str,
    coll: str,
    queries: np.ndarray,
    k: int,
    wire_dtype: str = "float32",
) -> dict[str, Any]:
    body, vector_encoding = _vector_body(queries, wire_dtype)
    started = time.perf_counter()
    response = client.post(
        f"{uri}/batch_search_binary",
        params={
            "database_name": db,
            "collection_name": coll,
            "dim": queries.shape[1],
            "n_queries": queries.shape[0],
            "k": k,
            "return_fields": "false",
            "vector_encoding": vector_encoding,
        },
        content=body,
        headers={"Content-Type": "application/octet-stream"},
    )
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"batch search failed: {response.status_code} {response.text}")
    return {
        "elapsed_s": elapsed,
        "request_bytes": len(body),
        "response_bytes": len(response.content),
    }


def _format_result(label: str, result: dict[str, Any], useful_bytes: int) -> str:
    elapsed = result["elapsed_s"]
    wire_bytes = result["request_bytes"] + result["response_bytes"]
    useful_mib = useful_bytes / (1024 * 1024)
    wire_mib = wire_bytes / (1024 * 1024)
    throughput = useful_mib / elapsed if elapsed > 0 else 0.0
    amplification = wire_bytes / useful_bytes if useful_bytes else 0.0
    return (
        f"{label}: {elapsed:.4f}s, useful={useful_mib:.2f} MiB, "
        f"wire={wire_mib:.2f} MiB, useful_throughput={throughput:.2f} MiB/s, "
        f"wire_amplification={amplification:.2f}x"
    )


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    vectors = rng.random((args.vectors, args.dim), dtype=np.float32)
    queries = rng.random((args.queries, args.dim), dtype=np.float32)
    useful_add_bytes = int(vectors.nbytes)
    useful_query_bytes = int(queries.nbytes)

    processes: list[subprocess.Popen] = []
    root = Path(tempfile.mkdtemp(prefix="lynsedb-transport-"))
    log_dir = root / "logs"
    log_dir.mkdir()
    try:
        single_port = _free_port()
        single_uri = f"http://127.0.0.1:{single_port}"
        processes.append(_start_server(single_port, root / "single", log_dir / "single.log"))
        _wait_http(single_uri)

        shard_ports = [_free_port(), _free_port()]
        for idx, port in enumerate(shard_ports):
            processes.append(
                _start_server(port, root / f"shard-{idx}", log_dir / f"shard-{idx}.log")
            )
            _wait_http(f"http://127.0.0.1:{port}")

        coord_port = _free_port()
        coord_uri = f"http://127.0.0.1:{coord_port}"
        cluster_config = {
            "bucket_count": 4096,
            "shard_groups": [
                {"name": f"sg{idx}", "primary": f"http://127.0.0.1:{port}"}
                for idx, port in enumerate(shard_ports)
            ],
        }
        config_path = root / "cluster.json"
        state_path = root / "cluster-state.json"
        config_path.write_text(json.dumps(cluster_config), encoding="utf-8")
        processes.append(
            _start_coordinator(coord_port, config_path, state_path, log_dir / "coordinator.log")
        )
        _wait_http(coord_uri)

        with httpx.Client(timeout=120.0) as client:
            _create_database(client, single_uri, args.database)
            _create_database(client, coord_uri, args.database)
            cases = [
                ("remote-json-add", single_uri, "single_json", _bench_json_add, "float32", "float32"),
                ("remote-binary-f32-add", single_uri, "single_binary_f32", _bench_binary_add, "float32", "float32"),
                ("remote-binary-f16-add", single_uri, "single_binary_f16", _bench_binary_add, "float16", "float16"),
                ("cluster-json-add", coord_uri, "cluster_json", _bench_json_add, "float32", "float32"),
                ("cluster-binary-f32-add", coord_uri, "cluster_binary_f32", _bench_binary_add, "float32", "float32"),
                ("cluster-binary-f16-add", coord_uri, "cluster_binary_f16", _bench_binary_add, "float16", "float16"),
            ]
            for _label, uri, coll, _fn, _wire_dtype, storage_dtype in cases:
                _require_collection(client, uri, args.database, coll, args.dim, storage_dtype)

            results = []
            for label, uri, coll, fn, wire_dtype, _storage_dtype in cases:
                result = fn(client, uri, args.database, coll, vectors, args.start_id, wire_dtype)
                results.append((label, result, useful_add_bytes))

            search_cases = [
                ("remote-batch-search-f32", single_uri, "single_binary_f32", "float32"),
                ("remote-batch-search-f16", single_uri, "single_binary_f16", "float16"),
                ("cluster-batch-search-f32", coord_uri, "cluster_binary_f32", "float32"),
                ("cluster-batch-search-f16", coord_uri, "cluster_binary_f16", "float16"),
            ]
            for label, uri, coll, wire_dtype in search_cases:
                result = _bench_batch_search(
                    client,
                    uri,
                    args.database,
                    coll,
                    queries,
                    args.k,
                    wire_dtype,
                )
                results.append((label, result, useful_query_bytes))

        print(
            json.dumps(
                {
                    "vectors": args.vectors,
                    "dim": args.dim,
                    "queries": args.queries,
                    "k": args.k,
                    "results": [
                        {
                            "label": label,
                            **result,
                            "useful_bytes": useful_bytes,
                            "wire_amplification": (
                                (result["request_bytes"] + result["response_bytes"])
                                / useful_bytes
                                if useful_bytes
                                else 0.0
                            ),
                        }
                        for label, result, useful_bytes in results
                    ],
                },
                indent=2,
            )
        )
        print()
        for label, result, useful_bytes in results:
            print(_format_result(label, result, useful_bytes))
    except Exception:
        _print_log_tails(log_dir)
        raise
    finally:
        _stop_processes(processes)
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LynseDB remote and cluster transport")
    parser.add_argument("--vectors", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--database", default="bench")
    parser.add_argument("--start-id", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
