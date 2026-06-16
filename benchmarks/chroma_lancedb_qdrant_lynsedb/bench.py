#!/usr/bin/env python3
"""Cross-engine embedded vector database benchmarks.

The suite intentionally keeps dependencies optional. Missing third-party
engines produce a skipped row instead of failing the whole run.

The default workload compares LynseDB, ChromaDB, LanceDB, and USEARCH on
100,000 rows of 128-dimensional vectors. Embedded adapters use each engine's
batch insert API during ingest.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_WORKDIR = Path(os.environ.get("VECTOR_BENCH_WORKDIR", "/tmp/vector_db_bench"))
DEFAULT_ENGINES = ["lynsedb", "chroma", "lancedb", "usearch"]
ALL_ENGINES = DEFAULT_ENGINES + ["lynsedb-f32", "lynsedb-http"]
COLLECTION = "items"
DATABASE = "bench"


class SkipEngine(RuntimeError):
    """Raised when an engine cannot run in this environment."""


@dataclass
class Dataset:
    vectors: np.ndarray
    queries: np.ndarray
    ids: list[str]
    categories: np.ndarray
    tenants: list[str]
    texts: list[str]
    query_categories: np.ndarray
    query_texts: list[str]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def make_dataset(n: int, dim: int, n_queries: int, n_categories: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    vectors = normalize_rows(rng.normal(size=(n, dim)).astype(np.float32))
    query_rows = rng.integers(0, n, size=n_queries)
    noise = rng.normal(scale=0.03, size=(n_queries, dim)).astype(np.float32)
    queries = normalize_rows(vectors[query_rows] + noise)
    categories = (np.arange(n, dtype=np.int32) % int(n_categories)).astype(np.int32)
    tenants = [f"tenant-{i % 16}" for i in range(n)]
    ids = [str(i) for i in range(n)]
    texts = [
        f"topic_{int(categories[i])} tenant_{i % 16} document_{i} vector database retrieval common"
        for i in range(n)
    ]
    query_categories = categories[query_rows]
    query_texts = [
        f"topic_{int(category)} vector database retrieval common"
        for category in query_categories
    ]
    return Dataset(
        vectors=vectors,
        queries=queries,
        ids=ids,
        categories=categories,
        tenants=tenants,
        texts=texts,
        query_categories=query_categories,
        query_texts=query_texts,
    )


def iter_batches(dataset: Dataset, batch_size: int):
    n = dataset.vectors.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        fields = [
            {
                "category": int(dataset.categories[i]),
                "tenant": dataset.tenants[i],
                "text": dataset.texts[i],
            }
            for i in range(start, end)
        ]
        yield (
            dataset.ids[start:end],
            dataset.vectors[start:end],
            fields,
            dataset.texts[start:end],
        )


def exact_topk(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int,
    *,
    categories: np.ndarray | None = None,
    category: int | None = None,
) -> list[str]:
    if categories is not None and category is not None:
        candidate_ids = np.flatnonzero(categories == int(category))
        if candidate_ids.size == 0:
            return []
        scores = vectors[candidate_ids] @ query
        local = topk_indices(scores, min(k, scores.shape[0]))
        return [str(int(candidate_ids[i])) for i in local]
    scores = vectors @ query
    top = topk_indices(scores, min(k, scores.shape[0]))
    return [str(int(i)) for i in top]


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k >= scores.shape[0]:
        return np.argsort(-scores, kind="stable")
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


def recall_at_k(actual: Iterable[str], expected: Iterable[str], k: int) -> float:
    expected_set = set(list(expected)[:k])
    if not expected_set:
        return 1.0
    actual_set = set(list(actual)[:k])
    return len(actual_set & expected_set) / len(expected_set)


def latency_stats_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "qps": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "qps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    }


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))


def free_port() -> int:
    candidates = list(range(21000, 52000))
    random.shuffle(candidates)
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("could not find a free port")


def wait_http(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.perf_counter() + timeout_s
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"service did not become ready at {url}: {last_error}")


class Engine:
    name = "base"
    supports_hybrid = False

    def __init__(self, path: Path, dim: int):
        self.path = path
        self.dim = dim

    def setup(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def ingest(self, dataset: Dataset, batch_size: int) -> None:
        raise NotImplementedError

    def search(self, vector: np.ndarray, k: int) -> list[str]:
        raise NotImplementedError

    def filtered_search(self, vector: np.ndarray, category: int, k: int) -> list[str]:
        raise NotImplementedError

    def hybrid_search(self, vector: np.ndarray, text: str, k: int) -> list[str]:
        raise SkipEngine(f"{self.name} hybrid benchmark is not implemented")


class LynseDBEngine(Engine):
    name = "lynsedb"
    supports_hybrid = True
    index_mode = "FLAT-IP"
    vector_dtype = "float16"

    def setup(self) -> None:
        import lynse

        self.client = lynse.VectorDBClient(str(self.path))
        self.db = self.client.create_database(DATABASE, drop_if_exists=True)
        self.collection = self.db.require_collection(
            COLLECTION,
            dim=self.dim,
            drop_if_exists=True,
            dtypes=self.vector_dtype,
            default_index=None,
        )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def ingest(self, dataset: Dataset, batch_size: int) -> None:
        rust_collection = getattr(self.collection, "_rust_coll", None)
        add_encoded_f16 = getattr(rust_collection, "add_items_encoded_f16", None)
        if rust_collection is not None and add_encoded_f16 is not None and self.vector_dtype == "float16":
            for ids, vectors, fields, _texts in iter_batches(dataset, batch_size):
                encoded = np.ascontiguousarray(vectors.astype(np.float16).view(np.uint16))
                add_encoded_f16(encoded, [int(item) for item in ids], fields)
            self.collection.build_index(self.index_mode)
            rust_collection.checkpoint_fast()
            return

        with self.collection:
            for ids, vectors, fields, _texts in iter_batches(dataset, batch_size):
                self.collection.add(
                    ids=ids,
                    vectors=vectors,
                    fields=fields,
                    batch_size=batch_size,
                )
        self.collection.build_index(self.index_mode)

    def search(self, vector: np.ndarray, k: int) -> list[str]:
        return self.collection.search(vector, k=k).ids.astype(str).tolist()

    def filtered_search(self, vector: np.ndarray, category: int, k: int) -> list[str]:
        return self.collection.search(vector, k=k, where=f"category = {int(category)}").ids.astype(str).tolist()

    def hybrid_search(self, vector: np.ndarray, text: str, k: int) -> list[str]:
        return self.collection.hybrid_search(
            vector=vector,
            text=text,
            k=k,
            text_fields=["text"],
        ).ids.astype(str).tolist()


class LynseDBF32Engine(LynseDBEngine):
    name = "lynsedb-f32"
    vector_dtype = "float32"


class LynseDBHttpEngine(LynseDBEngine):
    name = "lynsedb-http"

    def setup(self) -> None:
        import lynse

        self.port = free_port()
        self.log_path = self.path / "server.log"
        self.data_path = self.path / "server-data"
        self.path.mkdir(parents=True, exist_ok=True)
        log_file = self.log_path.open("wb")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lynse.server",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--data-dir",
                str(self.data_path),
                "--workers",
                "4",
                "--no-audit-log",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "LYNSE_LOG_LEVEL": os.environ.get("LYNSE_LOG_LEVEL", "warn")},
        )
        self.url = f"http://127.0.0.1:{self.port}"
        wait_http(self.url)
        self.client = lynse.VectorDBClient(self.url)
        self.db = self.client.create_database(DATABASE, drop_if_exists=True)
        self.collection = self.db.require_collection(
            COLLECTION,
            dim=self.dim,
            drop_if_exists=True,
            dtypes=self.vector_dtype,
            default_index=None,
        )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        proc = getattr(self, "process", None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


class ChromaEngine(Engine):
    name = "chroma"

    def setup(self) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise SkipEngine("pip install chromadb") from exc

        self.client = chromadb.PersistentClient(path=str(self.path))
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            COLLECTION,
            metadata={"hnsw:space": "ip"},
        )

    def ingest(self, dataset: Dataset, batch_size: int) -> None:
        for ids, vectors, fields, texts in iter_batches(dataset, batch_size):
            self.collection.add(
                ids=ids,
                embeddings=vectors.tolist(),
                metadatas=fields,
                documents=texts,
            )

    def search(self, vector: np.ndarray, k: int) -> list[str]:
        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            include=[],
        )
        return [str(item) for item in result.get("ids", [[]])[0]]

    def filtered_search(self, vector: np.ndarray, category: int, k: int) -> list[str]:
        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            where={"category": int(category)},
            include=[],
        )
        return [str(item) for item in result.get("ids", [[]])[0]]


class LanceDBEngine(Engine):
    name = "lancedb"
    supports_hybrid = True

    def setup(self) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise SkipEngine("pip install lancedb pyarrow") from exc

        self.db = lancedb.connect(str(self.path))
        try:
            self.db.drop_table(COLLECTION)
        except Exception:
            pass
        self.table = None

    def ingest(self, dataset: Dataset, batch_size: int) -> None:
        for ids, vectors, fields, texts in iter_batches(dataset, batch_size):
            rows = [
                {
                    "id": str(ids[offset]),
                    "vector": vectors[offset].astype(np.float32).tolist(),
                    "category": int(fields[offset]["category"]),
                    "tenant": fields[offset]["tenant"],
                    "text": texts[offset],
                }
                for offset in range(len(ids))
            ]
            if self.table is None:
                self.table = self.db.create_table(COLLECTION, data=rows)
            else:
                self.table.add(rows)
        try:
            self.table.create_fts_index("text", replace=True)
        except Exception:
            pass

    def search(self, vector: np.ndarray, k: int) -> list[str]:
        result = self.table.search(vector.astype(np.float32)).limit(k).to_list()
        return [str(row["id"]) for row in result]

    def filtered_search(self, vector: np.ndarray, category: int, k: int) -> list[str]:
        result = (
            self.table.search(vector.astype(np.float32))
            .where(f"category = {int(category)}")
            .limit(k)
            .to_list()
        )
        return [str(row["id"]) for row in result]

    def hybrid_search(self, vector: np.ndarray, text: str, k: int) -> list[str]:
        try:
            query = (
                self.table.search(query_type="hybrid")
                .vector(vector.astype(np.float32))
                .text(text)
            )
            result = query.limit(k).to_list()
        except Exception as exc:
            raise SkipEngine(f"lancedb hybrid unavailable: {exc}") from exc
        return [str(row["id"]) for row in result]


class UsearchEngine(Engine):
    name = "usearch"

    def setup(self) -> None:
        try:
            from usearch.index import Index
        except ImportError as exc:
            raise SkipEngine("pip install usearch") from exc

        expansion_search = int(os.environ.get("USEARCH_EXPANSION_SEARCH", "128"))
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.path / "index.usearch"
        self.index = Index(
            ndim=self.dim,
            metric="ip",
            dtype="f32",
            connectivity=16,
            expansion_add=128,
            expansion_search=expansion_search,
        )

    def close(self) -> None:
        self.index = None

    def ingest(self, dataset: Dataset, batch_size: int) -> None:
        for ids, vectors, _fields, _texts in iter_batches(dataset, batch_size):
            keys = np.asarray([int(item) for item in ids], dtype=np.uint64)
            self.index.add(keys, np.ascontiguousarray(vectors, dtype=np.float32), threads=0)
        self.index.save(self.index_path)

    def search(self, vector: np.ndarray, k: int) -> list[str]:
        result = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), k)
        return [str(int(item)) for item in np.asarray(result.keys).tolist()]

    def filtered_search(self, vector: np.ndarray, category: int, k: int) -> list[str]:
        raise SkipEngine("usearch is a vector-only index in this suite; metadata filters are not implemented")


ENGINE_CLASSES: dict[str, type[Engine]] = {
    "lynsedb": LynseDBEngine,
    "lynsedb-f32": LynseDBF32Engine,
    "lynsedb-http": LynseDBHttpEngine,
    "chroma": ChromaEngine,
    "lancedb": LanceDBEngine,
    "usearch": UsearchEngine,
}


def parse_engines(value: str, *, http_only: bool = False) -> list[str]:
    if value == "all":
        return ["lynsedb-http"] if http_only else list(DEFAULT_ENGINES)
    engines = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(engines) - set(ALL_ENGINES))
    if unknown:
        raise ValueError(f"unknown engines: {', '.join(unknown)}")
    return engines


def fresh_engine(engine_name: str, args: argparse.Namespace) -> Engine:
    path = Path(args.workdir) / f"{args.command}-{engine_name}"
    if path.exists() and not args.keep_data:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return ENGINE_CLASSES[engine_name](path=path, dim=args.dim)


def run_engine(
    engine_name: str,
    args: argparse.Namespace,
    dataset: Dataset,
    operation: Callable[[Engine], dict[str, Any]],
) -> dict[str, Any]:
    engine = fresh_engine(engine_name, args)
    row: dict[str, Any] = {
        "benchmark": args.command,
        "engine": engine_name,
        "n": args.n,
        "dim": args.dim,
        "queries": args.queries,
        "k": args.k,
    }
    rss_before = rss_mb()
    try:
        started = time.perf_counter()
        engine.setup()
        row["setup_s"] = time.perf_counter() - started
        row.update(operation(engine))
        row["disk_mb"] = dir_size_bytes(engine.path) / (1024 * 1024)
        rss_after = rss_mb()
        if rss_before is not None and rss_after is not None:
            row["rss_before_mb"] = rss_before
            row["rss_after_mb"] = rss_after
            row["rss_delta_mb"] = rss_after - rss_before
        row["status"] = "ok"
    except SkipEngine as exc:
        row["status"] = "skipped"
        row["reason"] = str(exc)
    except Exception as exc:
        row["status"] = "error"
        row["reason"] = repr(exc)
    finally:
        try:
            engine.close()
        finally:
            if not args.keep_data and engine.path.exists():
                shutil.rmtree(engine.path, ignore_errors=True)
    return row


def ingest_operation(args: argparse.Namespace, dataset: Dataset) -> Callable[[Engine], dict[str, Any]]:
    def op(engine: Engine) -> dict[str, Any]:
        started = time.perf_counter()
        engine.ingest(dataset, args.batch_size)
        elapsed = time.perf_counter() - started
        return {
            "ingest_s": elapsed,
            "vectors_per_s": dataset.vectors.shape[0] / elapsed if elapsed > 0 else 0.0,
        }

    return op


def prepared_operation(
    args: argparse.Namespace,
    dataset: Dataset,
    measured: Callable[[Engine], dict[str, Any]],
    *,
    requires_hybrid: bool = False,
) -> Callable[[Engine], dict[str, Any]]:
    def op(engine: Engine) -> dict[str, Any]:
        if requires_hybrid and not engine.supports_hybrid:
            raise SkipEngine(f"{engine.name} has no simple native hybrid adapter in this suite")
        started = time.perf_counter()
        engine.ingest(dataset, args.batch_size)
        ingest_s = time.perf_counter() - started
        result = {"prepare_ingest_s": ingest_s}
        result.update(measured(engine))
        return result

    return op


def query_measure(args: argparse.Namespace, dataset: Dataset) -> Callable[[Engine], dict[str, Any]]:
    ground_truth = [exact_topk(dataset.vectors, query, args.k) for query in dataset.queries]

    def measured(engine: Engine) -> dict[str, Any]:
        for query in dataset.queries[: args.warmup]:
            engine.search(query, args.k)
        latencies = []
        recalls = []
        for idx, query in enumerate(dataset.queries):
            started = time.perf_counter()
            ids = engine.search(query, args.k)
            latencies.append((time.perf_counter() - started) * 1000.0)
            recalls.append(recall_at_k(ids, ground_truth[idx], args.k))
        result = latency_stats_ms(latencies)
        result["recall_at_k"] = float(mean(recalls)) if recalls else 0.0
        return result

    return measured


def filtered_measure(args: argparse.Namespace, dataset: Dataset) -> Callable[[Engine], dict[str, Any]]:
    ground_truth = [
        exact_topk(
            dataset.vectors,
            query,
            args.k,
            categories=dataset.categories,
            category=int(dataset.query_categories[idx]),
        )
        for idx, query in enumerate(dataset.queries)
    ]

    def measured(engine: Engine) -> dict[str, Any]:
        for idx, query in enumerate(dataset.queries[: args.warmup]):
            engine.filtered_search(query, int(dataset.query_categories[idx]), args.k)
        latencies = []
        recalls = []
        for idx, query in enumerate(dataset.queries):
            category = int(dataset.query_categories[idx])
            started = time.perf_counter()
            ids = engine.filtered_search(query, category, args.k)
            latencies.append((time.perf_counter() - started) * 1000.0)
            recalls.append(recall_at_k(ids, ground_truth[idx], args.k))
        result = latency_stats_ms(latencies)
        result["filtered_recall_at_k"] = float(mean(recalls)) if recalls else 0.0
        return result

    return measured


def hybrid_measure(args: argparse.Namespace, dataset: Dataset) -> Callable[[Engine], dict[str, Any]]:
    def measured(engine: Engine) -> dict[str, Any]:
        if not engine.supports_hybrid:
            raise SkipEngine(f"{engine.name} has no simple native hybrid adapter in this suite")
        for idx, query in enumerate(dataset.queries[: args.warmup]):
            engine.hybrid_search(query, dataset.query_texts[idx], args.k)
        latencies = []
        topic_hits = []
        for idx, query in enumerate(dataset.queries):
            started = time.perf_counter()
            ids = engine.hybrid_search(query, dataset.query_texts[idx], args.k)
            latencies.append((time.perf_counter() - started) * 1000.0)
            category = int(dataset.query_categories[idx])
            if ids:
                topic_hits.append(
                    sum(int(item) % args.categories == category for item in ids) / len(ids)
                )
        result = latency_stats_ms(latencies)
        result["topic_hit_rate"] = float(mean(topic_hits)) if topic_hits else 0.0
        return result

    return measured


def resources_operation(args: argparse.Namespace, dataset: Dataset) -> Callable[[Engine], dict[str, Any]]:
    def op(engine: Engine) -> dict[str, Any]:
        before = rss_mb()
        started = time.perf_counter()
        engine.ingest(dataset, args.batch_size)
        elapsed = time.perf_counter() - started
        after = rss_mb()
        result = {
            "ingest_s": elapsed,
            "vectors_per_s": dataset.vectors.shape[0] / elapsed if elapsed > 0 else 0.0,
            "disk_mb": dir_size_bytes(engine.path) / (1024 * 1024),
        }
        if before is not None and after is not None:
            result["rss_before_ingest_mb"] = before
            result["rss_after_ingest_mb"] = after
            result["rss_ingest_delta_mb"] = after - before
        return result

    return op


def startup_operation(_args: argparse.Namespace, _dataset: Dataset) -> Callable[[Engine], dict[str, Any]]:
    def op(_engine: Engine) -> dict[str, Any]:
        return {}

    return op


def run_startup(engine_name: str, args: argparse.Namespace) -> dict[str, Any]:
    timings = []
    row = {
        "benchmark": args.command,
        "engine": engine_name,
        "iterations": args.startup_iterations,
        "n": args.n,
        "dim": args.dim,
    }
    for iteration in range(args.startup_iterations):
        path = Path(args.workdir) / f"startup-{engine_name}-{iteration}"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        engine = ENGINE_CLASSES[engine_name](path=path, dim=args.dim)
        try:
            started = time.perf_counter()
            engine.setup()
            timings.append((time.perf_counter() - started) * 1000.0)
        except SkipEngine as exc:
            row["status"] = "skipped"
            row["reason"] = str(exc)
            return row
        except Exception as exc:
            row["status"] = "error"
            row["reason"] = repr(exc)
            return row
        finally:
            try:
                engine.close()
            finally:
                if not args.keep_data:
                    shutil.rmtree(path, ignore_errors=True)
    row.update(latency_stats_ms(timings))
    row["status"] = "ok"
    return row


def print_rows(rows: list[dict[str, Any]]) -> None:
    columns = [
        "benchmark",
        "engine",
        "status",
        "vectors_per_s",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "qps",
        "recall_at_k",
        "filtered_recall_at_k",
        "topic_hit_rate",
        "disk_mb",
        "rss_delta_mb",
        "reason",
    ]
    active = [
        column
        for column in columns
        if any(column in row and row.get(column) not in (None, "") for row in rows)
    ]
    widths = {
        column: max(len(column), *(len(format_cell(row.get(column))) for row in rows))
        for column in active
    }
    print("  ".join(column.ljust(widths[column]) for column in active))
    print("  ".join("-" * widths[column] for column in active))
    for row in rows:
        print("  ".join(format_cell(row.get(column)).ljust(widths[column]) for column in active))


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_jsonl(path: str | None, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Chroma, LanceDB, LynseDB, and USEARCH on shared synthetic workloads.",
    )
    parser.add_argument(
        "command",
        choices=["ingest", "query", "recall", "filtered", "hybrid", "resources", "startup", "http", "all"],
    )
    parser.add_argument("--engines", default="all", help="Comma-separated engines or 'all'.")
    parser.add_argument("--n", type=int, default=100_000, help="Number of vectors.")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension.")
    parser.add_argument("--queries", type=int, default=100, help="Number of query vectors.")
    parser.add_argument("--k", type=int, default=10, help="Top-k.")
    parser.add_argument("--categories", type=int, default=100, help="Metadata category count.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Ingest batch size.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup queries.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="Working data directory.")
    parser.add_argument("--jsonl", default=None, help="Append machine-readable results to this JSONL file.")
    parser.add_argument("--keep-data", action="store_true", help="Keep engine data directories after the run.")
    parser.add_argument("--startup-iterations", type=int, default=5, help="Startup benchmark iterations.")
    return parser


def operations_for(command: str, args: argparse.Namespace, dataset: Dataset):
    if command == "ingest":
        return ingest_operation(args, dataset)
    if command in {"query", "recall"}:
        return prepared_operation(args, dataset, query_measure(args, dataset))
    if command == "filtered":
        return prepared_operation(args, dataset, filtered_measure(args, dataset))
    if command == "hybrid":
        return prepared_operation(
            args,
            dataset,
            hybrid_measure(args, dataset),
            requires_hybrid=True,
        )
    if command == "resources":
        return resources_operation(args, dataset)
    if command == "http":
        return prepared_operation(args, dataset, query_measure(args, dataset))
    raise ValueError(f"no operation for {command}")


def run_command(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.command == "startup":
        engines = parse_engines(args.engines)
        return [run_startup(engine, args) for engine in engines]

    dataset = make_dataset(
        n=args.n,
        dim=args.dim,
        n_queries=args.queries,
        n_categories=args.categories,
        seed=args.seed,
    )
    engines = parse_engines(args.engines, http_only=args.command == "http")
    if args.command == "http" and args.engines == "all":
        engines = ["lynsedb-http"]
    operation = operations_for(args.command, args, dataset)
    return [run_engine(engine, args, dataset, operation) for engine in engines]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    if args.command == "all":
        rows: list[dict[str, Any]] = []
        for command in ["ingest", "query", "recall", "filtered", "hybrid", "resources", "startup", "http"]:
            child = argparse.Namespace(**vars(args))
            child.command = command
            if command == "http":
                child.engines = "all"
            child_rows = run_command(child)
            rows.extend(child_rows)
            write_jsonl(args.jsonl, child_rows)
    else:
        rows = run_command(args)
        write_jsonl(args.jsonl, rows)

    print_rows(rows)
    return 0 if all(row.get("status") in {"ok", "skipped"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
