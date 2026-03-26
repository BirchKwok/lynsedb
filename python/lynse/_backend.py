"""
Rust backend bridge for LynseDB.

This module provides a thin Python wrapper around the Rust `lynse._core` extension,
allowing the existing LynseDB Python API to optionally delegate to the high-performance
Rust implementation for distance computation, indexing, and storage.

Usage:
    from lynse._backend import rust_available, RustEngine

    if rust_available():
        engine = RustEngine("/path/to/db")
        ...
"""

import json
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np

from .result_view import ResultView, _parse_index_mode

logger = logging.getLogger(__name__)

from . import _core as _lynse


# ─── Server management ──────────────────────────────────────────────────────

_server_thread = None
_server_lock = threading.Lock()


def start_server(host: str = "127.0.0.1", port: int = 7637, root_path: str = ".") -> None:
    """Start the Rust HTTP server (blocking). Call from a background thread."""
    _lynse.py_start_server(host, port, root_path)


def start_server_background(host: str = "127.0.0.1", port: int = 7637, root_path: str = ".") -> None:
    """Start the Rust HTTP server in a background thread.

    The server runs until the process exits. Safe to call multiple times
    (subsequent calls are no-ops if a server is already running).
    """
    global _server_thread
    with _server_lock:
        if _server_thread is not None and _server_thread.is_alive():
            return
        _server_thread = threading.Thread(
            target=start_server, args=(host, port, root_path), daemon=True
        )
        _server_thread.start()
        # Give server time to bind
        import time
        time.sleep(0.3)


# ─── DatabaseManager bridge ─────────────────────────────────────────────────

class DatabaseManager:
    """High-level wrapper around the Rust DatabaseManager.

    Manages multiple databases, each containing multiple collections.
    """

    def __init__(self, root_path: str):
        self._manager = _lynse.DatabaseManager(root_path)

    def create_database(self, name: str) -> None:
        self._manager.create_database(name)

    def drop_database(self, name: str) -> None:
        self._manager.drop_database(name)

    def list_databases(self) -> List[str]:
        return self._manager.list_databases()

    def database_exists(self, name: str) -> bool:
        return self._manager.database_exists(name)

    def show_collections(self, db_name: str) -> List[str]:
        return self._manager.show_collections(db_name)

    def require_collection(
        self,
        db_name: str,
        collection_name: str,
        dim: int,
        drop_if_exists: bool = False,
        description: Optional[str] = None,
    ) -> None:
        self._manager.require_collection(
            db_name, collection_name, dim, drop_if_exists, description
        )

    def drop_collection(self, db_name: str, collection_name: str) -> None:
        self._manager.drop_collection(db_name, collection_name)

    def collection_exists(self, db_name: str, collection_name: str) -> bool:
        return self._manager.collection_exists(db_name, collection_name)

    def update_collection_description(
        self, db_name: str, collection_name: str, description: Optional[str] = None
    ) -> None:
        self._manager.update_collection_description(db_name, collection_name, description)

    def get_collection(
        self,
        db_name: str,
        collection_name: str,
        dim: int,
    ) -> "Collection":
        """Get a PyCollection directly from the database manager."""
        coll = self._manager.get_collection(db_name, collection_name, dim)
        return Collection(coll)

    def get_collection_config(
        self,
        db_name: str,
        collection_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get collection config (dim, description) from collections.json."""
        result = self._manager.get_collection_config(db_name, collection_name)
        if result is None:
            return None
        dim, _chunk_size, description = result
        return {"dim": dim, "description": description}

    @property
    def root_path(self) -> str:
        return self._manager.root_path()


# ─── Distance computation bridge ─────────────────────────────────────────────

def compute_distance(a: np.ndarray, b: np.ndarray, metric: str = "ip") -> float:
    """Compute distance between two vectors using Rust SIMD."""
    return _lynse.py_compute_distance(
        a.astype(np.float32, copy=False),
        b.astype(np.float32, copy=False),
        metric,
    )


def top_k_search(
    query: np.ndarray,
    candidates: np.ndarray,
    metric: str = "ip",
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k search using Rust if available.

    Returns:
        (ids, distances) arrays sorted by relevance.
    """
    return _lynse.py_top_k_search(
        query.astype(np.float32, copy=False),
        candidates.astype(np.float32, copy=False),
        metric,
        k,
    )


# ─── Engine bridge ────────────────────────────────────────────────────────────

class RustEngine:
    """High-level wrapper around the Rust DatabaseEngine.

    Provides the same operations as the Python storage/index layers
    but delegates entirely to Rust for performance.
    """

    def __init__(self, root_path: str):
        self._engine = _lynse.DatabaseEngine(root_path)

    # ── Collection management ──

    def create_collection(
        self, name: str, dimension: int,
    ) -> "Collection":
        coll = self._engine.create_collection(name, dimension)
        return Collection(coll)

    def get_collection(
        self, name: str, dimension: int,
    ) -> "Collection":
        coll = self._engine.get_collection(name, dimension)
        return Collection(coll)

    def drop_collection(self, name: str) -> None:
        self._engine.drop_collection(name)

    def list_collections(self) -> List[str]:
        return self._engine.list_collections()

    def has_collection(self, name: str) -> bool:
        return self._engine.has_collection(name)

    @property
    def root_path(self) -> str:
        return self._engine.root_path()


class Collection:
    """Wrapper around a single Rust Collection."""

    def __init__(self, inner):
        self._inner = inner

    def add_items(
        self,
        vectors: np.ndarray,
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors (and optional per-vector field metadata).

        Args:
            vectors: shape (n, dim), dtype float32.
            fields: optional list of dicts, one per vector.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._inner.add_items(vectors, fields)

    def build_index(self, index_mode: str, **kwargs) -> None:
        """Build or rebuild the index.

        Args:
            index_mode (str): The index mode, must be one of the following:

                - 'FLAT': Flat index with inner product. (Default)
                - 'FLAT-L2': Flat index with squared L2 distance.
                - 'FLAT-COS': Flat index with cosine similarity.
                - 'FLAT-IP-SQ8': Flat index with inner product and SQ8 quantizer.
                - 'FLAT-L2-SQ8': Flat index with squared L2 distance and SQ8 quantizer.
                - 'FLAT-COS-SQ8': Flat index with cosine similarity and SQ8 quantizer.
                - 'FLAT-JACCARD-BINARY': Flat index with Jaccard distance (binary).
                - 'FLAT-HAMMING-BINARY': Flat index with Hamming distance (binary).
                - 'IVF': IVF index with inner product.
                - 'IVF-L2': IVF index with squared L2 distance.
                - 'IVF-COS': IVF index with cosine similarity.
                - 'IVF-IP-SQ8': IVF index with inner product and SQ8 quantizer.
                - 'IVF-L2-SQ8': IVF index with squared L2 distance and SQ8 quantizer.
                - 'IVF-COS-SQ8': IVF index with cosine similarity and SQ8 quantizer.
                - 'IVF-JACCARD-BINARY': IVF index with Jaccard distance (binary).
                - 'IVF-HAMMING-BINARY': IVF index with Hamming distance (binary).
            **kwargs: Additional keyword arguments:
                - 'n_clusters' (int): Number of clusters (IVF modes only).
        """
        self._inner.build_index(index_mode)

    def remove_index(self) -> None:
        self._inner.remove_index()

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        where: Optional[str] = None,
        nprobe: int = 10,
    ) -> ResultView:
        """Search for nearest neighbors.

        Args:
            vector: query vector, shape (dim,), dtype float32.
            k: number of results.
            where: optional SQL-like filter.
            nprobe: number of IVF probes.

        Returns:
            ResultView with ids, distances.
        """
        vector = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        result = self._inner.search(vector, k, where, nprobe)
        ids = result.ids()
        distances = result.distances()
        idx_type, metric = _parse_index_mode(result.index_mode())
        return ResultView(
            ids=ids, distances=distances,
            k=k, distance=metric, index=idx_type,
            result_type="search",
        )

    def retrieve_fields(self, ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve fields for given IDs (lazy loading after search)."""
        return self._inner.retrieve_fields(ids)

    def batch_search(
        self,
        vectors: np.ndarray,
        k: int = 10,
        where: Optional[str] = None,
        nprobe: int = 10,
    ) -> List[ResultView]:
        """Batch search: search multiple query vectors in parallel.

        Args:
            vectors: shape (n_queries, dim), dtype float32.
            k: number of results per query.
            where: optional SQL-like filter.
            nprobe: number of IVF probes.

        Returns:
            List of ResultView, one per query.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        results = self._inner.batch_search(vectors, k, where, nprobe)
        output = []
        for r in results:
            idx_type, metric = _parse_index_mode(r.index_mode())
            output.append(ResultView(
                ids=r.ids(), distances=r.distances(),
                k=k, distance=metric, index=idx_type,
                result_type="search",
            ))
        return output

    def update_items(
        self,
        ids: List[int],
        vectors: np.ndarray,
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update vectors by IDs: atomic delete + insert.

        Args:
            ids: list of vector IDs to update.
            vectors: shape (n, dim), dtype float32.
            fields: optional list of dicts, one per vector.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._inner.update_items(ids, vectors, fields)

    def commit(self) -> None:
        """Commit: clear WAL after successful writes."""
        self._inner.commit()

    def has_uncommitted_data(self) -> bool:
        """Check if there is uncommitted WAL data."""
        return self._inner.has_uncommitted_data()

    def sync_index(self) -> None:
        """Sync the index with any new data since last sync."""
        self._inner.sync_index()

    @property
    def fingerprint(self) -> str:
        """Get the current storage fingerprint."""
        return self._inner.fingerprint()

    @property
    def shape(self) -> Tuple[int, int]:
        return self._inner.shape()

    @property
    def name(self) -> str:
        return self._inner.name()

    @property
    def dimension(self) -> int:
        return self._inner.dimension()

    @property
    def index_mode(self) -> Optional[str]:
        return self._inner.index_mode()

    def head(self, n: int = 5) -> ResultView:
        """Return first n vectors + field metadata.

        Returns:
            ResultView with vectors, ids, fields.
        """
        flat_data, fields = self._inner.head(n)
        dim = self.dimension
        vectors = np.asarray(flat_data, dtype=np.float32).reshape(-1, dim)
        actual_n = vectors.shape[0]
        ids = np.arange(actual_n, dtype=np.int64)
        return ResultView(
            vectors=vectors, ids=ids, fields=list(fields),
            result_type="data",
        )

    def tail(self, n: int = 5) -> ResultView:
        """Return last n vectors + field metadata."""
        flat_data, fields = self._inner.tail(n)
        dim = self.dimension
        vectors = np.asarray(flat_data, dtype=np.float32).reshape(-1, dim)
        actual_n = vectors.shape[0]
        total = self.shape[0]
        ids = np.arange(max(0, total - actual_n), total, dtype=np.int64)
        return ResultView(
            vectors=vectors, ids=ids, fields=list(fields),
            result_type="data",
        )

    def query_fields(self, where: str) -> List[int]:
        """Query field metadata with SQL-like filter. Returns matching IDs."""
        return self._inner.query_fields(where)

    def query_with_fields(self, where: str) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Query field metadata with SQL-like filter. Returns (ids, fields) in one call."""
        ids, fields = self._inner.query_with_fields(where)
        return ids, list(fields)

    def retrieve_fields(self, ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve field metadata for specific IDs."""
        return list(self._inner.retrieve_fields(ids))

    def list_fields(self) -> List[str]:
        """List all field names in the collection."""
        return self._inner.list_fields()

    def delete(self) -> None:
        self._inner.delete()


# SearchResult is replaced by ResultView from result_view.py.
# Keep a backward-compatible alias.
SearchResult = ResultView
