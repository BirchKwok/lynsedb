"""
Rust backend bridge for LynseDB.

This module provides a thin Python wrapper around the Rust `lynse_core` extension,
allowing the existing LynseDB Python API to optionally delegate to the high-performance
Rust implementation for distance computation, indexing, and storage.

Usage:
    from lynse._rust_backend import rust_available, RustEngine

    if rust_available():
        engine = RustEngine("/path/to/db")
        ...
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_RUST_AVAILABLE = False
_lynse_core = None

try:
    import lynse_core as _lynse_core
    _RUST_AVAILABLE = True
    logger.info("LynseDB Rust backend loaded successfully.")
except ImportError:
    logger.debug("Rust backend (lynse_core) not available; using pure-Python fallback.")


def rust_available() -> bool:
    """Check if the Rust backend is available."""
    return _RUST_AVAILABLE


# ─── Distance computation bridge ─────────────────────────────────────────────

def compute_distance(a: np.ndarray, b: np.ndarray, metric: str = "ip") -> float:
    """Compute distance between two vectors using Rust SIMD if available."""
    if _RUST_AVAILABLE:
        return _lynse_core.py_compute_distance(
            a.astype(np.float32, copy=False),
            b.astype(np.float32, copy=False),
            metric,
        )
    raise RuntimeError("Rust backend not available")


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
    if _RUST_AVAILABLE:
        return _lynse_core.py_top_k_search(
            query.astype(np.float32, copy=False),
            candidates.astype(np.float32, copy=False),
            metric,
            k,
        )
    raise RuntimeError("Rust backend not available")


# ─── Engine bridge ────────────────────────────────────────────────────────────

class RustEngine:
    """High-level wrapper around the Rust DatabaseEngine.

    Provides the same operations as the Python storage/index layers
    but delegates entirely to Rust for performance.
    """

    def __init__(self, root_path: str):
        if not _RUST_AVAILABLE:
            raise RuntimeError("Rust backend (lynse_core) is not installed.")
        self._engine = _lynse_core.DatabaseEngine(root_path)

    # ── Collection management ──

    def create_collection(
        self, name: str, dimension: int, chunk_size: int = 100_000
    ) -> "RustCollection":
        coll = self._engine.create_collection(name, dimension, chunk_size)
        return RustCollection(coll)

    def get_collection(
        self, name: str, dimension: int, chunk_size: int = 100_000
    ) -> "RustCollection":
        coll = self._engine.get_collection(name, dimension, chunk_size)
        return RustCollection(coll)

    def drop_collection(self, name: str) -> None:
        self._engine.drop_collection(name)

    def list_collections(self) -> List[str]:
        return self._engine.list_collections()

    def has_collection(self, name: str) -> bool:
        return self._engine.has_collection(name)

    @property
    def root_path(self) -> str:
        return self._engine.root_path()


class RustCollection:
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

    def build_index(self, index_type: str) -> None:
        """Build or rebuild the index.

        Args:
            index_type: e.g. "Flat-IP", "HNSW-L2", "IVF-Cos-SQ8", etc.
        """
        self._inner.build_index(index_type)

    def remove_index(self) -> None:
        self._inner.remove_index()

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        search_filter: Optional[str] = None,
        nprobe: int = 10,
    ) -> "RustSearchResult":
        """Search for nearest neighbors.

        Args:
            vector: query vector, shape (dim,), dtype float32.
            k: number of results.
            search_filter: optional SQL-like filter.
            nprobe: number of IVF probes.

        Returns:
            RustSearchResult with ids, distances, fields.
        """
        vector = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        result = self._inner.search(vector, k, search_filter, nprobe)
        return RustSearchResult(result)

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

    def delete(self) -> None:
        self._inner.delete()


class RustSearchResult:
    """Wrapper around Rust search results matching the existing Result API."""

    def __init__(self, inner):
        self._inner = inner

    @property
    def ids(self) -> np.ndarray:
        return self._inner.ids()

    @property
    def distances(self) -> np.ndarray:
        return self._inner.distances()

    @property
    def fields(self) -> List[Dict[str, Any]]:
        return list(self._inner.fields())

    @property
    def index_mode(self) -> str:
        return self._inner.index_mode()

    def to_tuple(self):
        return self._inner.to_tuple()

    def __len__(self) -> int:
        return len(self._inner)

    def __repr__(self) -> str:
        return repr(self._inner)
