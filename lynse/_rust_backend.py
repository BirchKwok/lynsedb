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

import json
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

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

    def batch_search(
        self,
        vectors: np.ndarray,
        k: int = 10,
        search_filter: Optional[str] = None,
        nprobe: int = 10,
    ) -> List["RustSearchResult"]:
        """Batch search: search multiple query vectors in parallel.

        Args:
            vectors: shape (n_queries, dim), dtype float32.
            k: number of results per query.
            search_filter: optional SQL-like filter.
            nprobe: number of IVF probes.

        Returns:
            List of RustSearchResult, one per query.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        results = self._inner.batch_search(vectors, k, search_filter, nprobe)
        return [RustSearchResult(r) for r in results]

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

    def head(self, n: int = 5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Return first n vectors + field metadata.

        Returns:
            (vectors_2d, fields_list) where vectors_2d has shape (n, dim).
        """
        flat_data, fields = self._inner.head(n)
        dim = self.dimension
        vectors = np.asarray(flat_data, dtype=np.float32).reshape(-1, dim)
        fields = list(fields)
        return vectors, fields

    def tail(self, n: int = 5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Return last n vectors + field metadata."""
        flat_data, fields = self._inner.tail(n)
        dim = self.dimension
        vectors = np.asarray(flat_data, dtype=np.float32).reshape(-1, dim)
        fields = list(fields)
        return vectors, fields

    def query_fields(self, filter_expr: str) -> List[int]:
        """Query field metadata with SQL-like filter. Returns matching IDs."""
        return self._inner.query_fields(filter_expr)

    def retrieve_fields(self, ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve field metadata for specific IDs."""
        return list(self._inner.retrieve_fields(ids))

    def list_fields(self) -> List[str]:
        """List all field names in the collection."""
        return self._inner.list_fields()

    def delete(self) -> None:
        self._inner.delete()


class RustSearchResult:
    """Wrapper around Rust search results matching the existing Python Result API.

    Provides the same rich format conversions as lynse.execution_layer.result.Result:
    to_dict, to_list, to_json, to_pandas, to_polars, to_arrow.
    """

    def __init__(self, inner):
        self._inner = inner
        self._ids = None
        self._distances = None
        self._fields = None

    @property
    def ids(self) -> np.ndarray:
        if self._ids is None:
            self._ids = self._inner.ids()
        return self._ids

    @property
    def distances(self) -> np.ndarray:
        if self._distances is None:
            self._distances = self._inner.distances()
        return self._distances

    @property
    def fields(self) -> List[Dict[str, Any]]:
        if self._fields is None:
            self._fields = list(self._inner.fields())
        return self._fields

    @property
    def index_mode(self) -> str:
        return self._inner.index_mode()

    @property
    def res_num(self) -> int:
        return len(self)

    def to_tuple(self):
        return self._inner.to_tuple()

    def __iter__(self):
        """Support tuple unpacking: ids, distances, fields = result."""
        yield self.ids
        yield self.distances
        yield self.fields

    def __getitem__(self, index):
        """Support indexing for backward compatibility."""
        if index == 0:
            return self.ids
        elif index == 1:
            return self.distances
        elif index == 2:
            return self.fields
        raise IndexError(f"Index {index} out of range (0-2)")

    def __len__(self) -> int:
        return len(self._inner)

    def __repr__(self) -> str:
        return (
            f"RustSearchResult(n={len(self)}, "
            f"index_mode={self.index_mode!r})"
        )

    # ─── Rich format conversions (matching Python Result class) ───────────

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert results to a list of dicts.

        Each dict contains 'id', 'distance', and any field keys.
        """
        result = []
        ids = self.ids
        dists = self.distances
        fields = self.fields
        for i in range(len(ids)):
            entry = {
                "id": int(ids[i]),
                "distance": float(dists[i]),
            }
            if i < len(fields) and fields[i]:
                entry.update(fields[i])
            result.append(entry)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dict of arrays."""
        d = {
            "ids": self.ids.tolist(),
            "distances": self.distances.tolist(),
        }
        if self.fields:
            # Collect all field keys
            all_keys = set()
            for f in self.fields:
                if f:
                    all_keys.update(f.keys())
            for key in sorted(all_keys):
                d[key] = [f.get(key) if f else None for f in self.fields]
        return d

    def to_json(self, **kwargs) -> str:
        """Convert results to a JSON string."""
        return json.dumps(self.to_list(), **kwargs)

    def to_pandas(self):
        """Convert results to a pandas DataFrame.

        Requires pandas to be installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_pandas(). "
                "Install it with: pip install pandas"
            )
        return pd.DataFrame(self.to_list())

    def to_polars(self):
        """Convert results to a polars DataFrame.

        Requires polars to be installed.
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is required for to_polars(). "
                "Install it with: pip install polars"
            )
        return pl.DataFrame(self.to_list())

    def to_arrow(self):
        """Convert results to a PyArrow Table.

        Requires pyarrow to be installed.
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                "pyarrow is required for to_arrow(). "
                "Install it with: pip install pyarrow"
            )
        data = self.to_dict()
        arrays = {}
        for k, v in data.items():
            if k == "ids":
                arrays[k] = pa.array(v, type=pa.int64())
            elif k == "distances":
                arrays[k] = pa.array(v, type=pa.float32())
            else:
                arrays[k] = pa.array(v)
        return pa.table(arrays)
