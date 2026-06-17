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
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np

from .result_view import ResultView, _parse_index_mode

logger = logging.getLogger(__name__)

from . import _core as _lynse


def _normalize_sparse_vector(vector: Any) -> List[Tuple[int, float]]:
    """Normalize a sparse vector from {dim: value} or [(dim, value), ...]."""
    items = vector.items() if isinstance(vector, dict) else vector
    normalized: List[Tuple[int, float]] = []
    for item in items:
        if len(item) != 2:
            raise ValueError("sparse vector entries must be (index, value) pairs")
        index, value = item
        index = int(index)
        if index < 0:
            raise ValueError("sparse vector indices must be non-negative")
        normalized.append((index, float(value)))
    return normalized


# ─── Server management ──────────────────────────────────────────────────────

_server_thread = None
_server_lock = threading.Lock()


def start_server(host: str = "127.0.0.1", port: int = 7637, root_path: str = ".",
                 api_key: str = None) -> None:
    """Start the Rust HTTP server (blocking). Call from a background thread."""
    _lynse.py_start_server(host, port, root_path, api_key)


def start_server_background(host: str = "127.0.0.1", port: int = 7637, root_path: str = ".",
                            api_key: str = None) -> None:
    """Start the Rust HTTP server in a background thread.

    The server runs until the process exits. Safe to call multiple times
    (subsequent calls are no-ops if a server is already running).
    """
    global _server_thread
    with _server_lock:
        if _server_thread is not None and _server_thread.is_alive():
            return
        _server_thread = threading.Thread(
            target=start_server, args=(host, port, root_path, api_key), daemon=True
        )
        _server_thread.start()
        # Give server time to bind
        import time
        time.sleep(0.3)


# ─── DatabaseManager bridge ─────────────────────────────────────────────────

# One Rust DatabaseManager (and its flock) per resolved root path per process.
# Re-opening the same path in notebooks / REPLs reuses the handle instead of
# failing with "path is already open by another writer".
_MANAGER_CACHE: Dict[Tuple[str, bool], Any] = {}
_MANAGER_REFS: Dict[Tuple[str, bool], int] = {}
_MANAGER_CACHE_LOCK = threading.Lock()


def _manager_cache_key(root_path: str, read_only: bool) -> Tuple[str, bool]:
    return (str(Path(root_path).resolve()), read_only)


class DatabaseManager:
    """High-level wrapper around the Rust DatabaseManager.

    Manages multiple databases, each containing multiple collections.
    """

    def __init__(self, root_path: str, read_only: bool = False):
        self._root_path = str(Path(root_path).resolve())
        self._read_only = read_only
        key = _manager_cache_key(self._root_path, read_only)

        with _MANAGER_CACHE_LOCK:
            cached_inner = _MANAGER_CACHE.get(key)
            if cached_inner is not None:
                self._manager = cached_inner
                _MANAGER_REFS[key] = _MANAGER_REFS.get(key, 0) + 1
                return

            if read_only:
                self._manager = _lynse.DatabaseManager.open_read_only(self._root_path)
            else:
                self._manager = _lynse.DatabaseManager(self._root_path)
            _MANAGER_CACHE[key] = self._manager
            _MANAGER_REFS[key] = 1

    def close(self) -> None:
        """Release this handle; drop the process lock when the last handle closes."""
        if self._manager is None:
            return
        key = _manager_cache_key(self._root_path, self._read_only)
        with _MANAGER_CACHE_LOCK:
            refs = _MANAGER_REFS.get(key, 0) - 1
            if refs <= 0:
                _MANAGER_CACHE.pop(key, None)
                _MANAGER_REFS.pop(key, None)
            else:
                _MANAGER_REFS[key] = refs
        self._manager = None

    def create_database(self, name: str) -> None:
        self._manager.create_database(name)

    def drop_database(self, name: str) -> None:
        self._manager.drop_database(name)

    def snapshot_database(self, name: str, snapshot_path: str) -> None:
        self._manager.snapshot_database(name, snapshot_path)

    def restore_database(
        self,
        name: str,
        snapshot_path: str,
        overwrite: bool = False,
    ) -> None:
        self._manager.restore_database(name, snapshot_path, overwrite)

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
        dim: Optional[int],
        drop_if_exists: bool = False,
        description: Optional[str] = None,
        dtypes: str = "float32",
    ) -> None:
        self._manager.require_collection(
            db_name,
            collection_name,
            0 if dim is None else dim,
            drop_if_exists,
            description,
            dtypes,
        )

    def drop_collection(self, db_name: str, collection_name: str) -> None:
        self._manager.drop_collection(db_name, collection_name)

    def snapshot_collection(self, db_name: str, collection_name: str, snapshot_path: str) -> None:
        self._manager.snapshot_collection(db_name, collection_name, snapshot_path)

    def export_collection(self, db_name: str, collection_name: str, export_path: str) -> None:
        self._manager.export_collection(db_name, collection_name, export_path)

    def restore_collection(
        self,
        db_name: str,
        collection_name: str,
        snapshot_path: str,
        overwrite: bool = False,
    ) -> None:
        self._manager.restore_collection(db_name, collection_name, snapshot_path, overwrite)

    def import_collection(
        self,
        db_name: str,
        collection_name: str,
        export_path: str,
        overwrite: bool = False,
    ) -> None:
        self._manager.import_collection(db_name, collection_name, export_path, overwrite)

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
        dim: Optional[int],
    ) -> "Collection":
        """Get a PyCollection directly from the database manager."""
        coll = self._manager.get_collection(db_name, collection_name, 0 if dim is None else dim)
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
        dim, _chunk_size, description, dtypes = result
        return {"dim": dim, "description": description, "dtypes": dtypes}

    @property
    def root_path(self) -> str:
        return self._manager.root_path()

    @property
    def is_read_only(self) -> bool:
        return self._manager.is_read_only()


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

    def __init__(self, root_path: str, read_only: bool = False):
        if read_only:
            self._engine = _lynse.DatabaseEngine.open_read_only(root_path)
        else:
            self._engine = _lynse.DatabaseEngine(root_path)

    # ── Collection management ──

    def create_collection(
        self, name: str, dimension: int, dtypes: str = "float32",
    ) -> "Collection":
        coll = self._engine.create_collection(name, dimension, dtypes)
        return Collection(coll)

    def get_collection(
        self, name: str, dimension: int, dtypes: Optional[str] = None,
    ) -> "Collection":
        coll = self._engine.get_collection(name, dimension, dtypes)
        return Collection(coll)

    def drop_collection(self, name: str) -> None:
        self._engine.drop_collection(name)

    def snapshot_collection(self, name: str, snapshot_path: str) -> None:
        """Snapshot a collection to a file path.
        
        The snapshot is a point-in-time copy of the collection's data and index files,
        which can be used for backup or restore. The snapshot is stored as a single file
        that contains the collection's state at the time of snapshotting.
        Args:
            name: The name of the collection to snapshot.
            snapshot_path: The file path where the snapshot will be saved.
        
        """
        self._engine.snapshot_collection(name, snapshot_path)

    def export_collection(self, name: str, export_path: str) -> None:
        self._engine.export_collection(name, export_path)

    def restore_collection(
        self,
        name: str,
        snapshot_path: str,
        overwrite: bool = False,
    ) -> None:
        self._engine.restore_collection(name, snapshot_path, overwrite)

    def import_collection(
        self,
        name: str,
        export_path: str,
        overwrite: bool = False,
    ) -> None:
        self._engine.import_collection(name, export_path, overwrite)

    def list_collections(self) -> List[str]:
        return self._engine.list_collections()

    def has_collection(self, name: str) -> bool:
        return self._engine.has_collection(name)

    @property
    def root_path(self) -> str:
        return self._engine.root_path()

    @property
    def is_read_only(self) -> bool:
        return self._engine.is_read_only()


class Collection:
    """Wrapper around a single Rust Collection."""

    def __init__(self, inner):
        self._inner = inner

    @property
    def is_read_only(self) -> bool:
        return self._inner.is_read_only()

    def add_items(
        self,
        vectors: np.ndarray,
        ids: List[int],
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors with user-specified IDs and optional per-vector field metadata.

        Args:
            vectors: shape (n, dim), dtype float32.
            ids: list of integer user IDs, one per vector.
            fields: optional list of dicts, one per vector.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._inner.add_items(vectors, [int(i) for i in ids], fields)

    def add_items_encoded_f16(
        self,
        vectors: np.ndarray,
        ids: List[int],
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors that are already encoded as IEEE float16 bits."""
        encoded = np.ascontiguousarray(vectors, dtype=np.uint16)
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        self._inner.add_items_encoded_f16(encoded, [int(i) for i in ids], fields)

    def add_records(
        self,
        vectors: np.ndarray,
        ids: List[Union[str, int]],
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Union[str, int]]:
        """Add records with public string/integer IDs.

        Rust assigns internal integer IDs and stores the external-ID map.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return list(self._inner.add_records(vectors, list(ids), fields))

    def external_ids(self, ids: List[int]) -> List[Union[str, int]]:
        """Convert internal numeric IDs to public external IDs."""
        return list(self._inner.external_ids([int(i) for i in ids]))

    def internal_ids(self, ids: List[Union[str, int]]) -> List[int]:
        """Convert public external IDs to internal numeric IDs."""
        return [int(i) for i in self._inner.internal_ids(list(ids))]

    def is_external_id_exists(self, id: Union[str, int]) -> bool:
        """Check whether a public external ID exists."""
        return bool(self._inner.is_external_id_exists(id))

    def build_index(
        self,
        index_mode: str,
        field_name: str = "default",
        n_clusters: Optional[int] = None,
    ) -> None:
        """Build or rebuild the index.

        Args:
            index_mode (str): The index mode, must be one of the following:

                **Flat (brute-force):**

                - 'FLAT-IP': Flat index with inner product. (Default)
                - 'Flat-L2': Flat index with squared L2 distance.
                - 'Flat-Cos': Flat index with cosine similarity.
                - 'Flat-IP-SQ8': Flat index with inner product and SQ8 quantizer.
                - 'Flat-L2-SQ8': Flat index with squared L2 distance and SQ8 quantizer.
                - 'Flat-Cos-SQ8': Flat index with cosine similarity and SQ8 quantizer.
                - 'Flat-Jaccard-Binary': Flat index with Jaccard distance (binary vectors).
                - 'Flat-Hamming-Binary': Flat index with Hamming distance (binary vectors).

                **Flat + PQ (Product Quantization, two-pass ADC search):**

                - 'FLAT-IP-PQ': PQ with inner product (auto subspace count).
                - 'FLAT-L2-PQ': PQ with squared L2 distance.
                - 'FLAT-COS-PQ': PQ with cosine similarity.
                - 'FLAT-IP-PQ8': PQ with inner product and 8 subspaces.
                - 'FLAT-IP-PQ16': PQ with inner product and 16 subspaces.
                - 'FLAT-L2-PQ8': PQ with squared L2 and 8 subspaces.

                **Flat + RaBitQ (Randomized Binary Quantization, ~32x compression):**

                - 'FLAT-IP-RABITQ': RaBitQ with inner product.
                - 'FLAT-L2-RABITQ': RaBitQ with squared L2 distance.
                - 'FLAT-COS-RABITQ': RaBitQ with cosine similarity.

                **HNSW (graph-based ANN):**

                - 'HNSW-IP': HNSW index with inner product.
                - 'HNSW-L2': HNSW index with squared L2 distance.
                - 'HNSW-Cos': HNSW index with cosine similarity.
                - 'HNSW-IP-SQ8': HNSW index with inner product and SQ8 quantizer.
                - 'HNSW-L2-SQ8': HNSW index with squared L2 distance and SQ8 quantizer.
                - 'HNSW-Cos-SQ8': HNSW index with cosine similarity and SQ8 quantizer.

                **DiskANN (disk-friendly graph ANN):**

                - 'DiskANN-IP': DiskANN index with inner product.
                - 'DiskANN-L2': DiskANN index with squared L2 distance.
                - 'DiskANN-Cos': DiskANN index with cosine similarity.
                - 'DiskANN-IP-SQ8': DiskANN index with inner product and SQ8 quantizer.
                - 'DiskANN-L2-SQ8': DiskANN index with squared L2 distance and SQ8 quantizer.
                - 'DiskANN-Cos-SQ8': DiskANN index with cosine similarity and SQ8 quantizer.

                **SPANN (space-partition ANN):**

                - 'SPANN-IP': SPANN index with inner product.
                - 'SPANN-L2': SPANN index with squared L2 distance.
                - 'SPANN-Cos': SPANN index with cosine similarity.
                - 'SPANN-IP-SQ8': SPANN index with inner product and SQ8 quantizer.
                - 'SPANN-L2-SQ8': SPANN index with squared L2 distance and SQ8 quantizer.
                - 'SPANN-Cos-SQ8': SPANN index with cosine similarity and SQ8 quantizer.

                **IVF (inverted file ANN):**

                - 'IVF-IP': IVF index with inner product.
                - 'IVF-L2': IVF index with squared L2 distance.
                - 'IVF-Cos': IVF index with cosine similarity.
                - 'IVF-IP-SQ8': IVF index with inner product and SQ8 quantizer.
                - 'IVF-L2-SQ8': IVF index with squared L2 distance and SQ8 quantizer.
                - 'IVF-Cos-SQ8': IVF index with cosine similarity and SQ8 quantizer.
                - 'IVF-Jaccard-Binary': IVF index with Jaccard distance (binary vectors).
                - 'IVF-Hamming-Binary': IVF index with Hamming distance (binary vectors).

            field_name (str): Named vector field to build index for.
                Defaults to "default" (the primary collection vector).
            n_clusters (int, optional): Number of clusters. IVF and SPANN modes
                use it; other index modes silently ignore it.
        """
        effective_n_clusters = (
            n_clusters
            if index_mode.upper().startswith(("IVF", "SPANN"))
            else None
        )

        if field_name == "default":
            self._inner.build_index(index_mode, effective_n_clusters)
        else:
            self._inner.build_vector_field_index(field_name, index_mode, effective_n_clusters)

    def remove_index(self, field_name: str = "default") -> None:
        """Remove the index.

        Args:
            field_name (str): Named vector field to remove index for.
                Defaults to "default" (the primary collection index).
        """
        if field_name == "default":
            self._inner.remove_index()
        else:
            self._inner.remove_vector_field_index(field_name)

    def create_vector_field(
        self,
        name: str,
        dimension: int,
        metric: Optional[str] = None,
        index_mode: Optional[str] = None,
        dtypes: Optional[str] = None,
    ) -> None:
        """Create a named vector field with its own dimension and metric."""
        self._inner.create_vector_field(name, int(dimension), metric, index_mode, dtypes)

    def list_vector_fields(self) -> List[Dict[str, Any]]:
        """List vector fields, including the reserved default primary vector."""
        return [dict(field) for field in self._inner.list_vector_fields()]

    def add_named_vectors(
        self,
        field_name: str,
        vectors: np.ndarray,
        ids: List[int],
    ) -> None:
        """Attach vectors to a named vector field for existing IDs."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._inner.add_named_vectors(field_name, vectors, [int(i) for i in ids])

    def add_sparse_vectors(
        self,
        vectors: List[Union[Dict[int, float], List[Tuple[int, float]]]],
        ids: List[int],
    ) -> None:
        """Attach sparse feature vectors to existing IDs."""
        normalized = [_normalize_sparse_vector(vector) for vector in vectors]
        self._inner.add_sparse_vectors(normalized, [int(i) for i in ids])

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        where: Optional[str] = None,
        field_name: str = "default",
        nprobe: int = 10,
        approx: bool = False,
        eps: float = 1e-4,
    ) -> ResultView:
        """Search for nearest neighbors.

        Args:
            vector: query vector, shape (dim,), dtype float32.
            k: number of results.
            where: optional SQL-like filter.
            field_name: named vector field to search. Defaults to "default"
                (the primary collection vector).
            nprobe: controls search breadth by index type (default: 10).
                - **IVF / SPANN**: number of partitions to probe — higher = better recall, slower.
                - **HNSW**: ef_search beam width — higher = better recall, slower.
                - **Flat / PQ / RaBitQ**: ignored (exhaustive two-pass search).
            approx: if True, use metric-specific flat approximation for IP, L2,
                and Cosine. Ignored for Hamming/Jaccard, which always use exact
                binary-distance search.
            eps: distance rounding tolerance when ``approx=True`` for supported
                metrics (default 1e-4).

        Returns:
            ResultView with ids, distances.
        """
        vector = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        if field_name == "default":
            result = self._inner.search(vector, k, where, nprobe, approx, eps)
        else:
            result = self._inner.search_vector_field(field_name, vector, k, where, approx, eps)
        ids = result.ids()
        distances = result.distances()
        idx_type, metric = _parse_index_mode(result.index_mode())
        return ResultView(
            ids=ids, distances=distances,
            k=k, distance=metric, index=idx_type,
            result_type="search",
        )

    def search_sparse(
        self,
        vector: Union[Dict[int, float], List[Tuple[int, float]]],
        k: int = 10,
        where: Optional[str] = None,
    ) -> ResultView:
        """Search sparse vectors with inner product."""
        normalized = _normalize_sparse_vector(vector)
        result = self._inner.search_sparse(normalized, k, where)
        return ResultView(
            ids=result.ids(),
            distances=result.distances(),
            k=k,
            distance="IP",
            index=result.index_mode(),
            result_type="search",
        )

    def search_profile(
        self,
        vector: np.ndarray,
        k: int = 10,
        where: Optional[str] = None,
        nprobe: int = 10,
    ) -> Dict[str, Any]:
        """Search and return profile/explain metadata."""
        vector = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        return self._inner.search_profile(vector, k, where, nprobe)

    def text_search(
        self,
        text: str,
        text_fields: Optional[List[str]] = None,
        k: int = 10,
        where: Optional[str] = None,
    ) -> ResultView:
        """BM25 text search over metadata fields."""
        result = self._inner.text_search(text, text_fields, k, where)
        ids = result.ids()
        distances = result.distances()
        return ResultView(
            ids=ids, distances=distances,
            k=k, distance="bm25", index=result.index_mode(),
            result_type="search",
        )

    def hybrid_search(
        self,
        vector: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        k: int = 10,
        where: Optional[str] = None,
        text_fields: Optional[List[str]] = None,
        fusion: str = "rrf",
        vector_weight: float = 1.0,
        text_weight: float = 1.0,
        rrf_k: float = 60.0,
        candidate_limit: Optional[int] = None,
        nprobe: int = 10,
    ) -> ResultView:
        """Hybrid vector + BM25 text search with RRF or weighted fusion."""
        vec = None if vector is None else np.ascontiguousarray(vector, dtype=np.float32).ravel()
        result = self._inner.hybrid_search(
            vec, text, k, where, text_fields, fusion,
            float(vector_weight), float(text_weight), float(rrf_k), candidate_limit, nprobe,
        )
        ids = result.ids()
        distances = result.distances()
        return ResultView(
            ids=ids, distances=distances,
            k=k, distance="fusion", index=result.index_mode(),
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
            nprobe: controls search breadth by index type (default: 10).
                - **IVF / SPANN**: number of partitions to probe — higher = better recall, slower.
                - **HNSW**: ef_search beam width — higher = better recall, slower.
                - **Flat / PQ / RaBitQ**: ignored (exhaustive two-pass search).

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

    def upsert_items(
        self,
        ids: List[int],
        vectors: np.ndarray,
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or update vectors by IDs.

        Existing IDs are updated in place. Missing IDs are inserted.
        If fields are provided, each field map replaces the row's current fields.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._inner.upsert_items([int(i) for i in ids], vectors, fields)

    def commit(self) -> None:
        """Commit: clear WAL after successful writes without recursive fsync."""
        self._inner.commit()

    def checkpoint_fast(self) -> None:
        """Lightweight checkpoint: clear WAL without forcing recursive fsync."""
        self._inner.checkpoint_fast()

    def flush(self) -> None:
        """Flush pending bytes and fsync collection files without clearing WAL."""
        self._inner.flush()

    def checkpoint(self) -> None:
        """Checkpoint durable state and clear WAL."""
        self._inner.checkpoint()

    def close(self) -> None:
        """Flush and close the collection handle from an API perspective."""
        self._inner.close()

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
    def vector_dtype(self) -> str:
        return self._inner.vector_dtype()

    @property
    def index_mode(self) -> Optional[str]:
        return self._inner.index_mode()

    def get_index_mode(self) -> Optional[str]:
        """Return the current index mode (callable form)."""
        return self._inner.index_mode()

    def head(self, n: int = 5) -> ResultView:
        """Return first n vectors + field metadata.

        Returns:
            ResultView with vectors, ids, fields.
        """
        flat_data, ids_raw, fields = self._inner.head(n)
        dim = self.dimension
        vectors = np.asarray(flat_data, dtype=np.float32).reshape(-1, dim)
        ids = np.asarray(ids_raw, dtype=np.int64)
        return ResultView(
            vectors=vectors, ids=ids, fields=list(fields),
            result_type="data",
        )

    def tail(self, n: int = 5) -> ResultView:
        """Return last n vectors + field metadata."""
        flat_data, ids_raw, fields = self._inner.tail(n)
        dim = self.dimension
        vectors = np.asarray(flat_data, dtype=np.float32).reshape(-1, dim)
        ids = np.asarray(ids_raw, dtype=np.int64)
        return ResultView(
            vectors=vectors, ids=ids, fields=list(fields),
            result_type="data",
        )

    def get_vectors(self, ids: List[int]) -> "np.ndarray":
        """Retrieve vectors by user IDs. Returns numpy array of shape (len(ids), dim)."""
        return self._inner.get_vectors([int(i) for i in ids])

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

    def delete_items(self, ids: List[int]) -> None:
        """Soft-delete vectors by user ID."""
        self._inner.delete_items([int(i) for i in ids])

    def restore_items(self, ids: List[int]) -> None:
        """Restore previously soft-deleted vectors."""
        self._inner.restore_items([int(i) for i in ids])

    def list_deleted_ids(self) -> List[int]:
        """Return sorted list of all currently soft-deleted user IDs."""
        return list(self._inner.list_deleted_ids())

    def search_range(self, vector: "np.ndarray", threshold: float, max_results: int = 1000) -> ResultView:
        """Return all vectors within distance threshold."""
        vec = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        ids_raw, dists_raw = self._inner.search_range(vec, float(threshold), int(max_results))
        idx_type, metric = _parse_index_mode(self._inner.index_mode())
        ids = np.array(ids_raw, dtype=np.int64)
        dists = np.array(dists_raw, dtype=np.float32)
        return ResultView(
            ids=ids, distances=dists,
            k=len(ids), distance=metric, index=idx_type,
            result_type="search",
        )

    def is_id_exists(self, user_id: int) -> bool:
        """Check whether a user ID exists in the collection."""
        return self._inner.is_id_exists(int(user_id))

    def max_id(self) -> int:
        """Return the maximum user ID stored, or -1 if empty."""
        return self._inner.max_id()

    def compact(self) -> int:
        """Physically remove tombstoned vectors. Returns count removed."""
        return self._inner.compact()

    def delete(self) -> None:
        self._inner.delete()

    def snapshot_to(self, snapshot_path: str) -> None:
        self._inner.snapshot_to(snapshot_path)

    def export_to(self, export_path: str) -> None:
        self._inner.export_to(export_path)
