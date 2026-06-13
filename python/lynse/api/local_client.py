"""
Local client for LynseDB — direct Rust backend access without HTTP.

Mirrors the API of HTTPClient and Collection from http_api/client_api.py,
but calls the Rust PyO3 bindings directly, eliminating all network I/O overhead.
"""

import queue
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional, Callable
from threading import Lock

import numpy as np
from tqdm import trange

from ..utils.utils import collection_repr
from .._backend import DatabaseManager, Collection, SearchResult, _normalize_sparse_vector
from ..result_view import ResultView, _parse_index_mode
from .rerank import apply_external_rerank, should_fetch_fields


DEFAULT_INSERT_BUFFER_SIZE = 10_000


class LocalClient:
    """
    Local database client — direct Rust backend, no HTTP.

    Drop-in replacement for HTTPClient when running in local mode.
    """

    def __init__(self, manager: DatabaseManager, database_name: str):
        self._manager = manager
        self.database_name = database_name

    @property
    def is_read_only(self) -> bool:
        return self._manager.is_read_only

    def require_collection(
            self,
            collection: str,
            dim: int = None,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            drop_if_exists: bool = False,
            description: str = None,
            dtypes: str = "float32",
    ):
        """
        Create or open a collection.

        Parameters:
            collection (str): The name of the collection.
            dim (int): The dimension of the vectors.
            n_threads (int): The number of threads. Default is 10.
            warm_up (bool): Whether to warm up. Default is False.
            drop_if_exists (bool): Whether to drop the collection if it exists. Default is False.
            description (str): A description of the collection. Default is None.
            dtypes (str): Dense vector storage dtype, "float32" or "float16".

        Returns:
            LocalCollection: The collection object.
        """
        self._manager.require_collection(
            self.database_name, collection, dim, drop_if_exists, description, dtypes
        )
        rust_coll = self._manager.get_collection(
            self.database_name, collection, dim
        )
        return LocalCollection(
            manager=self._manager,
            database_name=self.database_name,
            collection_name=collection,
            rust_collection=rust_coll,
            dim=dim,
            n_threads=n_threads,
            warm_up=warm_up,
            drop_if_exists=drop_if_exists,
            description=description,
            dtypes=dtypes,
        )

    def get_collection(self, collection: str, warm_up=True):
        """
        Get an existing collection.

        Parameters:
            collection (str): The name of the collection.
            warm_up (bool): Whether to warm up. Default is True.

        Returns:
            LocalCollection: The collection object.
        """
        if not self._manager.collection_exists(self.database_name, collection):
            raise RuntimeError(f"Collection '{collection}' does not exist.")

        config = self._manager.get_collection_config(self.database_name, collection)
        if config is None:
            raise RuntimeError(f"Collection config for '{collection}' not found.")

        dim = config['dim']
        dtypes = config.get('dtypes', 'float32')

        rust_coll = self._manager.get_collection(
            self.database_name, collection, dim
        )
        return LocalCollection(
            manager=self._manager,
            database_name=self.database_name,
            collection_name=collection,
            rust_collection=rust_coll,
            dim=dim,
            dtypes=dtypes,
            warm_up=warm_up,
        )

    def drop_collection(self, collection: str):
        """
        Drop a collection.

        Parameters:
            collection (str): The name of the collection.

        Returns:
            dict: Status message.
        """
        self._manager.drop_collection(self.database_name, collection)
        return {'status': 'success'}

    def snapshot_collection(self, collection: str, snapshot_path: Union[str, Path]):
        """Create a filesystem snapshot for a collection."""
        self._manager.snapshot_collection(self.database_name, collection, str(snapshot_path))
        return {'status': 'success'}

    def export_collection(self, collection: str, export_path: Union[str, Path]):
        """Export a collection as JSONL metadata plus binary vectors."""
        self._manager.export_collection(self.database_name, collection, str(export_path))
        return {'status': 'success'}

    def restore_collection(
            self,
            collection: str,
            snapshot_path: Union[str, Path],
            overwrite: bool = False,
    ):
        """Restore a collection from a filesystem snapshot."""
        self._manager.restore_collection(
            self.database_name, collection, str(snapshot_path), overwrite
        )
        return {'status': 'success'}

    def import_collection(
            self,
            collection: str,
            export_path: Union[str, Path],
            overwrite: bool = False,
    ):
        """Import a collection from JSONL metadata plus binary vectors."""
        self._manager.import_collection(
            self.database_name, collection, str(export_path), overwrite
        )
        return {'status': 'success'}

    def snapshot_database(self, snapshot_path: Union[str, Path]):
        """Create a filesystem snapshot for this database."""
        self._manager.snapshot_database(self.database_name, str(snapshot_path))
        return {'status': 'success'}

    def restore_database(
            self,
            snapshot_path: Union[str, Path],
            overwrite: bool = False,
    ):
        """Restore this database from a filesystem snapshot."""
        self._manager.restore_database(self.database_name, str(snapshot_path), overwrite)
        return {'status': 'success'}

    def drop_database(self):
        """
        Drop the database.

        Returns:
            dict: Status message.
        """
        if not self._manager.database_exists(self.database_name):
            return {'status': 'success', 'message': 'The database does not exist.'}
        self._manager.drop_database(self.database_name)
        return {'status': 'success'}

    def database_exists(self):
        """
        Check if the database exists.

        Returns:
            dict: Response with exists flag.
        """
        exists = self._manager.database_exists(self.database_name)
        return {'params': {'exists': exists}}

    def show_collections(self):
        """
        Show all collections in the database.

        Returns:
            List: The list of collections.
        """
        return self._manager.show_collections(self.database_name)

    def update_collection_description(self, collection: str, description: str):
        """
        Update the description of a collection.

        Parameters:
            collection (str): The name of the collection.
            description (str): The description of the collection.

        Returns:
            dict: Status message.
        """
        self._manager.update_collection_description(self.database_name, collection, description)
        return {'status': 'success'}

    def show_collections_details(self):
        """
        Show all collections in the database with details.

        Returns:
            list or pandas.DataFrame: The details of the collections.
        """
        collections = self._manager.show_collections(self.database_name)
        details = []
        for coll_name in collections:
            config = self._manager.get_collection_config(self.database_name, coll_name)
            if config:
                details.append({
                    'collection': coll_name,
                    'dim': config['dim'],
                    'dtypes': config.get('dtypes', 'float32'),
                    'description': config.get('description'),
                })
        try:
            import pandas as pd
            return pd.DataFrame(details)
        except ImportError:
            return details

    def __repr__(self):
        exists = self._manager.database_exists(self.database_name)
        return f"LocalDatabaseInstance(name={self.database_name}, exists={exists})"

    def __str__(self):
        return self.__repr__()


class LocalCollection:
    """
    Local collection — direct Rust backend access, no HTTP.

    Drop-in replacement for the HTTP Collection class.
    """
    name = "Local"

    def __init__(
            self,
            manager,
            database_name,
            collection_name,
            rust_collection,
            dim: Union[int, None] = None,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            drop_if_exists: bool = False,
            description: Union[str, None] = None,
            dtypes: str = "float32",
    ):
        self.IS_DELETED = False
        self._manager = manager
        self._database_name = database_name
        self._collection_name = collection_name
        self._rust_coll = rust_collection
        self._init_params = {
            'dim': dim,
            'n_threads': n_threads,
            'warm_up': warm_up,
            'drop_if_exists': drop_if_exists,
            'description': description,
            'dtypes': dtypes,
        }

        self.COMMIT_FLAG = False
        self._mesosphere_list = queue.Queue()
        self._lock = Lock()

    @property
    def is_read_only(self) -> bool:
        return self._rust_coll.is_read_only

    @property
    def vector_dtype(self) -> str:
        return self._rust_coll.vector_dtype

    def exists(self):
        """Check if the collection exists."""
        return self._manager.collection_exists(self._database_name, self._collection_name)

    def add_item(self, vector: Union[list, np.ndarray], id: int, *,
                 field: Union[dict, None] = None,
                 buffer_size: int = True):
        """
        Add an item to the collection.

        Parameters:
            vector (list or np.ndarray): The vector of the item.
            id (int): The ID of the item.
            field (dict, optional): The fields of the item.
            buffer_size (int or bool): The buffer size.
                If True, the default buffered batch size (10000) is used.
        """
        if buffer_size is True:
            buffer_size = DEFAULT_INSERT_BUFFER_SIZE
        else:
            if buffer_size is False:
                buffer_size = 0
            elif not isinstance(buffer_size, int) or buffer_size < 0:
                raise ValueError('If buffer_size is not bool, it must be a positive integer.')

        if buffer_size == 0:
            vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
            fields = [field] if field else None
            self._rust_coll.add_items(vec, [int(id)], fields)
            self.COMMIT_FLAG = False
            return id
        else:
            vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
            with self._lock:
                self._mesosphere_list.put({
                    "vector": vec,
                    "id": id,
                    "field": field if field is not None else {},
                })

            if self._mesosphere_list.qsize() >= buffer_size:
                self._flush_buffer()

            return id

    def _flush_buffer(self):
        """Flush the internal buffer to the Rust collection."""
        with self._lock:
            items = list(self._mesosphere_list.queue)
            if not items:
                return
            vectors = np.array([item['vector'] for item in items], dtype=np.float32)
            ids = [int(item['id']) for item in items]
            fields = [item.get('field', {}) for item in items]
            has_fields = any(f for f in fields)
            self._rust_coll.add_items(vectors, ids, fields if has_fields else None)
            self.COMMIT_FLAG = False
            self._mesosphere_list = queue.Queue()

    def bulk_add_items(
            self,
            vectors: List[Union[
                Tuple[Union[List, Tuple, np.ndarray], int, dict],
                Tuple[Union[List, Tuple, np.ndarray], int]
            ]],
            batch_size: int = 1000,
            enable_progress_bar: bool = True,
            wire_dtype: str = "float32",
    ):
        """
        Add multiple items to the collection.

        Parameters:
            vectors: List of tuples (vector, id, fields) or (vector, id).
            batch_size (int): The batch size. Default is 1000.
            enable_progress_bar (bool): Whether to enable the progress bar.
            wire_dtype (str): Accepted for HTTP API parity; ignored for local calls.

        Returns:
            list: The IDs of the items added.
        """
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        ids = []

        if enable_progress_bar:
            iter_obj = trange(total_batches, desc='Adding items', unit='batch')
        else:
            iter_obj = range(total_batches)

        for i in iter_obj:
            start = i * batch_size
            end = (i + 1) * batch_size
            items = vectors[start:end]

            batch_vecs = []
            batch_fields = []
            batch_ids = []
            has_fields = False
            for item in items:
                if not isinstance(item, tuple):
                    raise TypeError('Each item must be a tuple of vector, ID, and fields(optional).')
                if len(item) == 3:
                    v, vid, vf = item
                    batch_fields.append(vf)
                    has_fields = True
                elif len(item) == 2:
                    v, vid = item
                    batch_fields.append({})
                else:
                    raise TypeError('Each item must be a tuple of vector, ID, and fields(optional).')
                batch_vecs.append(v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32))
                batch_ids.append(vid)

            vec_array = np.array(batch_vecs, dtype=np.float32)
            int_ids = [int(vid) for vid in batch_ids]
            self._rust_coll.add_items(vec_array, int_ids, batch_fields if has_fields else None)
            self.COMMIT_FLAG = False
            ids.extend(batch_ids)

        return ids

    def bulk_add_binary(
            self,
            vectors: np.ndarray,
            batch_size: int = 50000,
            enable_progress_bar: bool = True,
            wire_dtype: str = "float32",
    ):
        """
        High-performance binary bulk add. Directly passes numpy arrays to Rust.

        Parameters:
            vectors (np.ndarray): 2D array of shape (n, dim), dtype float32.
            batch_size (int): Number of vectors per batch. Default is 50000.
            enable_progress_bar (bool): Whether to enable the progress bar.
            wire_dtype (str): Accepted for HTTP API parity; ignored for local calls.

        Returns:
            int: Total number of vectors added.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        n_total = vectors.shape[0]
        total_batches = (n_total + batch_size - 1) // batch_size
        added = 0

        if enable_progress_bar:
            iter_obj = trange(total_batches, desc='Adding vectors (binary)', unit='batch')
        else:
            iter_obj = range(total_batches)

        offset = self._rust_coll.max_id()
        start_id = int(offset) + 1 if offset >= 0 else 0

        for i in iter_obj:
            start = i * batch_size
            end = min((i + 1) * batch_size, n_total)
            batch = vectors[start:end]
            n_batch = batch.shape[0]
            seq_ids = list(range(start_id, start_id + n_batch))
            self._rust_coll.add_items(batch, seq_ids, None)
            start_id += n_batch
            self.COMMIT_FLAG = False
            added += n_batch

        return added

    def upsert_item(self, vector: Union[list, np.ndarray], id: int, *,
                    field: Union[dict, None] = None):
        """
        Insert or update a single item by ID.

        Existing IDs are updated in place. Missing IDs are inserted. If
        ``field`` is provided, it replaces the current fields for that row; if
        omitted, existing fields are preserved.
        """
        vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        fields = [field] if field is not None else None
        self._rust_coll.upsert_items([int(id)], vec, fields)
        self.COMMIT_FLAG = False
        return id

    def upsert_items(
            self,
            vectors: List[Union[
                Tuple[Union[List, Tuple, np.ndarray], int, dict],
                Tuple[Union[List, Tuple, np.ndarray], int]
            ]],
            batch_size: int = 1000,
            enable_progress_bar: bool = True,
            wire_dtype: str = "float32",
    ):
        """
        Insert or update multiple items by ID.

        Two-tuples ``(vector, id)`` update only the vector and preserve existing
        fields. Three-tuples ``(vector, id, field)`` replace fields for that ID.
        ``wire_dtype`` is accepted for HTTP API parity and ignored locally.
        """
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        ids = []

        if enable_progress_bar:
            iter_obj = trange(total_batches, desc='Upserting items', unit='batch')
        else:
            iter_obj = range(total_batches)

        for i in iter_obj:
            start = i * batch_size
            end = (i + 1) * batch_size
            items = vectors[start:end]

            existing_vecs_with_fields = []
            existing_ids_with_fields = []
            existing_fields_with_fields = []
            new_vecs_with_fields = []
            new_ids_with_fields = []
            new_fields_with_fields = []
            vecs_without_fields = []
            ids_without_fields = []
            seen_ids = set()

            for item in items:
                if not isinstance(item, tuple):
                    raise TypeError('Each item must be a tuple of vector, ID, and fields(optional).')
                if len(item) == 3:
                    v, vid, vf = item
                    if int(vid) in seen_ids:
                        raise ValueError(f'duplicate id {vid} within the same upsert batch')
                    int_vid = int(vid)
                    seen_ids.add(int_vid)
                    vec = v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)
                    if self._rust_coll.is_id_exists(int_vid):
                        existing_vecs_with_fields.append(vec)
                        existing_ids_with_fields.append(int_vid)
                        existing_fields_with_fields.append(vf)
                    else:
                        new_vecs_with_fields.append(vec)
                        new_ids_with_fields.append(int_vid)
                        new_fields_with_fields.append(vf)
                    ids.append(vid)
                elif len(item) == 2:
                    v, vid = item
                    if int(vid) in seen_ids:
                        raise ValueError(f'duplicate id {vid} within the same upsert batch')
                    seen_ids.add(int(vid))
                    vecs_without_fields.append(v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32))
                    ids_without_fields.append(int(vid))
                    ids.append(vid)
                else:
                    raise TypeError('Each item must be a tuple of vector, ID, and fields(optional).')

            if vecs_without_fields:
                vec_array = np.array(vecs_without_fields, dtype=np.float32)
                self._rust_coll.upsert_items(ids_without_fields, vec_array, None)
            if existing_vecs_with_fields:
                vec_array = np.array(existing_vecs_with_fields, dtype=np.float32)
                self._rust_coll.upsert_items(
                    existing_ids_with_fields,
                    vec_array,
                    existing_fields_with_fields,
                )
            if new_vecs_with_fields:
                vec_array = np.array(new_vecs_with_fields, dtype=np.float32)
                self._rust_coll.add_items(
                    vec_array,
                    new_ids_with_fields,
                    new_fields_with_fields,
                )
            self.COMMIT_FLAG = False

        return ids

    def commit(self):
        """Commit the changes in the collection."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        self._rust_coll.commit()
        self.COMMIT_FLAG = True
        return {'status': 'success'}

    def flush(self):
        """Flush pending bytes and fsync collection files without clearing WAL."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        self._rust_coll.flush()
        return {'status': 'success'}

    def checkpoint(self):
        """Checkpoint durable state and clear WAL."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        self._rust_coll.checkpoint()
        self.COMMIT_FLAG = True
        return {'status': 'success'}

    def close(self):
        """Flush and close the collection handle from an API perspective."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        self._rust_coll.close()
        self.COMMIT_FLAG = True
        return {'status': 'success'}

    def snapshot_to(self, snapshot_path: Union[str, Path]):
        """Create a filesystem snapshot of this collection."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        self._rust_coll.snapshot_to(str(snapshot_path))
        return {'status': 'success'}

    def export_to(self, export_path: Union[str, Path]):
        """Export this collection as JSONL metadata plus binary vectors."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        self._rust_coll.export_to(str(export_path))
        return {'status': 'success'}

    def is_id_exists(self, id: int) -> bool:
        """Check if a user ID exists in the collection."""
        return self._rust_coll.is_id_exists(int(id))

    @property
    def max_id(self) -> int:
        """Return the maximum user ID stored in the collection, or -1 if empty."""
        return self._rust_coll.max_id()

    def compact(self) -> int:
        """Physically remove all tombstoned vectors and rebuild storage.

        Returns:
            int: Number of vectors physically removed.
        """
        return self._rust_coll.compact()

    def stats(self) -> dict:
        """Return collection statistics.

        Returns:
            dict: n_vectors, n_live, n_tombstoned, dimension, index_mode, max_id.
        """
        n_vectors, dimension = self._rust_coll.shape
        deleted = self._rust_coll.list_deleted_ids()
        n_tombstoned = len(deleted)
        n_live = max(0, n_vectors - n_tombstoned)
        return {
            'n_vectors': n_vectors,
            'n_live': n_live,
            'n_tombstoned': n_tombstoned,
            'dimension': dimension,
            'index_mode': self._rust_coll.index_mode or 'none',
            'max_id': self._rust_coll.max_id(),
        }

    def build_index(
            self,
            index_mode: str = 'FLAT',
            field_name: str = 'default',
            n_clusters: Union[int, None] = None,
    ):
        """
        Build the index for the collection.

        Parameters:
            index_mode (str): The index mode, must be one of the following:

                **Flat (brute-force):**

                - 'FLAT': Flat index with inner product. (Default)
                - 'FLAT-L2': Flat index with squared L2 distance.
                - 'FLAT-COS': Flat index with cosine similarity.
                - 'FLAT-IP-SQ8': Flat index with inner product and SQ8 quantizer.
                - 'FLAT-L2-SQ8': Flat index with squared L2 distance and SQ8 quantizer.
                - 'FLAT-COS-SQ8': Flat index with cosine similarity and SQ8 quantizer.
                - 'FLAT-JACCARD-BINARY': Flat index with Jaccard distance (binary vectors).
                - 'FLAT-HAMMING-BINARY': Flat index with Hamming distance (binary vectors).

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

                **Flat + PolarVec (training-free multi-bit quantization, 4-8x compression):**

                - 'FLAT-IP-POLARVEC': PolarVec with inner product (auto bits, default 4).
                - 'FLAT-L2-POLARVEC': PolarVec with squared L2 distance.
                - 'FLAT-COS-POLARVEC': PolarVec with cosine similarity.
                - 'FLAT-IP-POLARVEC3': PolarVec with inner product and 3-bit codes (~10.7x).
                - 'FLAT-IP-POLARVEC4': PolarVec with inner product and 4-bit codes (~8x).
                - 'FLAT-IP-POLARVEC8': PolarVec with inner product and 8-bit codes (~4x).

                **HNSW (graph-based ANN):**

                - 'HNSW': HNSW index with inner product.
                - 'HNSW-L2': HNSW index with squared L2 distance.
                - 'HNSW-Cos': HNSW index with cosine similarity.
                - 'HNSW-IP-SQ8': HNSW index with inner product and SQ8 quantizer.
                - 'HNSW-L2-SQ8': HNSW index with squared L2 distance and SQ8 quantizer.
                - 'HNSW-Cos-SQ8': HNSW index with cosine similarity and SQ8 quantizer.

                **DiskANN (disk-friendly graph ANN):**

                - 'DiskANN': DiskANN index with inner product.
                - 'DiskANN-L2': DiskANN index with squared L2 distance.
                - 'DiskANN-Cos': DiskANN index with cosine similarity.
                - 'DiskANN-IP-SQ8': DiskANN index with inner product and SQ8 quantizer.
                - 'DiskANN-L2-SQ8': DiskANN index with squared L2 distance and SQ8 quantizer.
                - 'DiskANN-Cos-SQ8': DiskANN index with cosine similarity and SQ8 quantizer.

                **IVF (inverted file ANN):**

                - 'IVF': IVF index with inner product.
                - 'IVF-L2': IVF index with squared L2 distance.
                - 'IVF-COS': IVF index with cosine similarity.
                - 'IVF-IP-SQ8': IVF index with inner product and SQ8 quantizer.
                - 'IVF-L2-SQ8': IVF index with squared L2 distance and SQ8 quantizer.
                - 'IVF-COS-SQ8': IVF index with cosine similarity and SQ8 quantizer.
                - 'IVF-JACCARD-BINARY': IVF index with Jaccard distance (binary vectors).
                - 'IVF-HAMMING-BINARY': IVF index with Hamming distance (binary vectors).
            field_name (str): Named vector field to build index for.
                Defaults to "default" (the primary collection vector).
            n_clusters (int, optional): The number of clusters. Only IVF modes
                use it; other index modes silently ignore it.

        Returns:
            dict: Status message.
        """
        effective_n_clusters = n_clusters if index_mode.upper().startswith("IVF") else None
        self._rust_coll.build_index(
            index_mode,
            field_name=field_name,
            n_clusters=effective_n_clusters,
        )
        return {'status': 'success'}

    def remove_index(self, field_name: str = 'default'):
        """Remove the index of the collection.

        Parameters:
            field_name (str): Named vector field to remove index for.
                Defaults to "default" (the primary collection index).
        """
        self._rust_coll.remove_index(field_name=field_name)
        return {'status': 'success'}

    def create_vector_field(
            self,
            name: str,
            dim: int,
            metric: str = "ip",
            index_mode: Union[str, None] = None,
            dtypes: Union[str, None] = None,
    ):
        """Create a named vector field with its own dimension and metric."""
        self._rust_coll.create_vector_field(name, int(dim), metric, index_mode, dtypes)
        return {'status': 'success'}

    def list_vector_fields(self):
        """List vector fields, including the reserved default primary vector."""
        return self._rust_coll.list_vector_fields()

    def add_named_vectors(
            self,
            field_name: str,
            vectors: Union[list, np.ndarray],
            ids: List[int],
    ):
        """Attach vectors to a named vector field for existing IDs."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self._rust_coll.add_named_vectors(field_name, vecs, [int(i) for i in ids])
        self.COMMIT_FLAG = False
        return {'status': 'success'}

    def add_sparse_vectors(
            self,
            vectors: List[Union[Dict[int, float], List[Tuple[int, float]]]],
            ids: List[int],
    ):
        """Attach sparse feature vectors to existing IDs."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        normalized = [_normalize_sparse_vector(vector) for vector in vectors]
        self._rust_coll.add_sparse_vectors(normalized, [int(i) for i in ids])
        self.COMMIT_FLAG = False
        return {'status': 'success'}

    def insert_session(self):
        """Start an insert session."""
        from ..execution_layer.session import DataInsertionSession
        return DataInsertionSession(self)

    def search(
            self, vector: Union[list, np.ndarray], k: int = 10, *,
            where: Union[str, None] = None,
            return_fields: bool = False,
            vector_field: str = "default",
            reranker: Optional[Callable[[Dict[str, Any]], Any]] = None,
            rerank_k: Optional[int] = None,
            rerank_with_fields: bool = False,
            nprobe: int = 10,
            approx: bool = False,
            eps: float = 1e-4,
            wire_dtype: str = "float32",
    ):
        """
        Search the collection for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray or list): The search vector.
            k (int): The number of nearest vectors to return.
            where (str, optional): SQL/WHERE expression string to filter results.
            return_fields (bool): Whether to return the fields of the search results.
            vector_field (str): Named vector field to search. ``default`` searches
                the primary collection vector.
            reranker (callable, optional): External rerank hook. It receives
                ``{"query": ..., "items": [...]}`` and can return IDs and/or scores.
            rerank_k (int, optional): Keep top-N after rerank. Defaults to the
                backend result size.
            rerank_with_fields (bool): Fetch candidate fields for reranker payload
                even when ``return_fields=False``.
            nprobe (int): Controls search breadth by index type (default: 10).
                - **IVF**: number of partitions to probe; higher improves recall and increases latency.
                - **HNSW**: ef_search beam width; higher improves recall and increases latency.
                - **Flat / PQ / RaBitQ / PolarVec**: ignored.
                - Named vector fields: ignored.
            approx (bool): Metric-specific flat approximation for IP, L2,
                and Cosine. Ignored for Hamming/Jaccard.
            eps (float): Distance rounding tolerance when ``approx=True``
                for supported metrics (default 1e-4). Ignored when
                ``approx=False`` or the metric does not support approximation.
            wire_dtype (str): Accepted for HTTP API parity; ignored for local calls.

        Returns:
            ResultView: Search results with ids, distances, and optional fields.
        """
        eps = float(eps)
        vec = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        result = self._rust_coll.search(
            vec,
            k=k,
            where=where,
            field_name=vector_field,
            nprobe=nprobe,
            approx=approx,
            eps=eps,
        )
        need_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        fields: List[Dict[str, Any]] = []
        if need_fields and len(result) > 0:
            fields = self._rust_coll.retrieve_fields(result.ids.tolist())
        ids, distances, reranked_fields = apply_external_rerank(
            ids=result.ids,
            scores=result.distances,
            fields=fields,
            reranker=reranker,
            query={
                "type": "vector_search",
                "vector_field": vector_field,
                "vector": vec.tolist(),
                "where": where,
                "nprobe": nprobe,
                "approx": approx,
                "eps": eps,
            },
            rerank_k=rerank_k,
        )
        return ResultView(
            ids=ids,
            distances=distances,
            fields=reranked_fields if return_fields else [],
            k=len(ids),
            distance=result.distance_metric,
            index=result.index_type,
            result_type="search",
        )

    def search_sparse(
            self, vector: Union[Dict[int, float], List[Tuple[int, float]]],
            k: int = 10, *, where: Union[str, None] = None,
            return_fields: bool = False,
            reranker: Optional[Callable[[Dict[str, Any]], Any]] = None,
            rerank_k: Optional[int] = None,
            rerank_with_fields: bool = True,
    ):
        """Sparse vector search using inner product."""
        sparse_vector = _normalize_sparse_vector(vector)
        result = self._rust_coll.search_sparse(sparse_vector, k=k, where=where)
        need_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        fields: List[Dict[str, Any]] = []
        if need_fields and len(result) > 0:
            fields = self._rust_coll.retrieve_fields(result.ids.tolist())
        ids, distances, reranked_fields = apply_external_rerank(
            ids=result.ids,
            scores=result.distances,
            fields=fields,
            reranker=reranker,
            query={
                "type": "sparse_search",
                "vector": sparse_vector,
                "where": where,
            },
            rerank_k=rerank_k,
        )
        return ResultView(
            ids=ids,
            distances=distances,
            fields=reranked_fields if return_fields else [],
            k=len(ids),
            distance=result.distance_metric,
            index=result.index_type,
            result_type="search",
        )

    def search_profile(
            self, vector: Union[list, np.ndarray], k: int = 10, *,
            where: Union[str, None] = None, nprobe: int = 10
    ):
        """Search and return profile/explain metadata."""
        return self._rust_coll.search_profile(
            np.ascontiguousarray(vector, dtype=np.float32).ravel(),
            k=k,
            where=where,
            nprobe=nprobe,
        )

    def text_search(
            self, text: str, k: int = 10, *,
            text_fields: Union[list[str], None] = None,
            where: Union[str, None] = None,
            return_fields: bool = False,
            reranker: Optional[Callable[[Dict[str, Any]], Any]] = None,
            rerank_k: Optional[int] = None,
            rerank_with_fields: bool = True,
    ):
        """BM25 text search over metadata fields."""
        result = self._rust_coll.text_search(text, text_fields=text_fields, k=k, where=where)
        need_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        fields: List[Dict[str, Any]] = []
        if need_fields and len(result) > 0:
            fields = self._rust_coll.retrieve_fields(result.ids.tolist())
        ids, distances, reranked_fields = apply_external_rerank(
            ids=result.ids,
            scores=result.distances,
            fields=fields,
            reranker=reranker,
            query={
                "type": "text_search",
                "text": text,
                "text_fields": text_fields,
                "where": where,
            },
            rerank_k=rerank_k,
        )
        return ResultView(
            ids=ids,
            distances=distances,
            fields=reranked_fields if return_fields else [],
            k=len(ids),
            distance=result.distance_metric,
            index=result.index_type,
            result_type="search",
        )

    def hybrid_search(
            self, vector: Union[list, np.ndarray, None] = None, text: Union[str, None] = None,
            k: int = 10, *, where: Union[str, None] = None,
            text_fields: Union[list[str], None] = None, fusion: str = "rrf",
            vector_weight: float = 1.0, text_weight: float = 1.0,
            rrf_k: float = 60.0, candidate_limit: Union[int, None] = None,
            nprobe: int = 10, return_fields: bool = False,
            reranker: Optional[Callable[[Dict[str, Any]], Any]] = None,
            rerank_k: Optional[int] = None,
            rerank_with_fields: bool = True,
    ):
        """Hybrid vector + BM25 text search with RRF or weighted fusion."""
        vec = None if vector is None else np.ascontiguousarray(vector, dtype=np.float32).ravel()
        result = self._rust_coll.hybrid_search(
            vector=vec, text=text, k=k, where=where, text_fields=text_fields,
            fusion=fusion, vector_weight=vector_weight, text_weight=text_weight,
            rrf_k=rrf_k, candidate_limit=candidate_limit, nprobe=nprobe,
        )
        need_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        fields: List[Dict[str, Any]] = []
        if need_fields and len(result) > 0:
            fields = self._rust_coll.retrieve_fields(result.ids.tolist())
        ids, distances, reranked_fields = apply_external_rerank(
            ids=result.ids,
            scores=result.distances,
            fields=fields,
            reranker=reranker,
            query={
                "type": "hybrid_search",
                "vector": None if vec is None else vec.tolist(),
                "text": text,
                "text_fields": text_fields,
                "where": where,
                "fusion": fusion,
                "vector_weight": float(vector_weight),
                "text_weight": float(text_weight),
                "rrf_k": float(rrf_k),
                "candidate_limit": candidate_limit,
                "nprobe": nprobe,
            },
            rerank_k=rerank_k,
        )
        return ResultView(
            ids=ids,
            distances=distances,
            fields=reranked_fields if return_fields else [],
            k=len(ids),
            distance=result.distance_metric,
            index=result.index_type,
            result_type="search",
        )

    def batch_search(
            self, vectors: Union[list, np.ndarray], k: int = 10, *,
            where: Union[str, None] = None,
            return_fields: bool = False, nprobe: int = 10,
            reranker: Optional[Callable[[Dict[str, Any]], Any]] = None,
            rerank_k: Optional[int] = None,
            rerank_with_fields: bool = False,
            wire_dtype: str = "float32",
    ):
        """
        Batch search: search multiple query vectors.

        Parameters:
            vectors (np.ndarray or list): Multiple query vectors, shape (n, dim).
            k (int): The number of nearest vectors to return per query.
            where (str, optional): SQL/WHERE expression string to filter results.
            return_fields (bool): Whether to return the fields.
            nprobe (int): Controls search breadth by index type (default: 10).
                - **IVF**: number of partitions to probe — higher = better recall, slower.
                - **HNSW**: ef_search beam width — higher = better recall, slower.
                - **Flat / PQ / RaBitQ**: ignored (exhaustive two-pass search).
            reranker (callable, optional): External rerank hook, applied per-query.
            rerank_k (int, optional): Keep top-N after rerank per query.
            rerank_with_fields (bool): Fetch candidate fields for reranker payload
                even when ``return_fields=False``.
            wire_dtype (str): Accepted for HTTP API parity; ignored for local calls.

        Returns:
            List[ResultView]: List of ResultView objects, one per query vector.
        """
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        results = self._rust_coll.batch_search(vecs, k=k, where=where, nprobe=nprobe)
        output = []
        need_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        for idx, r in enumerate(results):
            fields: List[Dict[str, Any]] = []
            if need_fields and len(r) > 0:
                fields = self._rust_coll.retrieve_fields(r.ids.tolist())
            ids, distances, reranked_fields = apply_external_rerank(
                ids=r.ids,
                scores=r.distances,
                fields=fields,
                reranker=reranker,
                query={
                    "type": "batch_vector_search",
                    "vector": vecs[idx].tolist(),
                    "where": where,
                    "nprobe": nprobe,
                    "query_index": idx,
                },
                rerank_k=rerank_k,
            )
            output.append(ResultView(
                ids=ids,
                distances=distances,
                fields=reranked_fields if return_fields else [],
                k=len(ids),
                distance=r.distance_metric,
                index=r.index_type,
                result_type="search",
            ))
        return output

    @property
    def shape(self):
        """Get the shape of the collection."""
        return self._rust_coll.shape

    def head(self, n: int = 5):
        """Get the first n items in the collection.

        Returns:
            ResultView: Data result with vectors, ids, and fields.
        """
        return self._rust_coll.head(n)

    def tail(self, n: int = 5):
        """Get the last n items in the collection.

        Returns:
            ResultView: Data result with vectors, ids, and fields.
        """
        return self._rust_coll.tail(n)

    def query(self, where=None, filter_ids=None, return_ids_only=False):
        """
        Query the collection.

        Parameters:
            where (str or None): SQL/WHERE expression string.
            filter_ids (list[int]): The list of IDs to filter.
            return_ids_only (bool): Whether to return the IDs only.

        Returns:
            ResultView: Query result with ids and optional fields.
        """
        if where is not None:
            ids = self._rust_coll.query_fields(where)
            ids_arr = np.array(ids, dtype=np.int64) if ids else np.array([], dtype=np.int64)
            if return_ids_only:
                return ResultView(ids=ids_arr, result_type="query")
            if ids:
                fields = [dict(f) for f in self._rust_coll.retrieve_fields([int(i) for i in ids])]
            else:
                fields = []
            return ResultView(ids=ids_arr, fields=fields, result_type="query")
        elif filter_ids is not None:
            ids = filter_ids
        else:
            ids = []

        ids_arr = np.array(ids, dtype=np.int64) if ids else np.array([], dtype=np.int64)

        if return_ids_only:
            return ResultView(ids=ids_arr, result_type="query")

        if ids:
            fields = [dict(f) for f in self._rust_coll.retrieve_fields([int(i) for i in ids])]
        else:
            fields = []

        return ResultView(ids=ids_arr, fields=fields, result_type="query")

    def query_vectors(self, where=None, filter_ids=None):
        """
        Query vectors by field filter or ID list.

        Parameters:
            where (str or None): SQL/WHERE expression string to filter fields.
            filter_ids (list[int] or None): List of external IDs to retrieve.

        Returns:
            ResultView: Data result with vectors, ids, and fields.
        """
        if where is not None:
            ids, fields = self._rust_coll.query_with_fields(where)
            ids_arr = np.array(ids, dtype=np.int64) if ids else np.array([], dtype=np.int64)
        elif filter_ids is not None:
            ids = [int(i) for i in filter_ids]
            ids_arr = np.array(ids, dtype=np.int64)
            fields = [dict(f) for f in self._rust_coll.retrieve_fields(ids)] if ids else []
        else:
            ids = []
            ids_arr = np.array([], dtype=np.int64)
            fields = []

        if len(ids_arr) == 0:
            return ResultView(
                vectors=np.empty((0, self._rust_coll.dimension), dtype=np.float32),
                ids=ids_arr, fields=[], result_type="data",
            )

        vecs = self._rust_coll.get_vectors(ids_arr.tolist())
        return ResultView(vectors=vecs, ids=ids_arr, fields=list(fields), result_type="data")

    def delete_items(self, ids):
        """
        Soft-delete vectors by ID.

        Deleted IDs are excluded from all future search results. The raw vector
        data is NOT physically removed from disk; it is only tombstoned in memory
        and persisted as ``tombstone.bin``.

        Parameters:
            ids (list[int]): External IDs to soft-delete.
        """
        self._rust_coll.delete_items([int(i) for i in ids])

    def restore_items(self, ids):
        """
        Restore previously soft-deleted vectors.

        Parameters:
            ids (list[int]): External IDs to restore.
        """
        self._rust_coll.restore_items([int(i) for i in ids])

    def list_deleted_ids(self):
        """
        Return the sorted list of all currently soft-deleted IDs.

        Returns:
            list[int]: Sorted list of soft-deleted external IDs.
        """
        return list(self._rust_coll.list_deleted_ids())

    def search_range(self, vector, threshold, max_results=1000):
        """
        Range search: return all non-deleted vectors within a distance threshold.

        For L2 metric: returns IDs where distance <= threshold.
        For IP / Cosine: returns IDs where score >= threshold.

        Parameters:
            vector (np.ndarray): Query vector of shape (dim,).
            threshold (float): Distance / score threshold.
            max_results (int): Maximum number of results to return (default 1000).

        Returns:
            ResultView: Search results with ids and distances.
        """
        import numpy as np
        vec = np.asarray(vector, dtype=np.float32)
        return self._rust_coll.search_range(vec, float(threshold), int(max_results))

    def list_fields(self):
        """List all fields of a collection."""
        return self._rust_coll.list_fields()

    def update_description(self, description: str):
        """Update the description of the collection."""
        self._manager.update_collection_description(
            self._database_name, self._collection_name, description
        )
        return {'status': 'success'}

    @property
    def index_mode(self):
        """Get the index mode of the collection."""
        return self._rust_coll.index_mode

    def __repr__(self):
        return collection_repr(self)

    def __str__(self):
        return self.__repr__()
