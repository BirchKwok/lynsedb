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

from ..utils.utils import collection_repr
from .._backend import DatabaseManager, Collection, _normalize_sparse_vector
from ..result_view import ResultView, _parse_index_mode
from .rerank import apply_external_rerank, should_fetch_fields
from ._embedding import embed_documents
from ._records import (
    attach_documents,
    id_array,
    normalize_documents,
    normalize_external_ids,
    normalize_fields,
    normalize_vectors,
    validate_unique_external_ids,
)


DEFAULT_INSERT_BUFFER_SIZE = 10_000
DEFAULT_COLLECTION_INDEX = "FLAT-IP"


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
            default_index: Union[str, None] = DEFAULT_COLLECTION_INDEX,
    ):
        """
        Create or open a collection.

        Parameters:
            collection (str): The name of the collection.
            dim (int): Optional vector dimension. If omitted for a new
                collection, LynseDB infers it from the first inserted vectors.
            n_threads (int): The number of threads. Default is 10.
            warm_up (bool): Whether to warm up. Default is False.
            drop_if_exists (bool): Whether to drop the collection if it exists. Default is False.
            description (str): A description of the collection. Default is None.
            dtypes (str): Dense vector storage dtype, "float32" or "float16".
            default_index (str or None): Index mode to build automatically after
                the first write to a newly created collection. Use None to
                disable automatic index creation.

        Returns:
            LocalCollection: The collection object.
        """
        existed_before = (
            self._manager.collection_exists(self.database_name, collection)
            if not drop_if_exists
            else False
        )
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
            default_index=default_index if not existed_before else None,
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
            default_index: Union[str, None] = None,
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
        self._default_index = default_index
        self._default_index_built = False

        self.COMMIT_FLAG = False
        self._mesosphere_list = queue.Queue()
        self._lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and not self.COMMIT_FLAG:
            self.commit()
        return False

    @property
    def is_read_only(self) -> bool:
        return self._rust_coll.is_read_only

    def _refresh_dim_from_backend(self):
        dim = getattr(self._rust_coll, "dimension", None)
        if dim is None:
            shape = getattr(self._rust_coll, "shape", None)
            if shape is not None and len(shape) >= 2:
                dim = shape[1]
        if dim:
            self._init_params['dim'] = dim
        return dim

    def _maybe_build_default_index(self):
        if not self._default_index or self._default_index_built:
            return
        current_index = getattr(self._rust_coll, "index_mode", None)
        if current_index and str(current_index).lower() != "none":
            self._default_index_built = True
            return
        n_vectors, dim = self.shape
        if n_vectors <= 0 or dim <= 0:
            return
        self.build_index(self._default_index)
        self._default_index_built = True

    @property
    def vector_dtype(self) -> str:
        return self._rust_coll.vector_dtype

    def exists(self):
        """Check if the collection exists."""
        return self._manager.collection_exists(self._database_name, self._collection_name)

    def add(
            self,
            ids=None,
            *,
            vectors=None,
            documents=None,
            fields=None,
            batch_size: int = 1000,
            wire_dtype: str = "float32",
    ):
        """
        Add one or more records.

        ``ids`` are public string/int IDs. When omitted, LynseDB assigns
        sequential integer IDs starting after the current max ID.
        Provide ``vectors`` for direct vector insert, or provide ``documents``
        without vectors to trigger lazy local embedding.
        """
        del wire_dtype  # Local calls pass numpy arrays directly.
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if ids is None:
            docs, _ = normalize_documents(documents) if documents is not None else (None, False)
            if vectors is None:
                if docs is None:
                    raise ValueError("add() requires vectors or documents")
                vec_array = embed_documents(docs)
                n_records = vec_array.shape[0]
                if n_records != len(docs):
                    raise ValueError("embedding output count must match documents length")
            else:
                vec_array = np.asarray(vectors, dtype=np.float32)
                if vec_array.ndim == 1:
                    vec_array = vec_array.reshape(1, -1)
                elif vec_array.ndim != 2:
                    raise ValueError("vectors must be a 1D vector or a 2D matrix")
                vec_array = np.ascontiguousarray(vec_array, dtype=np.float32)
                n_records = vec_array.shape[0]
                if docs is not None and len(docs) != n_records:
                    raise ValueError(f"documents length ({len(docs)}) must match vectors row count ({n_records})")

            stored_fields = (
                None
                if fields is None and docs is None
                else attach_documents(normalize_fields(fields, n_records), docs)
            )

            with self._lock:
                offset = self._rust_coll.max_id()
                start_id = int(offset) + 1 if offset >= 0 else 0
                generated_ids = list(range(start_id, start_id + n_records))
                for start in range(0, n_records, batch_size):
                    end = min(start + batch_size, n_records)
                    self._rust_coll.add_items(
                        vec_array[start:end],
                        generated_ids[start:end],
                        None if stored_fields is None else stored_fields[start:end],
                    )
                self._refresh_dim_from_backend()

            self._maybe_build_default_index()
            self.COMMIT_FLAG = False
            return generated_ids[0] if n_records == 1 else generated_ids

        external_ids, single_id = normalize_external_ids(ids)
        n_records = len(external_ids)
        validate_unique_external_ids(external_ids)
        docs, _ = normalize_documents(documents, n_records) if documents is not None else (None, False)

        if vectors is None:
            if docs is None:
                raise ValueError("add() requires vectors or documents")
            vec_array = embed_documents(docs)
            if vec_array.shape[0] != n_records:
                raise ValueError("embedding output count must match ids length")
        else:
            vec_array = normalize_vectors(vectors, n_records)

        field_list = normalize_fields(fields, n_records)
        stored_fields = attach_documents(field_list, docs)

        with self._lock:
            added_ids = []
            for start in range(0, n_records, batch_size):
                end = min(start + batch_size, n_records)
                added_ids.extend(self._rust_coll.add_records(
                    vec_array[start:end],
                    external_ids[start:end],
                    stored_fields[start:end],
                ))
            self._refresh_dim_from_backend()

        self._maybe_build_default_index()
        self.COMMIT_FLAG = False
        return added_ids[0] if single_id else added_ids

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
            self._refresh_dim_from_backend()
            self.COMMIT_FLAG = False
            self._mesosphere_list = queue.Queue()
        self._maybe_build_default_index()

    def _result_ids_and_fields(self, internal_ids, fetch_fields: bool = True):
        internal_list = [int(i) for i in internal_ids.tolist()] if isinstance(internal_ids, np.ndarray) else [int(i) for i in internal_ids]
        external_ids = id_array(self._rust_coll.external_ids(internal_list)) if internal_list else id_array([])
        fields = [dict(f) for f in self._rust_coll.retrieve_fields(internal_list)] if fetch_fields and internal_list else []
        return external_ids, fields

    def _add_items_encoded_f16(self, vectors, ids, fields=None):
        add_encoded = getattr(self._rust_coll, "add_items_encoded_f16", None)
        if add_encoded is None:
            raise AttributeError("Rust backend does not expose add_items_encoded_f16")
        add_encoded(vectors, ids, fields)
        self._refresh_dim_from_backend()
        self.COMMIT_FLAG = False

    def _internal_ids(self, ids, *, missing: str = "error") -> list[int]:
        if isinstance(ids, (list, tuple, set, np.ndarray)) and len(ids) == 0:
            return []
        external_ids, _ = normalize_external_ids(ids)
        if missing == "ignore":
            external_ids = [
                external_id
                for external_id in external_ids
                if self._rust_coll.is_external_id_exists(external_id)
            ]
            if not external_ids:
                return []
        return self._rust_coll.internal_ids(external_ids)

    def upsert(
            self,
            ids,
            *,
            vectors,
            fields=None,
            batch_size: int = 1000,
            wire_dtype: str = "float32",
    ):
        """
        Insert or update one or more records by public ID.

        Existing IDs are updated in place. Missing IDs are inserted with newly
        allocated internal IDs. If ``fields`` is omitted, existing fields are
        preserved for updated rows and empty fields are used for new rows.
        """
        del wire_dtype  # Local calls pass numpy arrays directly.
        external_ids, single_id = normalize_external_ids(ids)
        n_records = len(external_ids)
        validate_unique_external_ids(external_ids)
        vec_array = normalize_vectors(vectors, n_records)
        field_list = normalize_fields(fields, n_records) if fields is not None else None
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        with self._lock:
            for start in range(0, n_records, batch_size):
                end = min(start + batch_size, n_records)
                batch_ids = external_ids[start:end]
                batch_vectors = vec_array[start:end]
                batch_fields = field_list[start:end] if field_list is not None else None

                existing_ids = []
                existing_positions = []
                new_ids = []
                new_positions = []
                for offset, public_id in enumerate(batch_ids):
                    if self._rust_coll.is_external_id_exists(public_id):
                        existing_ids.append(self._rust_coll.internal_ids([public_id])[0])
                        existing_positions.append(offset)
                    else:
                        new_ids.append(public_id)
                        new_positions.append(offset)

                if existing_ids:
                    self._rust_coll.upsert_items(
                        existing_ids,
                        batch_vectors[existing_positions],
                        [batch_fields[i] for i in existing_positions] if batch_fields is not None else None,
                    )
                if new_ids:
                    self._rust_coll.add_records(
                        batch_vectors[new_positions],
                        new_ids,
                        [batch_fields[i] for i in new_positions] if batch_fields is not None else None,
                    )
            self._refresh_dim_from_backend()

        self._maybe_build_default_index()
        self.COMMIT_FLAG = False
        return external_ids[0] if single_id else external_ids

    def commit(self):
        """Commit changes and clear WAL without forcing recursive fsync."""
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

    def is_id_exists(self, id: Union[str, int]) -> bool:
        """Check if a public ID exists in the collection."""
        return self._rust_coll.is_external_id_exists(id)

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
            index_mode: str = 'FLAT-IP',
            field_name: str = 'default',
            n_clusters: Union[int, None] = None,
    ):
        """
        Build the index for the collection.

        Parameters:
            index_mode (str): The index mode, must be one of the following:

                **Flat (brute-force):**

                - 'FLAT-IP': Flat index with inner product. (Default)
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
                - 'SPANN-COS': SPANN index with cosine similarity.
                - 'SPANN-IP-SQ8': SPANN index with inner product and SQ8 quantizer.
                - 'SPANN-L2-SQ8': SPANN index with squared L2 distance and SQ8 quantizer.
                - 'SPANN-COS-SQ8': SPANN index with cosine similarity and SQ8 quantizer.

                **IVF (inverted file ANN):**

                - 'IVF-IP': IVF index with inner product.
                - 'IVF-L2': IVF index with squared L2 distance.
                - 'IVF-COS': IVF index with cosine similarity.
                - 'IVF-IP-SQ8': IVF index with inner product and SQ8 quantizer.
                - 'IVF-L2-SQ8': IVF index with squared L2 distance and SQ8 quantizer.
                - 'IVF-COS-SQ8': IVF index with cosine similarity and SQ8 quantizer.
                - 'IVF-JACCARD-BINARY': IVF index with Jaccard distance (binary vectors).
                - 'IVF-HAMMING-BINARY': IVF index with Hamming distance (binary vectors).
            field_name (str): Named vector field to build index for.
                Defaults to "default" (the primary collection vector).
            n_clusters (int, optional): The number of clusters. IVF and SPANN
                modes use it; other index modes silently ignore it.

        Returns:
            dict: Status message.
        """
        effective_n_clusters = (
            n_clusters
            if index_mode.upper().startswith(("IVF", "SPANN"))
            else None
        )
        self._rust_coll.build_index(
            index_mode,
            field_name=field_name,
            n_clusters=effective_n_clusters,
        )
        if field_name == "default":
            self._default_index_built = True
        return {'status': 'success'}

    def remove_index(self, field_name: str = 'default'):
        """Remove the index of the collection.

        Parameters:
            field_name (str): Named vector field to remove index for.
                Defaults to "default" (the primary collection index).
        """
        self._rust_coll.remove_index(field_name=field_name)
        if field_name == "default":
            self._default_index = None
            self._default_index_built = False
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
            ids: List[Union[str, int]],
    ):
        """Attach vectors to a named vector field for existing IDs."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self._rust_coll.add_named_vectors(field_name, vecs, self._internal_ids(ids))
        self.COMMIT_FLAG = False
        return {'status': 'success'}

    def add_sparse_vectors(
            self,
            vectors: List[Union[Dict[int, float], List[Tuple[int, float]]]],
            ids: List[Union[str, int]],
    ):
        """Attach sparse feature vectors to existing IDs."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        normalized = [_normalize_sparse_vector(vector) for vector in vectors]
        self._rust_coll.add_sparse_vectors(normalized, self._internal_ids(ids))
        self.COMMIT_FLAG = False
        return {'status': 'success'}

    def insert_session(self):
        """Start an insert session."""
        from ..execution_layer.session import DataInsertionSession
        return DataInsertionSession(self)

    def search(
            self, vector: Union[list, np.ndarray, None] = None, k: int = 10, *,
            document: Union[str, None] = None,
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
            document (str): Text to embed and search semantically. Exactly one
                of ``vector`` or ``document`` must be provided.
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
                - **IVF / SPANN**: number of partitions to probe; higher improves recall and increases latency.
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
        if (vector is None) == (document is None):
            raise ValueError("search() requires exactly one of vector or document")
        if document is not None:
            vec = embed_documents([document])[0]
        else:
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
        external_ids, raw_fields = self._result_ids_and_fields(result.ids, fetch_fields=need_fields)
        rerank_fields = raw_fields if need_fields else []
        ids, distances, reranked_fields = apply_external_rerank(
            ids=external_ids,
            scores=result.distances,
            fields=rerank_fields,
            reranker=reranker,
            query={
                "type": "document_search" if document is not None else "vector_search",
                "document": document,
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
        external_ids, raw_fields = self._result_ids_and_fields(result.ids, fetch_fields=need_fields)
        rerank_fields = raw_fields if need_fields else []
        ids, distances, reranked_fields = apply_external_rerank(
            ids=external_ids,
            scores=result.distances,
            fields=rerank_fields,
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

    def bm25_search(
            self, text: str, k: int = 10, *,
            text_fields: Union[list[str], None] = None,
            where: Union[str, None] = None,
            return_fields: bool = False,
            reranker: Optional[Callable[[Dict[str, Any]], Any]] = None,
            rerank_k: Optional[int] = None,
            rerank_with_fields: bool = True,
    ):
        """BM25 keyword search over metadata fields."""
        result = self._rust_coll.text_search(text, text_fields=text_fields, k=k, where=where)
        need_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        external_ids, raw_fields = self._result_ids_and_fields(result.ids, fetch_fields=need_fields)
        rerank_fields = raw_fields if need_fields else []
        ids, distances, reranked_fields = apply_external_rerank(
            ids=external_ids,
            scores=result.distances,
            fields=rerank_fields,
            reranker=reranker,
            query={
                "type": "bm25_search",
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
        external_ids, raw_fields = self._result_ids_and_fields(result.ids, fetch_fields=need_fields)
        rerank_fields = raw_fields if need_fields else []
        ids, distances, reranked_fields = apply_external_rerank(
            ids=external_ids,
            scores=result.distances,
            fields=rerank_fields,
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
                - **IVF / SPANN**: number of partitions to probe — higher = better recall, slower.
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
            external_ids, raw_fields = self._result_ids_and_fields(r.ids, fetch_fields=need_fields)
            rerank_fields = raw_fields if need_fields else []
            ids, distances, reranked_fields = apply_external_rerank(
                ids=external_ids,
                scores=r.distances,
                fields=rerank_fields,
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
        result = self._rust_coll.head(n)
        external_ids, _ = self._result_ids_and_fields(result.ids, fetch_fields=False)
        return ResultView(
            vectors=result.vectors,
            ids=external_ids,
            fields=result.fields,
            result_type="data",
        )

    def tail(self, n: int = 5):
        """Get the last n items in the collection.

        Returns:
            ResultView: Data result with vectors, ids, and fields.
        """
        result = self._rust_coll.tail(n)
        external_ids, _ = self._result_ids_and_fields(result.ids, fetch_fields=False)
        return ResultView(
            vectors=result.vectors,
            ids=external_ids,
            fields=result.fields,
            result_type="data",
        )

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
            internal_ids = self._rust_coll.query_fields(where)
            ids_arr = id_array(self._rust_coll.external_ids(internal_ids)) if internal_ids else id_array([])
            if return_ids_only:
                return ResultView(ids=ids_arr, result_type="query")
            if internal_ids:
                fields = [dict(f) for f in self._rust_coll.retrieve_fields([int(i) for i in internal_ids])]
            else:
                fields = []
            return ResultView(ids=ids_arr, fields=fields, result_type="query")
        elif filter_ids is not None:
            ids = self._internal_ids(filter_ids)
        else:
            ids = []

        ids_arr = id_array(self._rust_coll.external_ids(ids)) if ids else id_array([])

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
            ids_arr = id_array(self._rust_coll.external_ids(ids)) if ids else id_array([])
            internal_ids = ids
        elif filter_ids is not None:
            internal_ids = self._internal_ids(filter_ids)
            ids_arr = id_array(self._rust_coll.external_ids(internal_ids)) if internal_ids else id_array([])
            fields = [dict(f) for f in self._rust_coll.retrieve_fields(internal_ids)] if internal_ids else []
        else:
            internal_ids = []
            ids_arr = np.array([], dtype=np.int64)
            fields = []

        if len(ids_arr) == 0:
            return ResultView(
                vectors=np.empty((0, self._rust_coll.dimension), dtype=np.float32),
                ids=ids_arr, fields=[], result_type="data",
            )

        vecs = self._rust_coll.get_vectors(internal_ids)
        return ResultView(vectors=vecs, ids=ids_arr, fields=list(fields), result_type="data")

    def delete(self, ids):
        """
        Soft-delete vectors by ID.

        Deleted IDs are excluded from all future search results. The raw vector
        data is NOT physically removed from disk; it is only tombstoned in memory
        and persisted as ``tombstone.bin``.

        Parameters:
            ids (list[int]): External IDs to soft-delete.
        """
        self._rust_coll.delete_items(self._internal_ids(ids, missing="ignore"))
        self.COMMIT_FLAG = False

    def restore(self, ids):
        """
        Restore previously soft-deleted vectors.

        Parameters:
            ids (list[int]): External IDs to restore.
        """
        self._rust_coll.restore_items(self._internal_ids(ids, missing="ignore"))
        self.COMMIT_FLAG = False

    def list_deleted_ids(self):
        """
        Return the sorted list of all currently soft-deleted IDs.

        Returns:
            list[int]: Sorted list of soft-deleted external IDs.
        """
        internal_ids = self._rust_coll.list_deleted_ids()
        return self._rust_coll.external_ids(internal_ids) if internal_ids else []

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
        result = self._rust_coll.search_range(vec, float(threshold), int(max_results))
        external_ids, _ = self._result_ids_and_fields(result.ids, fetch_fields=False)
        return ResultView(
            ids=external_ids,
            distances=result.distances,
            k=len(external_ids),
            distance=result.distance_metric,
            index=result.index_type,
            result_type="search",
        )

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
