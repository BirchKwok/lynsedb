"""
Local client for LynseDB — direct Rust backend access without HTTP.

Mirrors the API of HTTPClient and Collection from http_api/client_api.py,
but calls the Rust PyO3 bindings directly, eliminating all network I/O overhead.
"""

import queue
from typing import Union, List, Tuple, Dict, Any, Optional
from threading import Lock

import numpy as np
from tqdm import trange

from ..utils.utils import collection_repr
from .._backend import DatabaseManager, Collection, SearchResult
from ..result_view import ResultView, _parse_index_mode


class LocalClient:
    """
    Local database client — direct Rust backend, no HTTP.

    Drop-in replacement for HTTPClient when running in local mode.
    """

    def __init__(self, manager: DatabaseManager, database_name: str):
        self._manager = manager
        self.database_name = database_name

    def require_collection(
            self,
            collection: str,
            dim: int = None,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            drop_if_exists: bool = False,
            description: str = None,
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

        Returns:
            LocalCollection: The collection object.
        """
        self._manager.require_collection(
            self.database_name, collection, dim, drop_if_exists, description
        )
        rust_coll = self._manager.get_collection(
            self.database_name, collection, dim
        )
        params = {
            'dim': dim,
            'n_threads': n_threads,
            'warm_up': warm_up,
            'drop_if_exists': drop_if_exists,
            'description': description,
        }
        return LocalCollection(
            manager=self._manager,
            database_name=self.database_name,
            collection_name=collection,
            rust_collection=rust_coll,
            **params,
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

        rust_coll = self._manager.get_collection(
            self.database_name, collection, dim
        )
        params = {
            'dim': dim,
            'warm_up': warm_up,
        }
        return LocalCollection(
            manager=self._manager,
            database_name=self.database_name,
            collection_name=collection,
            rust_collection=rust_coll,
            **params,
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

    def __init__(self, manager, database_name, collection_name, rust_collection, **params):
        self.IS_DELETED = False
        self._manager = manager
        self._database_name = database_name
        self._collection_name = collection_name
        self._rust_coll = rust_collection
        self._init_params = params

        self.COMMIT_FLAG = False
        self._mesosphere_list = queue.Queue()
        self._lock = Lock()

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
        """
        if buffer_size is True:
            buffer_size = 2**31
        else:
            if buffer_size is False:
                buffer_size = 0
            elif not isinstance(buffer_size, int) or buffer_size < 0:
                raise ValueError('If buffer_size is not bool, it must be a positive integer.')

        if buffer_size == 0:
            vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
            fields = [field] if field else None
            self._rust_coll.add_items(vec, fields)
            self.COMMIT_FLAG = False
            return id
        else:
            with self._lock:
                self._mesosphere_list.put({
                    "vector": vector if isinstance(vector, list) else vector.tolist(),
                    "id": id,
                    "field": field if field is not None else {},
                })

            if self._mesosphere_list.qsize() >= buffer_size:
                self._flush_buffer()

            return id

    def _flush_buffer(self):
        """Flush the internal buffer to the Rust collection."""
        items = list(self._mesosphere_list.queue)
        if not items:
            return
        vectors = np.array([item['vector'] for item in items], dtype=np.float32)
        fields = [item.get('field', {}) for item in items]
        has_fields = any(f for f in fields)
        self._rust_coll.add_items(vectors, fields if has_fields else None)
        self.COMMIT_FLAG = False
        self._mesosphere_list = queue.Queue()

    def bulk_add_items(
            self,
            vectors: List[Union[
                Tuple[Union[List, Tuple, np.ndarray], int, dict],
                Tuple[Union[List, Tuple, np.ndarray], int]
            ]],
            batch_size: int = 1000,
            enable_progress_bar: bool = True
    ):
        """
        Add multiple items to the collection.

        Parameters:
            vectors: List of tuples (vector, id, fields) or (vector, id).
            batch_size (int): The batch size. Default is 1000.
            enable_progress_bar (bool): Whether to enable the progress bar.

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
            self._rust_coll.add_items(vec_array, batch_fields if has_fields else None)
            self.COMMIT_FLAG = False
            ids.extend(batch_ids)

        return ids

    def bulk_add_binary(
            self,
            vectors: np.ndarray,
            batch_size: int = 50000,
            enable_progress_bar: bool = True
    ):
        """
        High-performance binary bulk add. Directly passes numpy arrays to Rust.

        Parameters:
            vectors (np.ndarray): 2D array of shape (n, dim), dtype float32.
            batch_size (int): Number of vectors per batch. Default is 50000.
            enable_progress_bar (bool): Whether to enable the progress bar.

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

        for i in iter_obj:
            start = i * batch_size
            end = min((i + 1) * batch_size, n_total)
            batch = vectors[start:end]
            self._rust_coll.add_items(batch, None)
            self.COMMIT_FLAG = False
            added += batch.shape[0]

        return added

    def commit(self):
        """Commit the changes in the collection."""
        if not self._mesosphere_list.empty():
            self._flush_buffer()
        self._rust_coll.commit()
        self.COMMIT_FLAG = True
        return {'status': 'success'}

    def is_id_exists(self, id: int):
        """Check if an ID exists in the collection."""
        # Use query_fields or retrieve_fields to check
        try:
            fields = self._rust_coll.retrieve_fields([id])
            return len(fields) > 0
        except Exception:
            return False

    @property
    def max_id(self):
        """Get the maximum ID in the collection."""
        n_rows, _ = self._rust_coll.shape
        if n_rows == 0:
            return -1
        return int(n_rows - 1)

    def build_index(self, index_mode: str = 'FLAT', **kwargs):
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
            kwargs: Additional keyword arguments. The following are available:

                - 'n_clusters' (int): The number of clusters. Only available for IVF modes.

        Returns:
            dict: Status message.
        """
        self._rust_coll.build_index(index_mode)
        return {'status': 'success'}

    def remove_index(self):
        """Remove the index of the collection."""
        self._rust_coll.remove_index()
        return {'status': 'success'}

    def insert_session(self):
        """Start an insert session."""
        from ..execution_layer.session import DataInsertionSession
        return DataInsertionSession(self)

    def search(
            self, vector: Union[list, np.ndarray], k: int = 10, *,
            where: Union[str, None] = None,
            return_fields: bool = False, **kwargs
    ):
        """
        Search the collection for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray or list): The search vector.
            k (int): The number of nearest vectors to return.
            where (str, optional): SQL/WHERE expression string to filter results.
            return_fields (bool): Whether to return the fields of the search results.
            **kwargs: Additional keyword arguments:

                - nprobe (int): Controls search breadth by index type (default: 10).
                    - **IVF**: number of partitions to probe — higher = better recall, slower.
                    - **HNSW**: ef_search beam width — higher = better recall, slower.
                    - **Flat / PQ / RaBitQ / PolarVec**: ignored (exhaustive two-pass search).

        Returns:
            ResultView: Search results with ids, distances, and optional fields.
        """
        nprobe = kwargs.get('nprobe', 10)
        result = self._rust_coll.search(
            np.ascontiguousarray(vector, dtype=np.float32).ravel(),
            k=k,
            where=where,
            nprobe=nprobe,
        )
        fields = []
        if return_fields and len(result) > 0:
            fields = self._rust_coll.retrieve_fields(result.ids.tolist())
        return ResultView(
            ids=result.ids, distances=result.distances, fields=fields,
            k=result.k, distance=result.distance_metric, index=result.index_type,
            result_type="search",
        )

    def batch_search(
            self, vectors: Union[list, np.ndarray], k: int = 10, *,
            where: Union[str, None] = None,
            return_fields: bool = False, nprobe: int = 10
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

        Returns:
            List[ResultView]: List of ResultView objects, one per query vector.
        """
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        results = self._rust_coll.batch_search(vecs, k=k, where=where, nprobe=nprobe)
        output = []
        for r in results:
            fields = []
            if return_fields and len(r) > 0:
                fields = self._rust_coll.retrieve_fields(r.ids.tolist())
            output.append(ResultView(
                ids=r.ids, distances=r.distances, fields=fields,
                k=r.k, distance=r.distance_metric, index=r.index_type,
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
            # Single-call: get both IDs and fields in one ApexBase query
            ids, fields = self._rust_coll.query_with_fields(where)
            ids_arr = np.array(ids, dtype=np.int64) if ids else np.array([], dtype=np.int64)
            if return_ids_only:
                return ResultView(ids=ids_arr, result_type="query")
            return ResultView(ids=ids_arr, fields=list(fields), result_type="query")
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
