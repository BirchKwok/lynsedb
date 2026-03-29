import json
import queue
import struct
import time
from typing import Union, List, Tuple
from threading import Lock

import numpy as np
import httpx

from tqdm import trange

from ...api import logger
from ...utils.asserts import raise_if
from ...utils.poster import Poster
from ...utils.utils import collection_repr
from ...result_view import ResultView, _parse_index_mode


def _decode_search_binary(buf: bytes, offset: int = 0):
    """Decode a compact binary search result block.

    Wire format:
        [4B n (u32 LE)]
        [n × 8B ids (u64 LE)]
        [n × 4B distances (f32 LE)]
        [4B fields_json_len (u32 LE)]
        [fields_json_len bytes UTF-8 JSON]

    Returns (ids, distances, fields, new_offset).
    """
    n = struct.unpack_from('<I', buf, offset)[0]; offset += 4
    ids = np.frombuffer(buf, dtype='<u8', count=n, offset=offset).astype(np.int64)
    offset += n * 8
    dists = np.frombuffer(buf, dtype='<f4', count=n, offset=offset).copy()
    offset += n * 4
    flen = struct.unpack_from('<I', buf, offset)[0]; offset += 4
    fields = json.loads(buf[offset:offset + flen]) if flen > 0 else []
    offset += flen
    return ids, dists, fields, offset


def _decode_vectors_binary(buf: bytes):
    """Decode a compact binary vectors block (head/tail).

    Wire format:
        [4B n (u32 LE)]
        [4B dim (u32 LE)]
        [n × dim × 4B vectors (f32 LE)]
        [n × 8B ids (u64 LE)]
        [4B fields_json_len (u32 LE)]
        [fields_json_len bytes UTF-8 JSON]

    Returns (vectors_2d, ids, fields).
    """
    off = 0
    n = struct.unpack_from('<I', buf, off)[0]; off += 4
    dim = struct.unpack_from('<I', buf, off)[0]; off += 4
    vecs = np.frombuffer(buf, dtype='<f4', count=n * dim, offset=off).reshape(n, dim).copy()
    off += n * dim * 4
    ids = np.frombuffer(buf, dtype='<u8', count=n, offset=off).astype(np.int64)
    off += n * 8
    flen = struct.unpack_from('<I', buf, off)[0]; off += 4
    fields = json.loads(buf[off:off + flen]) if flen > 0 else []
    return vecs, ids, fields


class ExecutionError(Exception):
    pass


def raise_error_response(response):
    """
    Raise an error response.

    Parameters:
        response: The response from the server.

    Raises:
        ExecutionError: If the server returns an error.
    """
    try:
        rj = response.json()
        raise ExecutionError(rj)
    except Exception as e:
        raise ExecutionError(response.text)




class HTTPClient:
    """
    The HTTPClient class is used to interact with the LynseDB server.
    """
    def __init__(self, uri, database_name):
        """
        Initialize the client.

        Parameters:
            uri (str): The URI of the server, must start with "http://" or "https://".
            database_name (str): The name of the database.

        Raises:
            TypeError: If the URI is not a string.
            ValueError: If the URI does not start with "http://" or "https://".
            ConnectionError: If the server cannot be connected to.
        """

        raise_if(TypeError, not isinstance(uri, str), 'The URI must be a string.')
        raise_if(ValueError, not uri.startswith('http://') or uri.startswith('https://'),
                 'The URI must start with "http://" or "https://".')

        self._session = httpx.Client()

        if uri.endswith('/'):
            self.uri = uri[:-1]
        else:
            self.uri = uri

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
        Create a collection.

        Parameters:
            collection (str): The name of the collection.
            dim (int): The dimension of the vectors. Default is None.
                When creating a new collection, the dimension of the vectors must be specified.
                When loading an existing collection, the dimension of the vectors is automatically loaded.
            n_threads (int): The number of threads. Default is 10.
            warm_up (bool): Whether to warm up. Default is False.
            drop_if_exists (bool): Whether to drop the collection if it exists. Default is False.
            description (str): A description of the collection. Default is None.
                The description is limited to 500 characters.

        Returns:
            Collection: The collection object.

        Raises:
            ConnectionError: If the server cannot be connected to.
        """
        uri = f'{self.uri}/required_collection'

        data = {
            "database_name": self.database_name,
            "collection_name": collection,
            "dim": dim,
            "n_threads": n_threads,
            "warm_up": warm_up,
            "drop_if_exists": drop_if_exists,
            "description": description,
        }

        try:
            response = self._session.post(uri, json=data)
            if response.status_code == 200:
                del data['collection_name']
                del data['database_name']
                collection = Collection(uri=self.uri, database_name=self.database_name,
                                        collection_name=collection, **data)
                return collection
            else:
                raise_error_response(response)
        except httpx.RequestError:
            raise ConnectionError(f'Failed to connect to the server at {uri}.')

    def get_collection(self, collection: str, warm_up=True):
        """
        Get a collection.

        Parameters:
            collection (str): The name of the collection.
            warm_up (bool): Whether to warm up. Default is True.

        Returns:
            Collection: The collection object.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self.uri}/is_collection_exists'
        data = {"database_name": self.database_name, "collection_name": collection}
        response = self._session.post(uri, json=data)

        if response.status_code == 200 and response.json()['params']['exists']:
            uri = f'{self.uri}/get_collection_config'
            data = {"database_name": self.database_name, "collection_name": collection}
            response = self._session.post(uri, json=data)

            params = response.json()['params']['config']
            params.update({'warm_up': warm_up})

            return Collection(uri=self.uri, database_name=self.database_name, collection_name=collection,
                              **params)
        else:
            raise_error_response(response)

    def drop_collection(self, collection: str):
        """
        Drop a collection.

        Parameters:
            collection (str): The name of the collection.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        try:
            _ = self.get_collection(collection)
        except ExecutionError:
            pass

        uri = f'{self.uri}/drop_collection'
        data = {"database_name": self.database_name, "collection_name": collection}
        return self._session.post(uri, json=data).json()

    def drop_database(self):
        """
        Drop the database.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        if not self.database_exists()['params']['exists']:
            return {'status': 'success', 'message': 'The database does not exist.'}

        uri = f'{self.uri}/drop_database'
        data = {"database_name": self.database_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def database_exists(self):
        """
        Check if the database exists.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self.uri}/database_exists'
        data = {"database_name": self.database_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def show_collections(self):
        """
        Show all collections in the database.

        Returns:
            List: The list of collections.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self.uri}/show_collections'
        data = {"database_name": self.database_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()['params']['collections']
        else:
            raise_error_response(response)

    def set_environment(self, env: dict):
        """
        Set the environment variables.

        Parameters:
            env (dict): The environment variables. It can be specified on the same time or separately.
                - LYNSE_LOG_LEVEL: The log level.
                - LYNSE_LOG_PATH: The log path.
                - LYNSE_TRUNCATE_LOG: Whether to truncate the log.
                - LYNSE_LOG_WITH_TIME: Whether to log with time.
                - LYNSE_KMEANS_EPOCHS: The number of epochs for KMeans.
                - LYNSE_SEARCH_CACHE_SIZE: The search cache size.
                - LYNSE_DATALOADER_BUFFER_SIZE: The dataloader buffer size.

        Returns:
            dict: The response from the server.

        Raises:
            TypeError: If the value of an environment variable is not a string.
            ExecutionError: If the server returns an error.
        """
        uri = f'{self.uri}/set_environment'

        env_list = ['LYNSE_LOG_LEVEL', 'LYNSE_LOG_PATH', 'LYNSE_TRUNCATE_LOG', 'LYNSE_LOG_WITH_TIME',
                    'LYNSE_KMEANS_EPOCHS', 'LYNSE_SEARCH_CACHE_SIZE', 'LYNSE_DATALOADER_BUFFER_SIZE']

        data = {"database_name": self.database_name}
        for key in env:
            if key in env_list:
                raise_if(TypeError, not isinstance(env[key], str), f'The value of {key} must be a string.')
                data[key] = env[key]

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def get_environment(self):
        """
        Get the environment variables.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self.uri}/get_environment'
        data = {"database_name": self.database_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def update_collection_description(self, collection: str, description: str):
        """
        Update the description of a collection.

        Parameters:
            collection (str): The name of the collection.
            description (str): The description of the collection.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self.uri}/update_collection_description'
        data = {"database_name": self.database_name, "collection_name": collection, "description": description}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def show_collections_details(self):
        """
        Show all collections in the database with details.

        Returns:
            pandas.DataFrame: The details of the collections.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self.uri}/show_collections_details'
        data = {"database_name": self.database_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            rj = response.json()['params']['collections']
            try:
                import pandas as pd
                rj = pd.DataFrame(rj)
            except ImportError:
                ...

            return rj
        else:
            raise_error_response(response)

    def __repr__(self):
        if self.database_exists()['params']['exists']:
            return f"RemoteDatabaseInstance(name={self.database_name}, exists=True)"
        else:
            return f"RemoteDatabaseInstance(name={self.database_name}, exists=False)"

    def __str__(self):
        return self.__repr__()


class Collection:
    """
    The Collection class is used to interact with a collection in the LynseDB.
    """
    name = "Remote"

    def __init__(self, uri, database_name, collection_name, **params):
        """
        Initialize the collection.

        Parameters:
            uri (str): The URI of the server.
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            **params: The collection parameters.
                - dim (int): The dimension of the vectors.
                - chunk_size (int): The chunk size.
                - cache_query (bool): Whether to use cache.
                - n_threads (int): The number of threads.
                - warm_up (bool): Whether to warm up.
                - drop_if_exists (bool): Whether to drop the collection if it exists.
                - cache_chunks (int): The number of chunks to cache.

        """
        self.IS_DELETED = False
        self._uri = uri
        self._database_name = database_name
        self._collection_name = collection_name
        self._session = Poster()
        self._init_params = params

        self.COMMIT_FLAG = False

        self._mesosphere_list = queue.Queue()
        self._lock = Lock()

    def exists(self):
        """
        Check if the collection exists.

        Returns:
            bool: Whether the collection exists.

        Raises:
            ExecutionError: If the server returns an error.

        """
        uri = f'{self._uri}/is_collection_exists'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()['params']['exists']
        else:
            raise_error_response(response)

    def add_item(self, vector: Union[list[float], np.ndarray], id: int, *,
                 field: Union[dict, None] = None,
                 buffer_size: int = True):
        """
        Add an item to the collection.
        It is recommended to use incremental ids for best performance.

        Parameters:
            vector (list[float], np.ndarray): The vector of the item.
            id (int): The ID of the item.
            field (dict, optional): The fields of the item.
            buffer_size (int or bool): The buffer size.
                Default is True, which means the default buffer size (1000) will be used.
                If buffer_size is 0, the function will add the item directly.
                If buffer_size is greater than 0, the function will add the item to the buffer.
                If buffer_size is False, the function will add the item directly and not use the buffer.
                If buffer_size is True, the function will add the item to the buffer and use the default buffer size.

        Returns:
            int: The ID of the item.
                If delay_num is greater than 0, and the number of items added is less than delay_num,
                the function will return None. Otherwise, the function will return the IDs of the items added.

        Raises:
            ValueError: If the collection has been deleted or does not exist.
            ExecutionError: If the server returns an error.
        """
        if buffer_size is True:
            buffer_size = 1000
        else:
            if buffer_size is False:
                buffer_size = 0
            else:
                raise_if(ValueError, (not isinstance(buffer_size, int)) or buffer_size < 0,
                         'If buffer_size is not bool, it must be a positive integer.')

        if buffer_size == 0:
            uri = f'{self._uri}/add_item'

            data = {
                "database_name": self._database_name,
                "collection_name": self._collection_name,
                "item": {
                    "vector": vector if isinstance(vector, list) else vector.tolist(),
                    "id": id,
                    "field": field if field is not None else {},
                },
            }

            response = self._session.post(uri, json=data)

            if response.status_code == 200:
                self.COMMIT_FLAG = False
                return response.json()['params']['item']['id']
            else:
                raise_error_response(response)
        else:
            with self._lock:
                self._mesosphere_list.put({
                    "vector": vector if isinstance(vector, list) else vector.tolist(),
                    "id": id,
                    "field": field if field is not None else {},
                })

            if self._mesosphere_list.qsize() >= buffer_size:
                mesosphere_list = list(self._mesosphere_list.queue)

                uri = f'{self._uri}/bulk_add_items'

                data = {
                    "database_name": self._database_name,
                    "collection_name": self._collection_name,
                    "items": mesosphere_list,
                }

                response = self._session.post(uri, json=data)

                if response.status_code == 200:
                    self.COMMIT_FLAG = False
                    self._mesosphere_list = queue.Queue()
                else:
                    raise_error_response(response)

            return id

    @staticmethod
    def _check_bulk_add_items(vectors):
        items = []
        for vector in vectors:
            raise_if(TypeError, not isinstance(vector, tuple), 'Each item must be a tuple of vector, '
                                                               'ID, and fields(optional).')
            vec_len = len(vector)

            if vec_len == 3:
                v1, v2, v3 = vector
                items.append({
                    "vector": v1.tolist() if isinstance(v1, np.ndarray) else v1,
                    "id": v2,
                    "field": v3,
                })
            elif vec_len == 2:
                v1, v2 = vector
                items.append({
                    "vector": v1.tolist() if isinstance(v1, np.ndarray) else v1,
                    "id": v2,
                    "field": {},
                })
            else:
                raise TypeError('Each item must be a tuple of vector, ID, and fields(optional).')

        return items

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
        It is recommended to use incremental ids for best performance.

        Parameters:
            vectors (List[Tuple[Union[List, Tuple, np.ndarray], int, dict]],
            List[Tuple[Union[List, Tuple, np.ndarray], int]]):
                The list of items to add. Each item is a tuple containing the vector, ID, and fields.
            batch_size (int): The batch size. Default is 1000.
            enable_progress_bar (bool): Whether to enable the progress bar. Default is True.

        Returns:
            dict: The response from the server.

        Raises:
            ValueError: If the collection has been deleted or does not exist.
            TypeError: If the vectors are not in the correct format.
            ExecutionError: If the server returns an error.
        """

        uri = f'{self._uri}/bulk_add_items'
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

            items_after_checking = self._check_bulk_add_items(items)

            data = {
                "database_name": self._database_name,
                "collection_name": self._collection_name,
                "items": items_after_checking,
            }

            response = self._session.post(uri, json=data)

            if response.status_code == 200:
                self.COMMIT_FLAG = False
                ids.extend(response.json()['params']['ids'])
            else:
                raise_error_response(response)

        return ids

    def bulk_add_binary(
            self,
            vectors: np.ndarray,
            batch_size: int = 50000,
            enable_progress_bar: bool = True
    ):
        """
        High-performance binary bulk add. Sends raw f32 bytes instead of JSON.
        Use this when adding vectors without fields for maximum throughput.

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

        n_total, dim = vectors.shape
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
            n_vecs = batch.shape[0]

            uri = (f'{self._uri}/bulk_add_binary'
                   f'?database_name={self._database_name}'
                   f'&collection_name={self._collection_name}'
                   f'&dim={dim}&n_vectors={n_vecs}')

            response = self._session.post(
                uri, content=batch.tobytes(),
                headers={'Content-Type': 'application/octet-stream'}
            )

            if response.status_code == 200:
                self.COMMIT_FLAG = False
                added += n_vecs
            else:
                raise_error_response(response)

        return added

    def commit(self):
        """
        Commit the changes in the collection.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/commit'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}

        if not self._mesosphere_list.empty():
            data["items"] = list(self._mesosphere_list.queue)

        self._mesosphere_list = queue.Queue()

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            self.COMMIT_FLAG = True
            return response.json()
        elif response.status_code == 202:
            task_id = response.json().get('task_id')
            status_uri = f'{self._uri}/status/{task_id}'

            while True:
                status_response = self._session.get(status_uri)
                status_data = status_response.json()

                if status_response.status_code == 200:
                    logger.info(f'Task status: {status_data}', rewrite_print=True)
                    if status_data['status'] == 'Error':
                        raise_error_response(status_response)
                    return status_data
                else:
                    raise_error_response(status_response)

                time.sleep(2)
        else:
            raise_error_response(response)

    def is_id_exists(self, id: int):
        """
        Check if an ID exists in the collection.

        Parameters:
            id (int): The ID to check.

        Returns:
            is_id_exists(Bool): Whether the ID exists in the collection.
        """
        uri = f'{self._uri}/is_id_exists'
        data = {"database_name": self._database_name, "collection_name": self._collection_name, "id": id}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()['params']['is_id_exists']
        else:
            raise_error_response(response)

    @property
    def max_id(self):
        """
        Get the maximum ID in the collection.

        Returns:
            int: The maximum ID in the collection.
        """
        uri = f'{self._uri}/max_id'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()['params']['max_id']
        else:
            raise_error_response(response)

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
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/build_index'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "index_mode": index_mode,
        }
        if 'n_clusters' in kwargs:
            data['n_clusters'] = kwargs['n_clusters']

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def remove_index(self):
        """
        Remove the index of the collection.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/remove_index'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def insert_session(self):
        """
        Start an insert session.
        """
        from ...execution_layer.session import DataInsertionSession

        return DataInsertionSession(self)

    def _search(self, vector, k, where, return_fields=False, **kwargs):
        """
        Search the collection using compact binary protocol.

        where is a SQL/WHERE expression string passed directly to ApexBase.
        """
        raise_if(ValueError, not isinstance(where, (type(None), str)),
                 'where must be a SQL/WHERE expression string or None.')

        vec = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        dim = len(vec)

        params = {
            'database_name': self._database_name,
            'collection_name': self._collection_name,
            'dim': dim,
            'k': k,
            'return_fields': str(return_fields).lower(),
        }
        if where is not None:
            params['where'] = where
        nprobe = kwargs.get('nprobe')
        if nprobe is not None:
            params['nprobe'] = nprobe

        uri = f'{self._uri}/search_binary'
        response = self._session.post(
            uri, params=params, content=vec.tobytes(),
            headers={'Content-Type': 'application/octet-stream'},
        )

        if response.status_code == 200:
            return response.content  # raw binary
        else:
            raise_error_response(response)

    def search(
            self, vector: Union[list[float], np.ndarray], k: int = 10, *,
            where: Union[str, None] = None,
            return_fields: bool = False, **kwargs
    ):
        """
        Search the database for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray or list): The search vectors, it can be a single vector or a list of vectors.
                The vectors must have the same dimension as the vectors in the database,
                and the type of vector can be a list or a numpy array.
            k (int): The number of nearest vectors to return.
            where (str, optional): SQL/WHERE expression string to filter results.
            return_fields (bool): Whether to return the fields of the search results.
            **kwargs: Additional keyword arguments:

                - nprobe (int): Controls search breadth by index type (default: 10).
                    - **IVF**: number of partitions to probe — higher = better recall, slower.
                    - **HNSW**: ef_search beam width — higher = better recall, slower.
                    - **Flat / PQ / RaBitQ / PolarVec**: ignored (exhaustive two-pass search).
                - rescore (bool): Re-score binary/SQ8 quantization results with exact f32.
                    Recommended when index mode contains 'Binary'. Default: False.
                - rescore_multiplier (int): Candidate pool size multiplier for rescoring.
                    Active only when rescore is True. Default: 10 for Binary, 2 otherwise.

        Returns:
            ResultView: Search results with ids, distances, and optional fields.

        Raises:
            ValueError: If the collection has been deleted or does not exist.
            ExecutionError: If the server returns an error.
        """
        buf = self._search(vector=vector, k=k, where=where, return_fields=return_fields, **kwargs)
        ids, dists, fields, _ = _decode_search_binary(buf)
        idx_type, metric = _parse_index_mode(self.index_mode)
        return ResultView(
            ids=ids, distances=dists, fields=fields,
            k=k, distance=metric, index=idx_type,
            result_type="search",
        )

    def batch_search(
            self, vectors: Union[list, np.ndarray], k: int = 10, *,
            where: Union[str, None] = None,
            return_fields: bool = False, nprobe: int = 10
    ):
        """
        Batch search: search multiple query vectors in a single request.
        Uses compact binary protocol + Rust's parallel batch_search for maximum throughput.

        Parameters:
            vectors (np.ndarray or list): Multiple query vectors, shape (n, dim).
            k (int): The number of nearest vectors to return per query.
            where (str, optional): SQL/WHERE expression string to filter results.
            return_fields (bool): Whether to return the fields of the search results.
            nprobe (int): Controls search breadth by index type (default: 10).
                - **IVF**: number of partitions to probe — higher = better recall, slower.
                - **HNSW**: ef_search beam width — higher = better recall, slower.
                - **Flat / PQ / RaBitQ / PolarVec**: ignored (exhaustive two-pass search).

        Returns:
            List[ResultView]: List of ResultView objects, one per query vector.
        """
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        n_queries, dim = vecs.shape

        params = {
            'database_name': self._database_name,
            'collection_name': self._collection_name,
            'dim': dim,
            'n_queries': n_queries,
            'k': k,
            'return_fields': str(return_fields).lower(),
            'nprobe': nprobe,
        }
        if where is not None:
            params['where'] = where

        uri = f'{self._uri}/batch_search_binary'
        response = self._session.post(
            uri, params=params, content=vecs.tobytes(),
            headers={'Content-Type': 'application/octet-stream'},
        )

        if response.status_code == 200:
            buf = response.content
            nq = struct.unpack_from('<I', buf, 0)[0]
            off = 4
            idx_type, metric = _parse_index_mode(self.index_mode)
            output = []
            for _ in range(nq):
                ids, dists, fields, off = _decode_search_binary(buf, off)
                output.append(ResultView(
                    ids=ids, distances=dists, fields=fields,
                    k=k, distance=metric, index=idx_type,
                    result_type="search",
                ))
            return output
        else:
            raise_error_response(response)

    @property
    def shape(self):
        """
        Get the shape of the collection.

        Returns:
            Tuple: The shape of the collection.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/collection_shape'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return tuple(response.json()['params']['shape'])
        else:
            rj = response.json()
            if 'error' in rj and rj['error'] == f"Collection '{self._collection_name}' does not exist.":
                return 0, self._init_params['dim']
            else:
                raise_error_response(response)

    def head(self, n: int = 5):
        """
        Get the first n items in the collection (compact binary protocol).

        Parameters:
            n (int): The number of items to return. Default is 5.

        Returns:
            ResultView: Data result with vectors, ids, and fields.

        Raises:
            ExecutionError: If the server returns an error.
        """
        params = {
            'database_name': self._database_name,
            'collection_name': self._collection_name,
            'n': n,
        }
        response = self._session.get(f'{self._uri}/head_binary', params=params)

        if response.status_code == 200:
            vecs, ids, fields = _decode_vectors_binary(response.content)
            return ResultView(
                vectors=vecs, ids=ids, fields=fields,
                result_type="data",
            )
        else:
            raise_error_response(response)

    def tail(self, n: int = 5):
        """
        Get the last n items in the collection (compact binary protocol).

        Parameters:
            n (int): The number of items to return. Default is 5.

        Returns:
            ResultView: Data result with vectors, ids, and fields.

        Raises:
            ExecutionError: If the server returns an error.
        """
        params = {
            'database_name': self._database_name,
            'collection_name': self._collection_name,
            'n': n,
        }
        response = self._session.get(f'{self._uri}/tail_binary', params=params)

        if response.status_code == 200:
            vecs, ids, fields = _decode_vectors_binary(response.content)
            return ResultView(
                vectors=vecs, ids=ids, fields=fields,
                result_type="data",
            )
        else:
            raise_error_response(response)

    def read_by_only_id(self, id: Union[int, list]):
        """
        Read the item by ID.

        Parameters:
            id (int, list): The ID of the item or a list of IDs.

        Returns:
            ResultView: Data result with vectors, ids, and fields.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/read_by_only_id'
        data = {"database_name": self._database_name, "collection_name": self._collection_name, "id": id}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            item = response.json()['params']['item']
            vecs = np.asarray(item[0], dtype=np.float32)
            ids = np.array(item[1], dtype=np.int64)
            fields = item[2] if len(item) > 2 else []
            return ResultView(
                vectors=vecs, ids=ids, fields=fields,
                result_type="data",
            )
        else:
            raise_error_response(response)

    def query(self, where=None, filter_ids=None, return_ids_only=False):
        """
        Query the collection.

        Parameters:
            where (str or None): SQL/WHERE expression string to filter fields.
            filter_ids (list[int]): The list of IDs to filter.
            return_ids_only (bool): Whether to return the IDs only.

        Returns:
            ResultView: Query result with ids and optional fields.

        Raises:
            ExecutionError: If the server returns an error.
        """

        uri = f'{self._uri}/query'

        raise_if(ValueError, not isinstance(where, (str, type(None))),
                 'where must be a SQL/WHERE expression string or None.')

        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "where": where,
            "filter_ids": filter_ids,
            "return_ids_only": return_ids_only
        }

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            result = response.json()['params']['result']
            if return_ids_only:
                ids_arr = np.array(result, dtype=np.int64) if result else np.array([], dtype=np.int64)
                return ResultView(ids=ids_arr, result_type="query")
            else:
                # result is a list of dicts
                records = result if isinstance(result, list) else []
                ids_list = [r.get('id', i) for i, r in enumerate(records)] if records else []
                ids_arr = np.array(ids_list, dtype=np.int64) if ids_list else np.array([], dtype=np.int64)
                return ResultView(ids=ids_arr, fields=records, result_type="query")
        else:
            raise_error_response(response)

    def query_vectors(self, where=None, filter_ids=None):
        """
        Query the vector data by the filter.

        Parameters:
            where (str or None):
                SQL/WHERE expression string to filter fields.
            filter_ids (list[int]):
                The list of external IDs to filter. Default is None.

        Returns:
            ResultView: Data result with vectors, ids, and fields.
        """
        uri = f'{self._uri}/query_vectors'

        raise_if(ValueError, not isinstance(where, (str, type(None))),
                 'where must be a SQL/WHERE expression string or None.')

        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "where": where,
            "filter_ids": filter_ids,
        }

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            result = response.json()['params']['result']
            vecs = np.asarray(result[0], dtype=np.float32)
            ids = np.array(result[1], dtype=np.int64)
            fields = result[2] if len(result) > 2 else []
            return ResultView(
                vectors=vecs, ids=ids, fields=fields,
                result_type="data",
            )
        else:
            raise_error_response(response)

    def list_fields(self):
        """
        List all fields of a collection.

        Returns:
            dict: The status of the operation.
        """
        uri = f'{self._uri}/list_fields'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()['params']['fields']
        else:
            raise_error_response(response)

    def update_description(self, description: str):
        """
        Update the description of the collection.

        Parameters:
            description (str): The description of the collection.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/update_description'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "description": description
        }

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def get_collection_path(self):
        """
        Get the path of the database.

        Returns:
            str: The path of the database.
        """
        uri = f'{self._uri}/get_collection_path'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()['params']['collection_path']
        else:
            raise_error_response(response)

    @property
    def index_mode(self):
        """
        Get the index mode of the collection.

        Returns:
            str: The index mode of the collection.
        """
        uri = f'{self._uri}/index_mode'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()['params']['index_mode']
        else:
            raise_error_response(response)

    def __repr__(self):
        return collection_repr(self)

    def __str__(self):
        return self.__repr__()
