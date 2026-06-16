import json
import queue
import struct
import time
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional, Callable
from threading import Lock

import numpy as np
import httpx

from ...cluster import _encode_fields_binary, _encode_ids_for_wire, _normalize_vector_encoding
from ...api import logger
from ...utils.asserts import raise_if
from ...utils.poster import Poster
from ...utils.utils import collection_repr
from ...result_view import ResultView, _parse_index_mode
from .._embedding import embed_documents
from .._records import (
    attach_documents,
    id_array,
    normalize_documents,
    normalize_external_ids,
    normalize_fields,
    normalize_vectors,
    validate_unique_external_ids,
)
from ..rerank import apply_external_rerank, should_fetch_fields


DEFAULT_COLLECTION_INDEX = "FLAT-IP"


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


def _normalize_sparse_vector(vector: Any) -> List[Tuple[int, float]]:
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


def _sparse_vector_payload(vector: Any) -> Tuple[Dict[str, list], List[Tuple[int, float]]]:
    normalized = _normalize_sparse_vector(vector)
    return {
        "indices": [int(index) for index, _ in normalized],
        "values": [float(value) for _, value in normalized],
    }, normalized


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
    def __init__(self, uri, database_name, api_key=None):
        """
        Initialize the client.

        Parameters:
            uri (str): The URI of the server, must start with "http://" or "https://".
            database_name (str): The name of the database.
            api_key (str or None): Optional Bearer token for authorization.

        Raises:
            TypeError: If the URI is not a string.
            ValueError: If the URI does not start with "http://" or "https://".
            ConnectionError: If the server cannot be connected to.
        """

        raise_if(TypeError, not isinstance(uri, str), 'The URI must be a string.')
        raise_if(ValueError, not (uri.startswith('http://') or uri.startswith('https://')),
                 'The URI must start with "http://" or "https://".')

        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        self._session = httpx.Client(headers=headers)
        self._api_key = api_key

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
            dtypes: str = "float32",
            default_index: Union[str, None] = DEFAULT_COLLECTION_INDEX,
    ):
        """
        Create a collection.

        Parameters:
            collection (str): The name of the collection.
            dim (int): Optional vector dimension. If omitted for a new
                collection, LynseDB infers it from the first inserted vectors.
            n_threads (int): The number of threads. Default is 10.
            warm_up (bool): Whether to warm up. Default is False.
            drop_if_exists (bool): Whether to drop the collection if it exists. Default is False.
            description (str): A description of the collection. Default is None.
                The description is limited to 500 characters.
            dtypes (str): Dense vector storage dtype, "float32" or "float16".
            default_index (str or None): Index mode to build automatically after
                the first write to a newly created collection. Use None to
                disable automatic index creation.

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
            "dtypes": dtypes,
        }

        try:
            existed_before = False
            if not drop_if_exists:
                exists_response = self._session.post(
                    f'{self.uri}/is_collection_exists',
                    json={"database_name": self.database_name, "collection_name": collection},
                )
                if exists_response.status_code == 200:
                    existed_before = exists_response.json()['params']['exists']
                else:
                    raise_error_response(exists_response)

            response = self._session.post(uri, json=data)
            if response.status_code == 200:
                collection = Collection(
                    uri=self.uri,
                    database_name=self.database_name,
                    collection_name=collection,
                    dim=dim,
                    n_threads=n_threads,
                    warm_up=warm_up,
                    drop_if_exists=drop_if_exists,
                    description=description,
                    dtypes=dtypes,
                    api_key=self._api_key,
                    default_index=default_index if not existed_before else None,
                )
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

            config = response.json()['params']['config']

            return Collection(
                uri=self.uri,
                database_name=self.database_name,
                collection_name=collection,
                dim=config.get('dim'),
                chunk_size=config.get('chunk_size'),
                description=config.get('description'),
                dtypes=config.get('dtypes', 'float32'),
                warm_up=warm_up,
                api_key=self._api_key,
            )
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

    def snapshot_collection(self, collection: str, snapshot_path: Union[str, Path]):
        """Create a filesystem snapshot for a collection on the server."""
        uri = f'{self.uri}/snapshot_collection'
        data = {
            "database_name": self.database_name,
            "collection_name": collection,
            "snapshot_path": str(snapshot_path),
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def export_collection(self, collection: str, export_path: Union[str, Path]):
        """Export a collection as JSONL metadata plus binary vectors on the server."""
        uri = f'{self.uri}/export_collection'
        data = {
            "database_name": self.database_name,
            "collection_name": collection,
            "export_path": str(export_path),
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def restore_collection(
            self,
            collection: str,
            snapshot_path: Union[str, Path],
            overwrite: bool = False,
    ):
        """Restore a collection from a filesystem snapshot on the server."""
        uri = f'{self.uri}/restore_collection'
        data = {
            "database_name": self.database_name,
            "collection_name": collection,
            "snapshot_path": str(snapshot_path),
            "overwrite": overwrite,
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def import_collection(
            self,
            collection: str,
            export_path: Union[str, Path],
            overwrite: bool = False,
    ):
        """Import a collection from JSONL metadata plus binary vectors on the server."""
        uri = f'{self.uri}/import_collection'
        data = {
            "database_name": self.database_name,
            "collection_name": collection,
            "export_path": str(export_path),
            "overwrite": overwrite,
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def snapshot_database(self, snapshot_path: Union[str, Path]):
        """Create a filesystem snapshot for this database on the server."""
        uri = f'{self.uri}/snapshot_database'
        data = {
            "database_name": self.database_name,
            "snapshot_path": str(snapshot_path),
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def restore_database(
            self,
            snapshot_path: Union[str, Path],
            overwrite: bool = False,
    ):
        """Restore this database from a filesystem snapshot on the server."""
        uri = f'{self.uri}/restore_database'
        data = {
            "database_name": self.database_name,
            "snapshot_path": str(snapshot_path),
            "overwrite": overwrite,
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

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

    def __init__(
            self,
            uri,
            database_name,
            collection_name,
            dim: Union[int, None] = None,
            chunk_size: Union[int, None] = None,
            description: Union[str, None] = None,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            drop_if_exists: bool = False,
            cache_query: Union[bool, None] = None,
            cache_chunks: Union[int, None] = None,
            api_key: Union[str, None] = None,
            dtypes: str = "float32",
            default_index: Union[str, None] = None,
    ):
        """
        Initialize the collection.

        Parameters:
            uri (str): The URI of the server.
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            dim (int): Optional vector dimension.
            chunk_size (int): The chunk size.
            description (str): Optional collection description.
            n_threads (int): The number of threads.
            warm_up (bool): Whether to warm up.
            drop_if_exists (bool): Whether to drop the collection if it exists.
            cache_query (bool): Whether to use cache. Currently ignored by the
                Rust HTTP client.
            cache_chunks (int): The number of chunks to cache. Currently
                ignored by the Rust HTTP client.
            api_key (str): Optional Bearer token.
            dtypes (str): Dense vector storage dtype.
            default_index (str or None): Deferred default index mode for newly
                created collections.

        """
        self.IS_DELETED = False
        self._uri = uri
        self._database_name = database_name
        self._collection_name = collection_name
        self._session = Poster(api_key=api_key)
        self._init_params = {
            'dim': dim,
            'chunk_size': chunk_size,
            'description': description,
            'n_threads': n_threads,
            'warm_up': warm_up,
            'drop_if_exists': drop_if_exists,
            'cache_query': cache_query,
            'cache_chunks': cache_chunks,
            'dtypes': dtypes,
        }
        self._default_index = default_index
        self._default_index_built = False

        self.COMMIT_FLAG = False

        self._mesosphere_list = queue.Queue()
        self._lock = Lock()
        self._cluster_mode = False
        try:
            response = self._session.get(f'{self._uri}/cluster_info')
            self._cluster_mode = response.status_code == 200
        except Exception:
            self._cluster_mode = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and not self.COMMIT_FLAG:
            self.commit()
        return False

    def _set_dim_if_known(self, dim):
        if dim:
            self._init_params['dim'] = int(dim)

    def _maybe_build_default_index(self):
        if not self._default_index or self._default_index_built:
            return
        current_index = self.index_mode
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
        return self._init_params.get('dtypes', 'float32')

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
        """Add one or more records through the unified insert API."""
        if ids is None:
            if vectors is None or documents is not None or fields is not None:
                raise ValueError("add(ids=None) requires vectors and does not accept documents or fields")
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            elif vectors.ndim != 2:
                raise ValueError("vectors must be a 1D vector or a 2D matrix")
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            n_total, dim = vectors.shape
            start_id = int(self.max_id) + 1
            generated_ids = list(range(start_id, start_id + n_total))
            vector_encoding = _normalize_vector_encoding(wire_dtype)
            total_batches = (n_total + batch_size - 1) // batch_size

            for i in range(total_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_total)
                batch = vectors[start:end]
                n_vecs = batch.shape[0]
                if vector_encoding == "float16":
                    body = np.ascontiguousarray(batch, dtype="<f2").tobytes()
                else:
                    body = batch.tobytes()

                uri = (f'{self._uri}/bulk_add_binary'
                       f'?database_name={self._database_name}'
                       f'&collection_name={self._collection_name}'
                       f'&dim={dim}&n_vectors={n_vecs}'
                       f'&vector_encoding={vector_encoding}')

                response = self._session.post(
                    uri, content=body, headers={'Content-Type': 'application/octet-stream'}
                )
                if response.status_code == 200:
                    self.COMMIT_FLAG = False
                    self._set_dim_if_known(dim)
                else:
                    raise_error_response(response)

            self._maybe_build_default_index()
            return generated_ids[0] if n_total == 1 else generated_ids

        del wire_dtype  # JSON HTTP add sends float32-compatible lists.
        external_ids, single_id = normalize_external_ids(ids)
        n_records = len(external_ids)
        validate_unique_external_ids(external_ids)
        docs, _ = normalize_documents(documents, n_records) if documents is not None else (None, False)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if vectors is None:
            if docs is None:
                raise ValueError("add() requires vectors or documents")
            vec_array = embed_documents(docs)
            if vec_array.shape[0] != n_records:
                raise ValueError("embedding output count must match ids length")
        else:
            vec_array = normalize_vectors(vectors, n_records)

        field_list = attach_documents(normalize_fields(fields, n_records), docs)
        uri = f'{self._uri}/add'
        added_ids = []

        with self._lock:
            for start in range(0, n_records, batch_size):
                end = min(start + batch_size, n_records)
                data = {
                    "database_name": self._database_name,
                    "collection_name": self._collection_name,
                    "ids": external_ids[start:end],
                    "vectors": vec_array[start:end].tolist(),
                    "fields": field_list[start:end],
                }
                response = self._session.post(uri, json=data)
                if response.status_code == 200:
                    self.COMMIT_FLAG = False
                    self._set_dim_if_known(vec_array.shape[1])
                    added_ids.extend(response.json()['params']['ids'])
                else:
                    raise_error_response(response)

        self._maybe_build_default_index()
        return added_ids[0] if single_id else added_ids

    @staticmethod
    def _prepare_binary_items(vectors, *, upsert: bool, wire_dtype: str = "float32"):
        rows = []
        ids = []
        fields = []
        has_field_payload = False
        dim = None

        for item in vectors:
            raise_if(TypeError, not isinstance(item, tuple), 'Each item must be a tuple of vector, '
                                                             'ID, and fields(optional).')
            vec_len = len(item)
            if vec_len not in (2, 3):
                raise TypeError('Each item must be a tuple of vector, ID, and fields(optional).')

            vector = np.asarray(item[0], dtype=np.float32).ravel()
            if dim is None:
                dim = int(vector.size)
            elif int(vector.size) != dim:
                raise ValueError("all vectors in a binary batch must have the same dimension")
            rows.append(vector)

            item_id = int(item[1])
            if item_id < 0:
                raise ValueError("item IDs must be non-negative")
            ids.append(item_id)

            if upsert:
                if vec_len == 3:
                    field = item[2]
                    has_field_payload = True
                    fields.append(field if field is not None else None)
                else:
                    fields.append(None)
            else:
                field = item[2] if vec_len == 3 else {}
                field = field or {}
                if not isinstance(field, dict):
                    raise TypeError("field payload entries must be dict or None")
                if field:
                    has_field_payload = True
                fields.append(field)

        if not rows:
            return b"", 0, 0, [], {"ids_encoding": "raw"}

        matrix = np.vstack(rows).astype(np.float32, copy=False)
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        vector_encoding = _normalize_vector_encoding(wire_dtype)
        id_raw, id_params = _encode_ids_for_wire(ids)
        if vector_encoding == "float16":
            vector_payload = np.ascontiguousarray(matrix, dtype="<f2").tobytes()
            id_params["vector_encoding"] = "float16"
        else:
            vector_payload = matrix.tobytes()
            id_params["vector_encoding"] = "float32"
        payload = vector_payload + id_raw
        if has_field_payload:
            payload += _encode_fields_binary(fields)
        return payload, matrix.shape[0], matrix.shape[1], ids, id_params

    def _post_binary_items(self, endpoint: str, payload: bytes, n_vectors: int, dim: int, id_params: dict):
        params = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "dim": dim,
            "n_vectors": n_vectors,
            "return_ids": "false",
            **id_params,
        }
        return self._session.post(
            f'{self._uri}/{endpoint}',
            params=params,
            content=payload,
            headers={'Content-Type': 'application/octet-stream'},
        )

    def upsert(
            self,
            ids,
            *,
            vectors,
            fields=None,
            batch_size: int = 1000,
            wire_dtype: str = "float32",
    ):
        """Insert or update one or more records by public ID."""
        del wire_dtype  # JSON HTTP upsert sends float32-compatible lists.
        external_ids, single_id = normalize_external_ids(ids)
        n_records = len(external_ids)
        validate_unique_external_ids(external_ids)
        vec_array = normalize_vectors(vectors, n_records)
        field_list = normalize_fields(fields, n_records) if fields is not None else None
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        uri = f'{self._uri}/upsert'
        returned_ids = []
        with self._lock:
            for start in range(0, n_records, batch_size):
                end = min(start + batch_size, n_records)
                data = {
                    "database_name": self._database_name,
                    "collection_name": self._collection_name,
                    "ids": external_ids[start:end],
                    "vectors": vec_array[start:end].tolist(),
                    "fields": field_list[start:end] if field_list is not None else None,
                }
                response = self._session.post(uri, json=data)
                if response.status_code == 200:
                    self.COMMIT_FLAG = False
                    self._set_dim_if_known(vec_array.shape[1])
                    returned_ids.extend(response.json()['params']['ids'])
                else:
                    raise_error_response(response)

        self._maybe_build_default_index()
        return returned_ids[0] if single_id else returned_ids

    def _flush_pending(self):
        if self._mesosphere_list.empty():
            return

        queued = list(self._mesosphere_list.queue)
        uri = f'{self._uri}/add'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "ids": [item["id"] for item in queued],
            "vectors": [item["vector"] for item in queued],
            "fields": [item.get("field", {}) for item in queued],
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            self.COMMIT_FLAG = False
            vectors = data.get("vectors") or []
            if vectors:
                self._set_dim_if_known(len(vectors[0]))
            self._mesosphere_list = queue.Queue()
        else:
            raise_error_response(response)

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

    def flush(self):
        """Flush pending bytes and fsync collection files without clearing WAL."""
        uri = f'{self._uri}/flush'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}

        if not self._mesosphere_list.empty():
            self._flush_pending()

        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def checkpoint(self):
        """Checkpoint durable state and clear WAL."""
        uri = f'{self._uri}/checkpoint'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}

        if not self._mesosphere_list.empty():
            self._flush_pending()

        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            self.COMMIT_FLAG = True
            return response.json()
        else:
            raise_error_response(response)

    def close(self):
        """Flush and close the collection handle from an API perspective."""
        uri = f'{self._uri}/close_collection'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}

        if not self._mesosphere_list.empty():
            self._flush_pending()

        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            self.COMMIT_FLAG = True
            return response.json()
        else:
            raise_error_response(response)

    def snapshot_to(self, snapshot_path: Union[str, Path]):
        """Create a filesystem snapshot of this collection on the server."""
        uri = f'{self._uri}/snapshot_collection'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "snapshot_path": str(snapshot_path),
        }

        if not self._mesosphere_list.empty():
            self._flush_pending()

        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def export_to(self, export_path: Union[str, Path]):
        """Export this collection as JSONL metadata plus binary vectors on the server."""
        uri = f'{self._uri}/export_collection'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "export_path": str(export_path),
        }

        if not self._mesosphere_list.empty():
            self._flush_pending()

        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def is_id_exists(self, id: Union[str, int]):
        """
        Check if an ID exists in the collection.

        Parameters:
            id (str | int): The public ID to check.

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
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        if field_name == 'default':
            uri = f'{self._uri}/build_index'
            data = {
                "database_name": self._database_name,
                "collection_name": self._collection_name,
                "index_mode": index_mode,
            }
        else:
            uri = f'{self._uri}/build_vector_field_index'
            data = {
                "database_name": self._database_name,
                "collection_name": self._collection_name,
                "field_name": field_name,
                "index_mode": index_mode,
            }
        if n_clusters is not None and index_mode.upper().startswith("IVF"):
            data['n_clusters'] = int(n_clusters)

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            if field_name == "default":
                self._default_index_built = True
            return response.json()
        else:
            raise_error_response(response)

    def remove_index(self, field_name: str = 'default'):
        """
        Remove the index of the collection.

        Parameters:
            field_name (str): Named vector field to remove index for.
                Defaults to "default" (the primary collection index).

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        if field_name == 'default':
            uri = f'{self._uri}/remove_index'
            data = {
                "database_name": self._database_name,
                "collection_name": self._collection_name,
            }
        else:
            uri = f'{self._uri}/remove_vector_field_index'
            data = {
                "database_name": self._database_name,
                "collection_name": self._collection_name,
                "field_name": field_name,
            }
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            if field_name == "default":
                self._default_index = None
                self._default_index_built = False
            return response.json()
        else:
            raise_error_response(response)

    def create_vector_field(
            self,
            name: str,
            dim: int,
            metric: str = "ip",
            index_mode: Union[str, None] = None,
            dtypes: Union[str, None] = None,
    ):
        """Create a named vector field with its own dimension and metric."""
        uri = f'{self._uri}/create_vector_field'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "field_name": name,
            "dim": int(dim),
            "metric": metric,
            "index_mode": index_mode,
            "dtypes": dtypes,
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()
        raise_error_response(response)

    def list_vector_fields(self):
        """List vector fields, including the reserved default primary vector."""
        uri = f'{self._uri}/list_vector_fields'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()['params']['fields']
        raise_error_response(response)

    def add_named_vectors(
            self,
            field_name: str,
            vectors: Union[list, np.ndarray],
            ids: List[int],
    ):
        """Attach vectors to a named vector field for existing IDs."""
        uri = f'{self._uri}/add_named_vectors'
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "field_name": field_name,
            "vectors": vecs.tolist(),
            "ids": normalize_external_ids(ids)[0],
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            self.COMMIT_FLAG = False
            return response.json()
        raise_error_response(response)

    def add_sparse_vectors(
            self,
            vectors: List[Union[Dict[int, float], List[Tuple[int, float]]]],
            ids: List[int],
    ):
        """Attach sparse feature vectors to existing IDs."""
        uri = f'{self._uri}/add_sparse_vectors'
        payloads = [_sparse_vector_payload(vector)[0] for vector in vectors]
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "vectors": payloads,
            "ids": normalize_external_ids(ids)[0],
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            self.COMMIT_FLAG = False
            return response.json()
        raise_error_response(response)

    def insert_session(self):
        """
        Start an insert session.
        """
        from ...execution_layer.session import DataInsertionSession

        return DataInsertionSession(self)

    def _search(
            self,
            vector,
            k,
            where,
            return_fields=False,
            vector_field: str = "default",
            nprobe: int = 10,
            approx: bool = False,
            eps: float = 1e-4,
            wire_dtype: str = "float32",
    ):
        """
        Search the collection using compact binary protocol.

        where is a SQL/WHERE expression string passed directly to ApexBase.
        """
        raise_if(ValueError, not isinstance(where, (type(None), str)),
                 'where must be a SQL/WHERE expression string or None.')

        vec = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        dim = len(vec)
        vector_encoding = _normalize_vector_encoding(wire_dtype)
        if vector_encoding == "float16":
            body = np.ascontiguousarray(vec, dtype="<f2").tobytes()
        else:
            body = vec.tobytes()

        params = {
            'database_name': self._database_name,
            'collection_name': self._collection_name,
            'dim': dim,
            'k': k,
            'return_fields': str(return_fields).lower(),
            'vector_encoding': vector_encoding,
        }
        if where is not None:
            params['where'] = where
        if vector_field is not None and vector_field != "default":
            params['vector_field'] = vector_field
        if nprobe is not None:
            params['nprobe'] = nprobe
        if approx is not None:
            params['approx'] = str(bool(approx)).lower()
        if eps is not None:
            params['eps'] = float(eps)

        uri = f'{self._uri}/search_binary'
        response = self._session.post(
            uri, params=params, content=body,
            headers={'Content-Type': 'application/octet-stream'},
        )

        if response.status_code == 200:
            return response.content  # raw binary
        else:
            raise_error_response(response)

    def search(
            self, vector: Union[list[float], np.ndarray, None] = None, k: int = 10, *,
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
        Search the database for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray or list): The search vectors, it can be a single vector or a list of vectors.
                The vectors must have the same dimension as the vectors in the database,
                and the type of vector can be a list or a numpy array.
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

        Returns:
            ResultView: Search results with ids, distances, and optional fields.

        Raises:
            ValueError: If the collection has been deleted or does not exist.
            ExecutionError: If the server returns an error.
        """
        del wire_dtype  # JSON search is used so public string IDs can round-trip.
        if (vector is None) == (document is None):
            raise ValueError("search() requires exactly one of vector or document")
        send_return_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        if document is not None:
            vec = embed_documents([document])[0]
        else:
            vec = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        uri = f'{self._uri}/search'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "vector": vec.tolist(),
            "vector_field": vector_field,
            "k": k,
            "where": where,
            "return_fields": send_return_fields,
            "nprobe": nprobe,
            "approx": approx,
            "eps": float(eps),
        }
        response = self._session.post(uri, json=data)
        if response.status_code != 200:
            raise_error_response(response)
        items = response.json()['params']['items']
        ids = id_array(items.get('ids', []))
        dists = np.array(items.get('scores', []), dtype=np.float32)
        fields = items.get('fields', [])
        ids, dists, reranked_fields = apply_external_rerank(
            ids=ids,
            scores=dists,
            fields=fields,
            reranker=reranker,
            query={
                "type": "document_search" if document is not None else "vector_search",
                "document": document,
                "vector_field": vector_field,
                "vector": vec.tolist(),
                "where": where,
                "nprobe": nprobe,
                "approx": approx,
                "eps": float(eps),
            },
            rerank_k=rerank_k,
        )
        if vector_field == "default":
            index_mode = self.index_mode
        else:
            vector_fields = self.list_vector_fields()
            index_mode = next(
                (
                    field.get("index_mode")
                    for field in vector_fields
                    if field.get("name") == vector_field
                ),
                "FLAT",
            )
        idx_type, metric = _parse_index_mode(index_mode)
        return ResultView(
            ids=ids,
            distances=dists,
            fields=reranked_fields if return_fields else [],
            k=len(ids),
            distance=metric,
            index=idx_type,
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
        payload, normalized = _sparse_vector_payload(vector)
        uri = f'{self._uri}/sparse_search'
        send_return_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "vector": payload,
            "k": k,
            "where": where,
            "return_fields": send_return_fields,
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            items = response.json()['params']['items']
            ids = id_array(items.get('ids', []))
            scores = np.array(items.get('scores', []), dtype=np.float32)
            fields = items.get('fields', [])
            ids, scores, reranked_fields = apply_external_rerank(
                ids=ids,
                scores=scores,
                fields=fields,
                reranker=reranker,
                query={
                    "type": "sparse_search",
                    "vector": normalized,
                    "where": where,
                },
                rerank_k=rerank_k,
            )
            return ResultView(
                ids=ids,
                distances=scores,
                fields=reranked_fields if return_fields else [],
                k=len(ids),
                distance="IP",
                index=items.get('index', 'SPARSE-FLAT-IP'),
                result_type="search",
            )
        raise_error_response(response)

    def search_profile(
            self, vector: Union[list[float], np.ndarray], k: int = 10, *,
            where: Union[str, None] = None, nprobe: int = 10
    ):
        """Search and return profile/explain metadata."""
        vec = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        uri = f'{self._uri}/search_profile'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "vector": vec.tolist(),
            "k": k,
            "where": where,
            "nprobe": nprobe,
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()['params']
        raise_error_response(response)

    def bm25_search(
            self, text: str, k: int = 10, *,
            text_fields: Union[list[str], None] = None,
            where: Union[str, None] = None,
            return_fields: bool = False,
            reranker: Optional[Callable[[Dict[str, Any]], Any]] = None,
            rerank_k: Optional[int] = None,
            rerank_with_fields: bool = True,
        ):
        """BM25 text search over metadata fields."""
        uri = f'{self._uri}/bm25_search'
        send_return_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "text": text,
            "text_fields": text_fields,
            "k": k,
            "where": where,
            "return_fields": send_return_fields,
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            items = response.json()['params']['items']
            ids = id_array(items.get('ids', []))
            scores = np.array(items.get('scores', []), dtype=np.float32)
            fields = items.get('fields', [])
            ids, scores, reranked_fields = apply_external_rerank(
                ids=ids,
                scores=scores,
                fields=fields,
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
                distances=scores,
                fields=reranked_fields if return_fields else [],
                k=len(ids),
                distance="bm25",
                index=items.get('index', 'BM25-SCAN'),
                result_type="search",
            )
        raise_error_response(response)

    def hybrid_search(
            self, vector: Union[list[float], np.ndarray, None] = None,
            text: Union[str, None] = None, k: int = 10, *,
            where: Union[str, None] = None,
            text_fields: Union[list[str], None] = None,
            fusion: str = "rrf",
            vector_weight: float = 1.0,
            text_weight: float = 1.0,
            rrf_k: float = 60.0,
            candidate_limit: Union[int, None] = None,
            nprobe: int = 10,
            return_fields: bool = False,
            reranker: Optional[Callable[[Dict[str, Any]], Any]] = None,
            rerank_k: Optional[int] = None,
            rerank_with_fields: bool = True,
    ):
        """Hybrid vector + BM25 text search with RRF or weighted fusion."""
        vec = None if vector is None else np.ascontiguousarray(vector, dtype=np.float32).ravel().tolist()
        uri = f'{self._uri}/hybrid_search'
        send_return_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "vector": vec,
            "text": text,
            "text_fields": text_fields,
            "k": k,
            "where": where,
            "fusion": fusion,
            "vector_weight": vector_weight,
            "text_weight": text_weight,
            "rrf_k": rrf_k,
            "candidate_limit": candidate_limit,
            "return_fields": send_return_fields,
            "nprobe": nprobe,
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            items = response.json()['params']['items']
            ids = id_array(items.get('ids', []))
            scores = np.array(items.get('scores', []), dtype=np.float32)
            fields = items.get('fields', [])
            ids, scores, reranked_fields = apply_external_rerank(
                ids=ids,
                scores=scores,
                fields=fields,
                reranker=reranker,
                query={
                    "type": "hybrid_search",
                    "vector": vec,
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
                distances=scores,
                fields=reranked_fields if return_fields else [],
                k=len(ids),
                distance="fusion",
                index=items.get('index', 'HYBRID-RRF'),
                result_type="search",
            )
        raise_error_response(response)

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
            reranker (callable, optional): External rerank hook, applied per-query.
            rerank_k (int, optional): Keep top-N after rerank per query.
            rerank_with_fields (bool): Fetch candidate fields for reranker payload
                even when ``return_fields=False``.

        Returns:
            List[ResultView]: List of ResultView objects, one per query vector.
        """
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        del wire_dtype
        send_return_fields = should_fetch_fields(
            return_fields=return_fields,
            reranker=reranker,
            rerank_with_fields=rerank_with_fields,
        )
        uri = f'{self._uri}/batch_search'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "vectors": vecs.tolist(),
            "k": k,
            "where": where,
            "return_fields": send_return_fields,
            "nprobe": nprobe,
        }
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            idx_type, metric = _parse_index_mode(self.index_mode)
            output = []
            for query_index, item in enumerate(response.json()['params']['results']):
                ids = id_array(item.get('ids', []))
                dists = np.array(item.get('scores', []), dtype=np.float32)
                fields = item.get('fields', [])
                ids, dists, reranked_fields = apply_external_rerank(
                    ids=ids,
                    scores=dists,
                    fields=fields,
                    reranker=reranker,
                    query={
                        "type": "batch_vector_search",
                        "vector": vecs[query_index].tolist(),
                        "where": where,
                        "nprobe": nprobe,
                        "query_index": query_index,
                    },
                    rerank_k=rerank_k,
                )
                output.append(ResultView(
                    ids=ids,
                    distances=dists,
                    fields=reranked_fields if return_fields else [],
                    k=len(ids),
                    distance=metric,
                    index=idx_type,
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
        Get the first n items in the collection.

        Parameters:
            n (int): The number of items to return. Default is 5.

        Returns:
            ResultView: Data result with vectors, ids, and fields.

        Raises:
            ExecutionError: If the server returns an error.
        """
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "n": n,
        }
        response = self._session.post(f'{self._uri}/head', json=data)

        if response.status_code == 200:
            head = response.json()['params']['head']
            vecs = np.asarray(head[0], dtype=np.float32)
            ids = id_array(head[1])
            fields = head[2] if len(head) > 2 else []
            return ResultView(
                vectors=vecs, ids=ids, fields=fields,
                result_type="data",
            )
        else:
            raise_error_response(response)

    def tail(self, n: int = 5):
        """
        Get the last n items in the collection.

        Parameters:
            n (int): The number of items to return. Default is 5.

        Returns:
            ResultView: Data result with vectors, ids, and fields.

        Raises:
            ExecutionError: If the server returns an error.
        """
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "n": n,
        }
        response = self._session.post(f'{self._uri}/tail', json=data)

        if response.status_code == 200:
            tail = response.json()['params']['tail']
            vecs = np.asarray(tail[0], dtype=np.float32)
            ids = id_array(tail[1])
            fields = tail[2] if len(tail) > 2 else []
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
            ids = id_array(item[1])
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

        if where is None and filter_ids is None:
                return ResultView(ids=id_array([]), result_type="query")

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
                ids_arr = id_array(result) if result else id_array([])
                return ResultView(ids=ids_arr, result_type="query")
            else:
                # result is a list of dicts
                records = result if isinstance(result, list) else []
                ids_list = [r.get('id', i) for i, r in enumerate(records)] if records else []
                ids_arr = id_array(ids_list) if ids_list else id_array([])
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

        if where is None and filter_ids is None:
            dim = int(self._init_params.get('dim') or 0)
            return ResultView(
                vectors=np.empty((0, dim), dtype=np.float32),
                ids=id_array([]),
                fields=[],
                result_type="data",
            )

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
            ids = id_array(result[1])
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

    def delete(self, ids):
        """
        Soft-delete vectors by ID.

        Deleted IDs are excluded from all future search results.

        Parameters:
            ids (list[int]): External IDs to soft-delete.
        """
        uri = f'{self._uri}/delete'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "ids": normalize_external_ids(ids)[0],
        }
        response = self._session.post(uri, json=data)
        if response.status_code != 200:
            raise_error_response(response)
        self.COMMIT_FLAG = False

    def restore(self, ids):
        """
        Restore previously soft-deleted vectors.

        Parameters:
            ids (list[int]): External IDs to restore.
        """
        uri = f'{self._uri}/restore'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "ids": normalize_external_ids(ids)[0],
        }
        response = self._session.post(uri, json=data)
        if response.status_code != 200:
            raise_error_response(response)
        self.COMMIT_FLAG = False

    def list_deleted_ids(self):
        """
        Return the sorted list of all currently soft-deleted IDs.

        Returns:
            list[int]: Sorted list of soft-deleted external IDs.
        """
        uri = f'{self._uri}/list_deleted_ids'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()['params']['ids']
        else:
            raise_error_response(response)

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
            tuple: (ids, distances) as numpy arrays.
        """
        import numpy as np
        uri = f'{self._uri}/search_range'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "vector": np.asarray(vector, dtype=np.float32).tolist(),
            "threshold": float(threshold),
            "max_results": int(max_results),
        }
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            result = response.json()['params']['result']
            ids = id_array(result['ids'])
            dists = np.array(result['distances'], dtype=np.float32)
            from ...result_view import ResultView
            return ResultView(
                ids=ids, distances=dists,
                k=len(ids),
                result_type="search",
            )
        else:
            raise_error_response(response)

    def compact(self) -> int:
        """Physically remove all tombstoned vectors and rebuild storage.

        Returns:
            int: Number of vectors physically removed.
        """
        uri = f'{self._uri}/compact'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()['params']['vectors_removed']
        else:
            raise_error_response(response)

    def stats(self) -> dict:
        """Return collection statistics.

        Returns:
            dict: n_vectors, n_live, n_tombstoned, dimension, index_mode, max_id.
        """
        uri = f'{self._uri}/stats'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)
        if response.status_code == 200:
            return response.json()['params']['stats']
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
