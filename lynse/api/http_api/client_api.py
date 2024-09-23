import queue
import time
from datetime import datetime
from typing import Union, List, Tuple

import msgpack
import numpy as np
import httpx
from spinesUtils.asserts import raise_if
from tqdm import trange

from ...core_components.fields_cache import IndexSchema, ExpressionParser
from ...core_components.fields_cache.filter import Filter
from ...api import logger
from ...core_components.locks import ThreadLock
from ...utils.poster import Poster
from ...utils.utils import collection_repr


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


def pack_data(data):
    """
    Pack the data.

    Parameters:
        data: The data to pack.

    Returns:
        bytes: The packed data.
    """
    packed_data = msgpack.packb(data, use_bin_type=True)
    return packed_data


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
            chunk_size: int = 100_000,
            dtypes: str = 'float32',
            use_cache: bool = True,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            drop_if_exists: bool = False,
            description: str = None,
            cache_chunks: int = 20
    ):
        """
        Create a collection.

        Parameters:
            collection (str): The name of the collection.
            dim (int): The dimension of the vectors. Default is None.
                When creating a new collection, the dimension of the vectors must be specified.
                When loading an existing collection, the dimension of the vectors is automatically loaded.
            chunk_size (int): The chunk size. Default is 100,000.
            dtypes (str): The data types. Default is 'float32'.
            use_cache (bool): Whether to use cache for search. Default is True.
            n_threads (int): The number of threads. Default is 10.
            warm_up (bool): Whether to warm up. Default is False.
            drop_if_exists (bool): Whether to drop the collection if it exists. Default is False.
            description (str): A description of the collection. Default is None.
                The description is limited to 500 characters.
            cache_chunks (int): The number of chunks to cache in memory. Default is 20.

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
            "chunk_size": chunk_size,
            "dtypes": dtypes,
            "use_cache": use_cache,
            "n_threads": n_threads,
            "warm_up": warm_up,
            "drop_if_exists": drop_if_exists,
            "description": description,
            "cache_chunks": cache_chunks
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

    def get_collection(self, collection: str, cache_chunks=20, warm_up=True):
        """
        Get a collection.

        Parameters:
            collection (str): The name of the collection.
            cache_chunks (int): The number of chunks to cache. Default is 20.
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
            params.update({'cache_chunks': cache_chunks, 'warm_up': warm_up})

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
                - dtypes (str): The data types.
                - use_cache (bool): Whether to use cache.
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
        self._lock = ThreadLock()

    def _get_commit_msg(self):
        """
        Get the commit message.

        Returns:
            str: The last commit time.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/get_commit_msg'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}

        response = self._session.post(uri, json=data)

        rj = response.json()
        if response.status_code == 200:
            if (rj['params']['commit_msg'] is None or
                    rj['params']['commit_msg'] == 'No commit message found for this collection'):
                return None
            return rj['params']['commit_msg']['last_commit_time']
        else:
            raise ExecutionError(rj)

    def _update_commit_msg(self, last_commit_time):
        """
        Update the commit message.

        Parameters:
            last_commit_time (str): The last commit time.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/update_commit_msg'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "last_commit_time": last_commit_time,
        }

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            try:
                rj = response.json()
                raise ExecutionError(rj)
            except Exception as e:
                print(e)
                raise ExecutionError(response.text)

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

    def add_item(self, vector: Union[list[float], np.ndarray], id: int, *, field: Union[dict, None] = None,
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
            headers = {
                'Content-Type': 'application/msgpack'
            }

            data = {
                "database_name": self._database_name,
                "collection_name": self._collection_name,
                "item": {
                    "vector": vector if isinstance(vector, list) else vector.tolist(),
                    "id": id,
                    "field": field if field is not None else {},
                },
            }

            response = self._session.post(uri, data=None, content=pack_data(data), headers=headers)

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
                headers = {
                    'Content-Type': 'application/msgpack'
                }

                data = {
                    "database_name": self._database_name,
                    "collection_name": self._collection_name,
                    "items": mesosphere_list,
                }

                response = self._session.post(uri, data=None, content=pack_data(data), headers=headers)

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

        headers = {
            'Content-Type': 'application/msgpack',
        }

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

            response = self._session.post(uri, content=pack_data(data), headers=headers)

            if response.status_code == 200:
                self.COMMIT_FLAG = False
                ids.extend(response.json()['params']['ids'])
            else:
                raise_error_response(response)

        return ids

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

        response = self._session.post(uri, content=pack_data(data), headers={'Content-Type': 'application/msgpack'})

        if response.status_code == 202:
            task_id = response.json().get('task_id')
            status_uri = f'{self._uri}/status/{task_id}'

            while True:
                status_response = self._session.get(status_uri)
                status_data = status_response.json()

                if status_response.status_code == 200:
                    logger.info(f'Task status: {status_data}', rewrite_print=True)
                    if status_data['status'] in ['Success', 'Error']:
                        if status_data['status'] == 'Success':
                            self._update_commit_msg(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        else:
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

    def build_index(self, index_mode: str = 'IVF-FLAT', n_clusters: int = 16):
        """
        Build the index of the collection.

        Parameters:
            index_mode (str): The index mode. Default is 'IVF-FLAT'.
            n_clusters (int): The number of clusters. Default is 16.

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
            "n_clusters": n_clusters
        }

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
        from ...execution_layer.session import DataOpsSession

        return DataOpsSession(self)

    def _search(self, vector, k, search_filter, return_fields=False, **kwargs):
        """
        Search the collection.
        """
        uri = f'{self._uri}/search'

        raise_if(ValueError, not isinstance(search_filter, (Filter, type(None), str)),
                 'search_filter must be Filter or None or FieldExpression string.')

        if isinstance(search_filter, str):
            search_filter = ExpressionParser(search_filter).to_filter()

        if search_filter is not None:
            search_filter = search_filter.to_dict()

        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "vector": vector if isinstance(vector, list) else vector.tolist(),
            "k": k,
            "search_filter": search_filter,
            "return_fields": return_fields,
            **kwargs
        }
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def search(
            self, vector: Union[list[float], np.ndarray], k: int = 10, *,
            search_filter: Union[Filter, None] = None,
            return_fields: bool = False, **kwargs
    ):
        """
        Search the database for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray or list): The search vectors, it can be a single vector or a list of vectors.
                The vectors must have the same dimension as the vectors in the database,
                and the type of vector can be a list or a numpy array.
            k (int): The number of nearest vectors to return.
            search_filter (Filter or FilterExpression string, optional): The filter to apply to the search.
            return_fields (bool): Whether to return the fields of the search results.
            kwargs: Additional keyword arguments. The following are valid:
                rescore (bool): Whether to rescore the results of binary or scaler quantization searches.
                    Default is False. It is recommended to set it to True when the index mode is 'Binary'.
                rescore_multiplier (int): The multiplier for the rescore operation.
                    It is only available when rescore is True.
                    If 'Binary' is in the index mode, the default is 10. Otherwise, the default is 2.

        Returns:
            Tuple: If return_fields is True, the indices, similarity scores,
                    and fields of the nearest vectors in the database.
                Otherwise, the indices and similarity scores of the nearest vectors in the database.

        Raises:
            ValueError: If the collection has been deleted or does not exist.
            ExecutionError: If the server returns an error.
        """
        rjson = self._search(vector=vector, k=k, search_filter=search_filter, return_fields=return_fields, **kwargs)

        ids, scores = (np.array(rjson['params']['items']['ids']),
                       np.array(rjson['params']['items']['scores']))
        fields = rjson['params']['items']['fields']

        return ids, scores, fields

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
            Tuple: The vectors, IDs, and fields of the items.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/head'
        data = {"database_name": self._database_name, "collection_name": self._collection_name, "n": n}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            head = response.json()['params']['head']
            return np.asarray(head[0]), np.asarray(head[1]), head[2]
        else:
            raise_error_response(response)

    def tail(self, n: int = 5):
        """
        Get the last n items in the collection.

        Parameters:
            n (int): The number of items to return. Default is 5.

        Returns:
            Tuple: The vectors, IDs, and fields of the items.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/tail'
        data = {"database_name": self._database_name, "collection_name": self._collection_name, "n": n}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            tail = response.json()['params']['tail']
            return np.asarray(tail[0]), np.asarray(tail[1]), tail[2]
        else:
            raise_error_response(response)

    def read_by_only_id(self, id: Union[int, list]):
        """
        Read the item by ID.

        Parameters:
            id (int, list): The ID of the item or a list of IDs.

        Returns:
            Tuple: The vectors, IDs, and fields of the items.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/read_by_only_id'
        data = {"database_name": self._database_name, "collection_name": self._collection_name, "id": id}
        response = self._session.post(uri, json=data)

        item = response.json()['params']['item']

        if response.status_code == 200:
            return item[0], item[1], item[2]
        else:
            raise_error_response(response)

    def query(self, query_filter, filter_ids=None, return_ids_only=False):
        """
        Query the collection.

        Parameters:
            query_filter (Filter or dict): The filter object.
            filter_ids (list[int]): The list of IDs to filter.
            return_ids_only (bool): Whether to return the IDs only.

        Returns:
            List[dict]: The records. If not return_ids_only, the records will be returned.
            List[int]: The external IDs. If return_ids_only, the external IDs will be returned.

        Raises:
            ExecutionError: If the server returns an error.
        """

        uri = f'{self._uri}/query'

        raise_if(ValueError, not isinstance(query_filter, (Filter, str, type(None), dict)),
                 'query_filter must be Filter or dict or FieldExpression string or None.')

        if isinstance(query_filter, str):
            query_filter = ExpressionParser(query_filter).to_filter()

        if query_filter is not None and isinstance(query_filter, Filter):
            query_filter = query_filter.to_dict()

        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "query_filter": query_filter,
            "filter_ids": filter_ids,
            "return_ids_only": return_ids_only
        }

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()['params']['result']
        else:
            raise_error_response(response)

    def query_vectors(self, query_filter, filter_ids=None):
        """
        Query the vector data by the filter.

        Parameters:
            query_filter (Filter or dict or FieldExpression str or None):
                The filter object or the specify data to filter.
            filter_ids (list[int]):
                The list of external IDs to filter. Default is None.

        Returns:
            (Tuple[List[np.ndarray], List[int], List[Dict]]): The vectors, IDs, and fields of the items.
        """
        uri = f'{self._uri}/query_vectors'

        raise_if(ValueError, not isinstance(query_filter, (Filter, str, type(None), dict)),
                 'query_filter must be Filter or dict or FieldExpression string or None.')

        if isinstance(query_filter, str):
            query_filter = ExpressionParser(query_filter).to_filter()

        if query_filter is not None and isinstance(query_filter, Filter):
            query_filter = query_filter.to_dict()

        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "query_filter": query_filter,
            "filter_ids": filter_ids,
        }

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            result = response.json()['params']['result']
            return np.asarray(result[0]), np.array(result[1]), result[2]
        else:
            raise_error_response(response)

    def build_field_index(self, schema, rebuild_if_exists=False):
        """
        Build the field index of the collection.

        Parameters:
            schema (IndexSchema): The schema of the field index.
            rebuild_if_exists (bool): Whether to rebuild the field index if it exists.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        if not isinstance(schema, IndexSchema):
            raise TypeError("schema must be an instance of IndexSchema.")

        uri = f'{self._uri}/build_field_index'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "schema": schema.to_dict(),
            "rebuild_if_exists": rebuild_if_exists
        }

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def list_field_index(self):
        """
        List the field index of the collection.

        Returns:
            dict: The field index of the collection.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/list_field_index'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()['params']['field_indices']
        else:
            raise_error_response(response)

    def remove_field_index(self, field_name):
        """
        Remove the field index of the collection.

        Parameters:
            field_name (str): The name of the field.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/remove_field_index'
        data = {
            "database_name": self._database_name,
            "collection_name": self._collection_name,
            "field_name": field_name
        }

        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise_error_response(response)

    def remove_all_field_indices(self):
        """
        Remove all the field indices of the collection.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        uri = f'{self._uri}/remove_all_field_indices'
        data = {"database_name": self._database_name, "collection_name": self._collection_name}
        response = self._session.post(uri, json=data)

        if response.status_code == 200:
            return response.json()
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
