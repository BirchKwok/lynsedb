import time
from datetime import datetime
from typing import Union, List, Tuple

import numpy as np
import requests
from spinesUtils.asserts import raise_if

from min_vec.structures.filter import Filter
from min_vec.api import config
from min_vec.utils.utils import QueryVectorCache


class ExecutionError(Exception):
    pass


class Collection:
    def __init__(self, url, collection_name, **params):
        """
        Initialize the collection.
            .. versionadded:: 0.3.2

        Parameters:
            url (str): The URL of the server.
            collection_name (str): The name of the collection.
            **params: The collection parameters.
                - dim (int): The dimension of the vectors.
                - n_clusters (int): The number of clusters.
                - chunk_size (int): The chunk size.
                - distance (str): The distance metric.
                - index_mode (str): The index mode.
                - dtypes (str): The data types.
                - use_cache (bool): Whether to use cache.
                - scaler_bits (int): The scaler bits.
                - n_threads (int): The number of threads.
                - warm_up (bool): Whether to warm up.
                - drop_if_exists (bool): Whether to drop the collection if it exists.
        """
        self.IS_DELETED = False
        self._url = url
        self._collection_name = collection_name
        self._init_params = params

        self.most_recent_query_report = {}
        self.query_report = {}

        self.COMMIT_FLAG = False

    def _get_commit_msg(self):
        """
        Get the commit message.

        Returns:
            str: The last commit time.

        Raises:
            ExecutionError: If the server returns an error.
        """
        url = f'{self._url}/get_commit_msg'
        data = {"collection_name": self._collection_name}

        response = requests.post(url, json=data)
        rj = response.json()
        if response.status_code == 200:
            if rj['params']['commit_msg'] is None:
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
        url = f'{self._url}/update_commit_msg'
        data = {
            "collection_name": self._collection_name,
            "last_commit_time": last_commit_time,
        }

        response = requests.post(url, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise ExecutionError(response.json())

    def _if_exists(self):
        """
        Check if the collection exists.

        Returns:
            bool: Whether the collection exists.

        Raises:
            ExecutionError: If the server returns an error.
        """
        url = f'{self._url}/is_collection_exists'
        data = {"collection_name": self._collection_name}

        response = requests.post(url, json=data)

        if response.status_code == 200:
            return response.json()['params']['exists']
        else:
            raise ExecutionError(response.json())

    def add_item(self, vector: Union[list[float], np.ndarray], id: int, field: dict):
        """
        Add an item to the collection.
            .. versionadded:: 0.3.2

        Parameters:
            vector (list[float], np.ndarray): The vector of the item.
            id (int): The ID of the item.
            field (dict): The fields of the item.

        Returns:
            int: The ID of the item.

        Raises:
            ValueError: If the collection has been deleted or does not exist.
            ExecutionError: If the server returns an error.
        """
        raise_if(ValueError, self.IS_DELETED or not self._if_exists(),
                 'The collection has been deleted or does not exist.')

        url = f'{self._url}/add_item'
        data = {
            "collection_name": self._collection_name,
            "item": {
                "vector": vector if isinstance(vector, list) else vector.tolist(),
                "id": id,
                "field": field
            }
        }
        response = requests.post(url, json=data)

        if response.status_code == 200:
            self.COMMIT_FLAG = False
            return response.json()['params']['item']['id']
        else:
            raise ExecutionError(response.json())

    def bulk_add_items(
            self,
            vectors: List[Union[
                Tuple[Union[List, Tuple, np.ndarray], int, dict],
                Tuple[Union[List, Tuple, np.ndarray], int],
                Tuple[Union[List, Tuple, np.ndarray]]
            ]]
    ):
        """
        Add multiple items to the collection.
            .. versionadded:: 0.3.2

        Parameters:
            vectors (List[Tuple[Union[List, Tuple, np.ndarray], int, dict]],
            List[Tuple[Union[List, Tuple, np.ndarray], int]] , List[Tuple[Union[List, Tuple, np.ndarray]]]):
                The list of items to add. Each item is a tuple containing the vector, ID, and fields.

        Returns:
            dict: The response from the server.

        Raises:
            ValueError: If the collection has been deleted or does not exist.
            TypeError: If the vectors are not in the correct format.
            ExecutionError: If the server returns an error.
        """
        raise_if(ValueError, self.IS_DELETED or not self._if_exists(),
                 'The collection has been deleted or does not exist.')

        items = []
        for vector in vectors:
            raise_if(TypeError, not isinstance(vector, tuple), 'Each item must be a tuple of vector, '
                                                               'ID(optional), and fields(optional).')

            if len(vector) == 3:
                items.append({
                    "vector": vector[0].tolist() if isinstance(vector[0], np.ndarray) else vector[0],
                    "id": vector[1],
                    "field": vector[2]
                })
            elif len(vector) == 2:
                items.append({
                    "vector": vector[0].tolist() if isinstance(vector[0], np.ndarray) else vector[0],
                    "id": vector[1],
                    "field": {}
                })
            else:
                items.append({
                    "vector": vector[0].tolist() if isinstance(vector[0], np.ndarray) else vector[0],
                    "id": None,
                    "field": {}
                })

        url = f'{self._url}/bulk_add_items'
        data = {
            "collection_name": self._collection_name,
            "items": items
        }

        response = requests.post(url, json=data)

        if response.status_code == 200:
            self.COMMIT_FLAG = False
            return response.json()['params']['ids']
        else:
            raise ExecutionError(response.json())

    def commit(self):
        """
        Commit the changes in the collection.
            .. versionadded:: 0.3.2

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        url = f'{self._url}/commit'
        data = {"collection_name": self._collection_name}
        response = requests.post(url, json=data)

        if response.status_code == 200:
            self._update_commit_msg(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            return response.json()
        else:
            raise ExecutionError(response.json())

    def insert_session(self):
        """
        Start an insert session.
            .. versionadded:: 0.3.2
        """
        from min_vec.execution_layer.session import DatabaseSession

        return DatabaseSession(self)

    @QueryVectorCache(config.MVDB_QUERY_CACHE_SIZE)
    def _query(self, vector: Union[list[float], np.ndarray], k: int = 10, distance: str = 'cosine',
               query_filter: Union[Filter, None] = None, return_similarity: bool = True, **kwargs):
        """
        Query the collection.
            .. versionadded:: 0.3.2

        Parameters:
            vector (list[float] or np.ndarray): The query vector.
            k (int): The number of results to return. Default is 10.
            distance (str): The distance metric. Default is 'cosine', it can be 'cosine' or 'L2'.
            query_filter (Filter, optional): The field filter to apply to the query, must be a Filter object.
            return_similarity (bool): Whether to return the similarity. Default is True.

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the collection has been deleted or does not exist.
            ExecutionError: If the server returns an error.
        """
        raise_if(ValueError, self.IS_DELETED or not self._if_exists(),
                 'The collection has been deleted or does not exist.')

        url = f'{self._url}/query'

        if query_filter is not None:
            raise_if(TypeError, not isinstance(query_filter, Filter), 'The query filter must be a Filter object.')
            query_filter = query_filter.to_dict()

        data = {
            "collection_name": self._collection_name,
            "vector": vector if isinstance(vector, list) else vector.tolist(),
            "k": k,
            'distance': distance,
            "query_filter": query_filter,
            "return_similarity": return_similarity
        }
        response = requests.post(url, json=data)

        if response.status_code == 200:
            rjson = response.json()
            self.most_recent_query_report['Collection Shape'] = self.shape
            self.most_recent_query_report['Query Time'] = rjson['params']['items']['query time']
            self.most_recent_query_report['Query Distance'] = rjson['params']['items']['distance']
            self.most_recent_query_report['Query K'] = k

            ids, scores = np.array(rjson['params']['items']['ids']), np.array(rjson['params']['items']['scores'])

            if ids is not None:
                self.most_recent_query_report[f'Top {k} Results ID'] = ids
                if return_similarity:
                    self.most_recent_query_report[f'Top {k} Results Similarity'] = scores

            return ids, scores
        else:
            raise ExecutionError(response.json())

    def query(
            self, vector: Union[list[float], np.ndarray], k: int = 10, distance: str = 'cosine',
            query_filter: Union[Filter, None] = None, return_similarity: bool = True
    ):
        """
        Query the collection.
            .. versionadded:: 0.3.2

        Parameters:
            vector (list[float] or np.ndarray): The query vector.
            k (int): The number of results to return. Default is 10.
            distance (str): The distance metric. Default is 'cosine'. It can be 'cosine' or 'L2'.
            query_filter (Filter, optional): The field filter to apply to the query, must be a Filter object.
            return_similarity (bool): Whether to return the similarity. Default is True.

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the collection has been deleted or does not exist.
            ExecutionError: If the server returns an error.
        """
        tik = time.time()
        if self._init_params['use_cache']:
            res = self._query(vector, k, distance, query_filter, return_similarity)
        else:
            res = self._query(vector, k, distance, query_filter, return_similarity=return_similarity, now=time.time())

        tok = time.time()

        self.most_recent_query_report['Query Time'] = f"{tok - tik :>.5f} s"

        return res

    @property
    def shape(self):
        """
        Get the shape of the collection.
            .. versionadded:: 0.3.2

        Returns:
            Tuple: The shape of the collection.

        Raises:
            ExecutionError: If the server returns an error.
        """
        url = f'{self._url}/collection_shape'
        data = {"collection_name": self._collection_name}
        response = requests.post(url, json=data)

        if response.status_code == 200:
            return tuple(response.json()['params']['shape'])
        else:
            rj = response.json()
            if 'error' in rj and rj['error'] == f"Collection '{self._collection_name}' does not exist.":
                return 0, self._init_params['dim']
            else:
                raise ExecutionError(response.json())

    @property
    def query_report_(self):
        """
        Get the query report of the collection.
            .. versionadded:: 0.3.2

        Returns:
            str: The query report.
        """
        report = '\n* - MOST RECENT QUERY REPORT -\n'
        for key, value in self.most_recent_query_report.items():
            if key == "Collection Shape":
                value = self.shape

            report += f'| - {key}: {value}\n'

        report += '* - END OF REPORT -\n'

        return report

    def __repr__(self):
        if self.status_report_['COLLECTION STATUS REPORT']['Collection status'] == 'DELETED':
            title = "Deleted MinVectorDB collection with status: \n"
        else:
            title = "MinVectorDB collection with status: \n"

        report = '\n* - COLLECTION STATUS REPORT -\n'
        for key, value in self.status_report_['COLLECTION STATUS REPORT'].items():
            report += f'| - {key}: {value}\n'

        return title + report

    def __str__(self):
        return self.__repr__()

    @property
    def status_report_(self):
        """
        Return the database report.

        Returns:
            dict: The database report.

        Raises:
            ExecutionError: If the server returns an error.
        """
        name = "Collection"

        url = f'{self._url}/is_collection_exists'
        data = {"collection_name": self._collection_name}
        response = requests.post(url, json=data)

        if response.status_code != 200:
            raise ExecutionError(response.json())

        is_exists = response.json()['params']['exists']

        last_commit_time = self._get_commit_msg()
        db_report = {f'{name.upper()} STATUS REPORT': {
            f'{name} shape': (0, self._init_params['dim']) if self.IS_DELETED else self.shape,
            f'{name} last_commit_time': last_commit_time,
            f'{name} index_mode': self._init_params['index_mode'],
            f'{name} distance': self._init_params['distance'],
            f'{name} use_cache': self._init_params['use_cache'],
            f'{name} status': 'DELETED' if self.IS_DELETED or not is_exists else 'ACTIVE'
        }}

        return db_report


class MinVectorDBHTTPClient:
    def __init__(self, url):
        """
        Initialize the client.
            .. versionadded:: 0.3.2

        Parameters:
            url (str): The URL of the server, must start with "http://" or "https://".

        Raises:
            TypeError: If the URL is not a string.
            ValueError: If the URL does not start with "http://" or "https://".
            ConnectionError: If the server cannot be connected to.
        """

        raise_if(TypeError, not isinstance(url, str), 'The URL must be a string.')
        raise_if(ValueError, not url.startswith('http://') or url.startswith('https://'),
                 'The URL must start with "http://" or "https://".')

        if url.endswith('/'):
            self.url = url[:-1]
        else:
            self.url = url

        try:
            if requests.get(url).json() != {'status': 'success', 'message': 'MinVectorDB HTTP API'}:
                raise ConnectionError(f'Failed to connect to the server at {url}.')

        except requests.exceptions.ConnectionError:
            raise ConnectionError(f'Failed to connect to the server at {url}.')

    def require_collection(
            self,
            collection: str,
            dim: int = None,
            n_clusters: int = 16,
            chunk_size: int = 100_000,
            distance: str = 'cosine',
            index_mode: str = 'IVF-FLAT',
            dtypes: str = 'float32',
            use_cache: bool = True,
            scaler_bits: Union[int, None] = 8,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            drop_if_exists: bool = False
    ):
        """
        Create a collection.
            .. versionadded:: 0.3.2

        Parameters:
            collection (str): The name of the collection.
            dim (int): The dimension of the vectors. Default is None.
                When creating a new collection, the dimension of the vectors must be specified.
                When loading an existing collection, the dimension of the vectors is automatically loaded.
            n_clusters (int): The number of clusters. Default is 16.
            chunk_size (int): The chunk size. Default is 100,000.
            distance (str): The distance metric. Default is 'cosine'.
            index_mode (str): The index mode. Default is 'IVF-FLAT'.
            dtypes (str): The data types. Default is 'float32'.
            use_cache (bool): Whether to use cache. Default is True.
            scaler_bits (int): The scaler bits. Default is 8.
            n_threads (int): The number of threads. Default is 10.
            warm_up (bool): Whether to warm up. Default is False.
            drop_if_exists (bool): Whether to drop the collection if it exists. Default is False.

        Returns:
            Collection: The collection object.

        Raises:
            ConnectionError: If the server cannot be connected to.
        """
        url = f'{self.url}/required_collection'

        data = {
            "collection_name": collection,
            "dim": dim,
            "n_clusters": n_clusters,
            "chunk_size": chunk_size,
            "distance": distance,
            "index_mode": index_mode,
            "dtypes": dtypes,
            "use_cache": use_cache,
            "scaler_bits": scaler_bits,
            "n_threads": n_threads,
            "warm_up": warm_up,
            "drop_if_exists": drop_if_exists
        }

        try:
            requests.post(url, json=data)
            del data['collection_name']
            return Collection(self.url, collection, **data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f'Failed to connect to the server at {url}.')

    def drop_collection(self, collection: str):
        """
        Drop a collection.
            .. versionadded:: 0.3.2

        Parameters:
            collection (str): The name of the collection.

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        url = f'{self.url}/drop_collection'
        data = {"collection_name": collection}
        response = requests.post(url, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise ExecutionError(response.json())

    def drop_database(self):
        """
        Drop the database.
            .. versionadded:: 0.3.2

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        if not self.database_exists()['params']['exists']:
            return {'status': 'success', 'message': 'The database does not exist.'}

        url = f'{self.url}/drop_database'
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise ExecutionError(response.text)

    def database_exists(self):
        """
        Check if the database exists.
            .. versionadded:: 0.3.2

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        url = f'{self.url}/database_exists'
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise ExecutionError(response.json())

    def show_collections(self):
        """
        Show all collections in the database.
            .. versionadded:: 0.3.2

        Returns:
            List: The list of collections.

        Raises:
            ExecutionError: If the server returns an error.
        """
        url = f'{self.url}/show_collections'
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()['params']['collections']
        else:
            raise ExecutionError(response.json())

    def set_environment(self, env: dict):
        """
        Set the environment variables.
            .. versionadded:: 0.3.2

        Parameters:
            env (dict): The environment variables. It can be specified on the same time or separately.
                - MVDB_LOG_LEVEL: The log level.
                - MVDB_LOG_PATH: The log path.
                - MVDB_TRUNCATE_LOG: Whether to truncate the log.
                - MVDB_LOG_WITH_TIME: Whether to log with time.
                - MVDB_KMEANS_EPOCHS: The number of epochs for KMeans.
                - MVDB_QUERY_CACHE_SIZE: The query cache size.
                - MVDB_DATALOADER_BUFFER_SIZE: The dataloader buffer size.

        Returns:
            dict: The response from the server.

        Raises:
            TypeError: If the value of an environment variable is not a string.
            ExecutionError: If the server returns an error.
        """
        url = f'{self.url}/set_environment'

        env_list = ['MVDB_LOG_LEVEL', 'MVDB_LOG_PATH', 'MVDB_TRUNCATE_LOG', 'MVDB_LOG_WITH_TIME',
                    'MVDB_KMEANS_EPOCHS', 'MVDB_QUERY_CACHE_SIZE', 'MVDB_DATALOADER_BUFFER_SIZE']

        data = {}
        for key in env:
            if key in env_list:
                raise_if(TypeError, not isinstance(env[key], str), f'The value of {key} must be a string.')
                data[key] = env[key]

        response = requests.post(url, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise ExecutionError(response.json())

    def get_environment(self):
        """
        Get the environment variables.
            .. versionadded:: 0.3.2

        Returns:
            dict: The response from the server.

        Raises:
            ExecutionError: If the server returns an error.
        """
        url = f'{self.url}/get_environment'
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise ExecutionError(response.json())

    def get_collection(self, collection: str):
        """
        Get a collection.
            .. versionadded:: 0.3.2

        Parameters:
            collection (str): The name of the collection.

        Returns:
            Collection: The collection object.

        Raises:
            ExecutionError: If the server returns an error.
        """
        url = f'{self.url}/is_collection_exists'
        data = {"collection_name": collection}
        response = requests.post(url, json=data)

        if response.status_code == 200 and response.json()['params']['exists']:
            url = f'{self.url}/get_collection_config'
            data = {"collection_name": collection}
            response = requests.post(url, json=data)

            return Collection(self.url, collection, **response.json()['params']['config'])
        else:
            raise ExecutionError(response.json())

    def __repr__(self):
        if self.database_exists()['params']['exists']:
            return f'MinVectorDB HTTP Client connected to {self.url}.'
        else:
            return f'MinVectorDB remote server at {self.url} does not exist.'

    def __str__(self):
        return self.__repr__()
