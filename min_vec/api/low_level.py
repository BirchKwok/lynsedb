"""low_level.py - The MinVectorDB API."""
import os
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
from spinesUtils.asserts import raise_if, ParameterTypeAssert
from spinesUtils.timer import Timer

from min_vec.configs.parameters_validator import ParametersValidator
from min_vec.core_components.cross_lock import ThreadLock
from min_vec.execution_layer.query import Query
from min_vec.execution_layer.matrix_serializer import MatrixSerializer
from min_vec.utils.utils import unavailable_if_deleted
from min_vec.api import logger
from min_vec.core_components.filter import Filter


class StandaloneMinVectorDB:
    """
    A class for managing a vector database stored in .mvdb files and computing vectors similarity.
    """

    @ParametersValidator(
        update_configs=['dim', 'n_clusters', 'chunk_size', 'index_mode', 'dtypes', 'scaler_bits'],
        logger=logger
    )
    @ParameterTypeAssert({
        'dim': int,
        'database_path': str,
        'n_clusters': int,
        'chunk_size': int,
        'distance': str,
        'index_mode': str,
        'dtypes': str,
        'use_cache': bool,
        'scaler_bits': (None, int),
        'n_threads': (None, int),
        'warm_up': bool,
        'initialize_as_collection': bool
    }, func_name='StandaloneMinVectorDB')
    def __init__(
            self, dim: int, database_path: Union[str, Path], n_clusters: int = 16, chunk_size: int = 100_000,
            distance: str = 'cosine', index_mode: str = 'IVF-FLAT', dtypes: str = 'float32',
            use_cache: bool = True, scaler_bits: Union[int, None] = 8, n_threads: Union[int, None] = 10,
            warm_up: bool = False, initialize_as_collection: bool = False
    ) -> None:
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str or Path): The path to the database file.
            n_clusters (int): The number of clusters for the IVF-FLAT index. Default is 8.
            chunk_size (int): The size of each data chunk. Default is 100_000.
            distance (str): Method for calculating vector distance.
                Options are 'cosine' or 'L2' for Euclidean distance. Default is 'cosine'.
            index_mode (str): The storage mode of the database.
                Options are 'FLAT' or 'IVF-FLAT'. Default is 'IVF-FLAT'.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            use_cache (bool): Whether to use cache for query. Default is True.
            scaler_bits (int): The number of bits for scalar quantization.
                Options are 8, 16, or 32. The default is None, which means no scalar quantization.
                The 8 for 8-bit, 16 for 16-bit, and 32 for 32-bit.
            n_threads (int): The number of threads to use for parallel processing. Default is 10.
            warm_up (bool): Whether to warm up the database. Default is False.
                .. versionadded:: 0.2.6
            initialize_as_collection (bool): Whether to initialize the database as a collection.
                .. versionadded:: 0.3.0

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        raise_if(ValueError, chunk_size <= 1, 'chunk_size must greater than 1')
        raise_if(ValueError, distance not in ('cosine', 'L2'), 'distance must be "cosine" or "L2"')
        raise_if(ValueError, index_mode not in ('FLAT', 'IVF-FLAT'), 'index_mode must be "FLAT" or "IVF-FLAT"')
        raise_if(ValueError, dtypes not in ('float16', 'float32', 'float64'),
                 'dtypes must be "float16", "float32" or "float64')
        raise_if(ValueError, not isinstance(n_clusters, int) or n_clusters <= 0,
                 'n_clusters must be int and greater than 0')
        raise_if(ValueError, scaler_bits not in (8, 16, 32, None), 'sq_bits must be 8, 16, 32 or None')

        if not initialize_as_collection:
            logger.info("Initializing MinVectorDB with: \n "
                        f"\r//    dim={dim}, database_path='{database_path}', \n"
                        f"\r//    n_clusters={n_clusters}, chunk_size={chunk_size},\n"
                        f"\r//    distance='{distance}', index_mode='{index_mode}', \n"
                        f"\r//    dtypes='{dtypes}', use_cache={use_cache}, \n"
                        f"\r//    scaler_bits={scaler_bits}, n_threads={n_threads}, \n"
                        f"\r//    warm_up={warm_up}, initialize_as_collection={initialize_as_collection}"
                        )

        if chunk_size <= 1:
            raise ValueError('chunk_size must be greater than 1')

        self._database_path = database_path

        self._matrix_serializer = MatrixSerializer(
            dim=dim,
            collection_path=self._database_path,
            n_clusters=n_clusters,
            chunk_size=chunk_size,
            index_mode=index_mode,
            logger=logger,
            dtypes=dtypes,
            scaler_bits=scaler_bits,
            warm_up=warm_up
        )
        self._data_loader = self._matrix_serializer.dataloader
        self._id_filter = self._matrix_serializer.id_filter

        self._timer = Timer()
        self._use_cache = use_cache
        self._distance = distance

        raise_if(TypeError, n_threads is not None and not isinstance(n_threads, int), "n_threads must be an integer.")
        raise_if(ValueError, n_threads is not None and n_threads <= 0, "n_threads must be greater than 0.")

        self._query = Query(
            matrix_serializer=self._matrix_serializer,
            n_threads=n_threads if n_threads else min(32, os.cpu_count() + 4),
            distance=distance
        )

        self._query.query.clear_cache()

        self.most_recent_query_report = {}

        self._initialize_as_collection = initialize_as_collection

        if warm_up and self._matrix_serializer.shape[0] > 0:
            # Pre query once to cache the jax function
            self.query(np.ones(dim), k=1)

            self.most_recent_query_report = {}

        self.lock = ThreadLock()

    @unavailable_if_deleted
    def add_item(self, vector: Union[np.ndarray, list], id: int, *, field: dict = None) -> int:
        """
        Add a single vector to the database.

        Parameters:
            vector (np.ndarray or list): The vector to be added.
            id (int): The ID of the vector.
            field (dict, optional, keyword-only): The field of the vector. Default is None. If None, the field will be
                set to an empty string.

        Returns:
            int: The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        return self._matrix_serializer.add_item(vector, index=id, field=field)

    @unavailable_if_deleted
    def bulk_add_items(
            self, vectors: Union[List[Tuple[np.ndarray, int, dict]], List[Tuple[np.ndarray, int]]],
            **kwargs
    ):
        """
        Bulk add vectors to the database in batches.

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved. Each vector can be a tuple of (
            vector, id, field).
            kwargs: Additional keyword arguments. Of no practical significance, only to maintain programming norms.

        Returns:
            list: A list of indices where the vectors are stored.
        """
        return self._matrix_serializer.bulk_add_items(vectors)

    @unavailable_if_deleted
    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        This method is required to be called after saving vectors to query them.
        """
        self._matrix_serializer.commit()

    @unavailable_if_deleted
    def query(self, vector: Union[np.ndarray, list], k: int = 12, *,
              query_filter: Filter = None,
              distance: Union[str, None] = None,
              return_similarity: bool = True):
        """
        Query the database for the vectors most similar to the given vector in batches.

        Parameters:
            vector (np.ndarray or list): The query vector.
            k (int): The number of nearest vectors to return.
            query_filter (Filter, optional): The field filter to apply to the query.
            distance (str): The distance metric to use for the query.
                .. versionadded:: 0.2.7
            return_similarity (bool): Whether to return the similarity scores.Default is True.
                .. versionadded:: 0.2.5

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the database is empty.
        """
        raise_if(ValueError, not isinstance(vector, (np.ndarray, list)), 'vector must be np.ndarray or list.')

        import datetime

        logger.debug(f'Query vector: {vector.tolist() if isinstance(vector, np.ndarray) else vector}')
        logger.debug(f'Query k: {k}')
        logger.debug(f'Query distance: {self._distance if distance is None else distance}')
        logger.debug(f'Query return_similarity: {return_similarity}')

        raise_if(TypeError, not isinstance(k, int) and not (isinstance(k, str) and k != 'all'),
                 'k must be int or "all".')
        raise_if(ValueError, k <= 0, 'k must be greater than 0.')
        raise_if(ValueError, not isinstance(query_filter, (Filter, type(None))), 'query_filter must be Filter or None.')

        raise_if(ValueError, len(vector) != self._matrix_serializer.shape[1],
                 'vector must be same dim with database.')

        raise_if(ValueError, not isinstance(return_similarity, bool), 'return_similarity must be bool.')
        raise_if(ValueError, distance is not None and distance not in ['cosine', 'L2'],
                 'distance must be "cosine" or "L2" or None.')

        if self._matrix_serializer.shape[0] == 0:
            raise ValueError('database is empty.')

        if k > self._matrix_serializer.shape[0]:
            k = self._matrix_serializer.shape[0]

        self.most_recent_query_report = {}

        # update scaler
        self._query.update_scaler(self._matrix_serializer.scaler)

        self._timer.start()
        if self._use_cache:
            res = self._query.query(vector=vector, k=k, query_filter=query_filter, index_mode=self._matrix_serializer.index_mode,
                                    distance=distance, return_similarity=return_similarity)
        else:
            res = self._query.query(vector=vector, k=k, query_filter=query_filter,
                                    index_mode=self._matrix_serializer.index_mode,
                                    now_time=datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'),
                                    distance=distance,
                                    return_similarity=return_similarity)

        time_cost = self._timer.last_timestamp_diff()
        self.most_recent_query_report['Collection Shape'] = self.shape
        self.most_recent_query_report['Query Time'] = f"{time_cost :>.5f} s"
        self.most_recent_query_report['Query Distance'] = self._distance if distance is None else distance
        self.most_recent_query_report['Query K'] = k

        if len(res[0]) > 0:
            self.most_recent_query_report[f'Top {k} Results ID'] = res[0]
            if return_similarity:
                self.most_recent_query_report[f'Top {k} Results Similarity'] = np.array([round(i, 6) for i in res[1]])

        return res

    @property
    def shape(self):
        """
        Return the shape of the entire database.

        Returns:
            tuple: The number of vectors and the dimension of each vector in the database.
        """
        return self._matrix_serializer.shape

    @unavailable_if_deleted
    def insert_session(self):
        """
        Create a session to insert data, which will automatically commit the data when the session ends.
        """
        from min_vec.execution_layer.session import DatabaseSession

        return DatabaseSession(self)

    def delete(self):
        """
        Delete the database.
        """
        if self._matrix_serializer.IS_DELETED:
            return

        import gc

        self._matrix_serializer.delete()
        self._query.query.clear_cache()
        self._query.delete()

        gc.collect()

    @property
    def query_report_(self):
        """
        Return the most recent query report.
        """
        # print as a pretty string
        # title use bold font
        report = '\n* - MOST RECENT QUERY REPORT -\n'
        for key, value in self.most_recent_query_report.items():
            if key == 'Collection Shape':
                value = self.shape

            report += f'| - {key}: {value}\n'

        report += '* - END OF REPORT -\n'

        return report

    @property
    def status_report_(self):
        """
        Return the database report.
        """
        if self._initialize_as_collection:
            name = "Collection"
        else:
            name = "Database"

        db_report = {f'{name.upper()} STATUS REPORT': {
            f'{name} shape': (0, self._matrix_serializer.dim) if self._matrix_serializer.IS_DELETED else self.shape,
            f'{name} last_commit_time': self._matrix_serializer.last_commit_time,
            f'{name} index_mode': self._matrix_serializer.index_mode,
            f'{name} distance': self._distance,
            f'{name} use_cache': self._use_cache,
            f'{name} status': 'DELETED' if self._matrix_serializer.IS_DELETED else 'ACTIVE'
        }}

        return db_report

    def __repr__(self):
        if self._matrix_serializer.IS_DELETED:
            if self._initialize_as_collection:
                title = "Deleted MinVectorDB collection with status: \n"
            else:
                title = "Deleted MinVectorDB object with status: \n"
        else:
            if self._initialize_as_collection:
                title = "MinVectorDB collection with status: \n"
            else:
                title = "MinVectorDB object with status: \n"

        if self._initialize_as_collection:
            name = "Collection"
        else:
            name = "Database"

        report = f'\n* - {name.upper()} STATUS REPORT -\n'
        for key, value in self.status_report_[f'{name.upper()} STATUS REPORT'].items():
            report += f'| - {key}: {value}\n'

        return title + report

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.shape[0]

    def is_deleted(self):
        """To check if the database is deleted."""
        return self._matrix_serializer.IS_DELETED
