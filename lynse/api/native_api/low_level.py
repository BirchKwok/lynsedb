import os
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
from spinesUtils.asserts import raise_if, ParameterTypeAssert
from spinesUtils.timer import Timer

from lynse.configs.parameters_validator import ParametersValidator
from lynse.core_components.kv_cache import VeloKV, IndexSchema
from lynse.execution_layer.cluster_worker import ClusterWorker
from lynse.execution_layer.search import Search
from lynse.execution_layer.matrix_serializer import MatrixSerializer
from lynse.utils import copy_doc
from lynse.utils.utils import unavailable_if_deleted
from lynse.api import logger
from lynse.core_components.kv_cache.filter import Filter


class ExclusiveDB:
    """
    A class for managing a vector database stored in chunk files and computing vectors similarity.
    The class is exclusive and cannot be shared with other threads or processes,
    so it is not thread-safe or process-safe.
    """
    name = "Local"

    @ParametersValidator(
        update_configs=['dim', 'chunk_size', 'dtypes', 'scaler_bits'],
        logger=logger
    )
    @ParameterTypeAssert({
        'dim': int,
        'database_path': str,
        'chunk_size': int,
        'distance': str,
        'dtypes': str,
        'use_cache': bool,
        'scaler_bits': (None, int),
        'n_threads': (None, int),
        'warm_up': bool,
        'initialize_as_collection': bool,
        'cache_chunks': int
    }, func_name='ExclusiveDB')
    def __init__(
            self,
            dim: int,
            database_path: Union[str, Path],
            chunk_size: int = 100_000,
            distance: str = 'cosine',
            dtypes: str = 'float32',
            use_cache: bool = True,
            scaler_bits: Union[int, None] = 8,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            initialize_as_collection: bool = False,
            cache_chunks: int = 20
    ) -> None:
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str or Path): The path to the database file.
            chunk_size (int): The size of each data chunk. Default is 100_000.
            distance (str): Method for calculating vector distance.
                Options are 'cosine' or 'L2' for Euclidean distance. Default is 'cosine'.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            use_cache (bool): Whether to use cache for search. Default is True.
            scaler_bits (int): The number of bits for scalar quantization.
                Options are 8, 16, or 32. The default is None, which means no scalar quantization.
                The 8 for 8-bit, 16 for 16-bit, and 32 for 32-bit.
            n_threads (int): The number of threads to use for parallel processing. Default is 10.
            warm_up (bool): Whether to warm up the database. Default is False.
            initialize_as_collection (bool): Whether to initialize the database as a collection.
            cache_chunks (int): The buffer size for reading and writing data. Default is 20.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        raise_if(ValueError, chunk_size <= 1, 'chunk_size must greater than 1')
        raise_if(ValueError, distance not in ('cosine', 'L2'), 'distance must be "cosine" or "L2"')
        raise_if(ValueError, dtypes not in ('float16', 'float32', 'float64'),
                 'dtypes must be "float16", "float32" or "float64')
        raise_if(ValueError, scaler_bits not in (8, 16, 32, None), 'sq_bits must be 8, 16, 32 or None')

        if not initialize_as_collection:
            logger.info("Initializing LynseDB with: \n "
                        f"\r//    dim={dim}, database_path='{database_path}', \n"
                        f"\r//    chunk_size={chunk_size}, distance='{distance}', \n"
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
            chunk_size=chunk_size,
            logger=logger,
            dtypes=dtypes,
            scaler_bits=scaler_bits,
            warm_up=warm_up,
            cache_chunks=cache_chunks
        )
        self._data_loader = self._matrix_serializer.dataloader
        self._id_filter = self._matrix_serializer.id_filter

        self._timer = Timer()
        self._use_cache = use_cache
        self._distance = distance

        raise_if(TypeError, n_threads is not None and not isinstance(n_threads, int), "n_threads must be an integer.")
        raise_if(ValueError, n_threads is not None and n_threads <= 0, "n_threads must be greater than 0.")

        self._cluster_worker = ClusterWorker(
            logger=logger,
            dataloader=self._data_loader,
            storage_worker=self._matrix_serializer.storage_worker,
            collections_path_parent=self._matrix_serializer.collections_path_parent
        )

        self._search = Search(
            matrix_serializer=self._matrix_serializer,
            cluster_worker=self._cluster_worker,
            n_threads=n_threads if n_threads else min(32, os.cpu_count() + 4),
            distance=distance
        )

        self._search.search.clear_cache()

        self.most_recent_search_report = {}

        self._initialize_as_collection = initialize_as_collection

        # Set the docstrings
        copy_doc(self.add_item, MatrixSerializer.add_item)
        copy_doc(self.bulk_add_items, MatrixSerializer.bulk_add_items)
        copy_doc(self.build_index, ClusterWorker.build_index)
        copy_doc(self.remove_index, ClusterWorker.remove_index)
        copy_doc(self.search, Search.search)
        copy_doc(self.query, VeloKV.query)

    @unavailable_if_deleted
    def add_item(self, vector: Union[np.ndarray, list], id: int, *, field: dict = None,
                 normalize: bool = False, buffer_size: Union[int, None, bool] = None):
        return self._matrix_serializer.add_item(vector, id=id, field=field,
                                                normalize=normalize, buffer_size=buffer_size)

    @unavailable_if_deleted
    def bulk_add_items(
            self, vectors: Union[List[Tuple[np.ndarray, int, dict]], List[Tuple[np.ndarray, int]]],
            *, normalize=False,
            **kwargs
    ):
        return self._matrix_serializer.bulk_add_items(vectors, normalize=normalize)

    @unavailable_if_deleted
    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        """
        self._matrix_serializer.commit()

    @unavailable_if_deleted
    def build_index(self, index_mode='IVF-FLAT', n_clusters=32):
        self._cluster_worker.build_index(index_mode, n_clusters)

    @unavailable_if_deleted
    def remove_index(self):
        self._cluster_worker.remove_index()

    @unavailable_if_deleted
    def search(self, vector: Union[np.ndarray, list], k: int = 12, *,
               search_filter: Filter = None,
               distance: Union[str, None] = None, normalize=False, return_fields=False):
        raise_if(ValueError, not isinstance(vector, (np.ndarray, list)), 'vector must be np.ndarray or list.')

        import datetime

        logger.debug(f'Search vector: {vector.tolist() if isinstance(vector, np.ndarray) else vector}')
        logger.debug(f'Search k: {k}')
        logger.debug(f'Search distance: {self._distance if distance is None else distance}')

        raise_if(TypeError, not isinstance(k, int) and not (isinstance(k, str) and k != 'all'),
                 'k must be int or "all".')
        raise_if(ValueError, k <= 0, 'k must be greater than 0.')
        raise_if(ValueError, not isinstance(search_filter, (Filter, type(None))),
                 'search_filter must be Filter or None.')

        raise_if(ValueError, len(vector) != self._matrix_serializer.shape[1],
                 'vector must be same dim with database.')

        raise_if(ValueError, distance is not None and distance not in ['cosine', 'L2', 'IP'],
                 'distance must be "cosine" or "L2" or "IP" or None.')

        if self._matrix_serializer.shape[0] == 0:
            raise ValueError('database is empty.')

        if k > self._matrix_serializer.shape[0]:
            k = self._matrix_serializer.shape[0]

        self.most_recent_search_report = {}

        # update scaler
        self._search.update_scaler(self._matrix_serializer.scaler)

        self._timer.start()
        if self._use_cache:
            res = self._search.search(vector=vector, k=k, search_filter=search_filter,
                                      distance=distance, normalize=normalize, return_fields=return_fields)
        else:
            res = self._search.search(vector=vector, k=k, search_filter=search_filter,
                                      now_time=datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'),
                                      distance=distance,
                                      normalize=normalize, return_fields=return_fields)

        time_cost = self._timer.last_timestamp_diff()
        self.most_recent_search_report['Collection Shape'] = self.shape
        self.most_recent_search_report['Search Time'] = f"{time_cost :>.5f} s"
        self.most_recent_search_report['Search Distance'] = self._distance if distance is None else distance
        self.most_recent_search_report['Search K'] = k

        if res is not None:
            self.most_recent_search_report[f'Top {k} Results ID'] = res[0]
            self.most_recent_search_report[f'Top {k} Results Similarity'] = \
                np.array([round(i, 6) for i in res[1]]) if len(res[1]) > 0 else res[1]

        return res

    @unavailable_if_deleted
    def query(self, filter_instance, filter_ids=None, return_ids_only=False):
        return self._matrix_serializer.kv_index.query(filter_instance, filter_ids, return_ids_only)

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
        from lynse.execution_layer.session import DataOpsSession

        return DataOpsSession(self)

    def head(self, n=5):
        """
        Return the first n vectors in the database.

        Parameters:
            n (int): The number of vectors to return. Default is 5.

        Returns:
            np.ndarray: The first n vectors in the database.
        """
        filenames = self._matrix_serializer.storage_worker.get_all_files()
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        data, indices = [], []

        count = 0
        for filename in filenames:
            data_chunk, index_chunk = self._matrix_serializer.dataloader(filename)
            data.extend(data_chunk)
            indices.extend(index_chunk)
            count += len(index_chunk)
            if count >= n:
                break

        if data:
            return (np.vstack(data)[:n], np.array(indices)[:n],
                    self._matrix_serializer.kv_index.retrieve_ids(indices[:n], include_external_id=True))
        return np.array(data), np.array(indices), []

    def tail(self, n=5):
        """
        Return the last n vectors in the database.

        Parameters:
            n (int): The number of vectors to return. Default is 5.

        Returns:
            np.ndarray: The last n vectors in the database.
        """
        filenames = self._matrix_serializer.storage_worker.get_all_files()
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        data, indices = [], []

        count = 0
        for filename in filenames[::-1]:
            data_chunk, index_chunk = self._matrix_serializer.dataloader(filename)
            data.insert(0, data_chunk)
            indices = index_chunk.tolist() + indices
            count += len(index_chunk)
            if count >= n:
                break

        if data:
            return (np.vstack(data)[-n:], np.array(indices)[-n:],
                    self._matrix_serializer.kv_index.retrieve_ids(indices[-n:], include_external_id=True))
        return np.array(data), np.array(indices), []

    def read_by_only_id(self, id: Union[int, list]):
        """
        Read the vector data by the external ID.

        Parameters:
            id (int or list): The external ID or list of external IDs.

        Returns:
            tuple: The vector data and the ID and field of the vector.
        """
        data, ids = self._matrix_serializer.storage_worker.read_by_only_id(id)
        if data.shape[0] > 0:
            return np.array(data), np.array(ids), \
                self._matrix_serializer.kv_index.retrieve_ids(ids, include_external_id=True)
        return np.array(data), np.array(ids), []

    def build_field_index(self, schema, rebuild_if_exists=False):
        """
        Build an index for the field.

        Parameters:
            schema (IndexSchema): The schema of the field.
            rebuild_if_exists (bool): Whether to rebuild the index if it already exists.
        """
        if not isinstance(schema, IndexSchema):
            raise TypeError("schema must be an instance of IndexSchema.")
        self._matrix_serializer.kv_index.build_index(schema, rebuild_if_exists=rebuild_if_exists)

    def remove_field_index(self, field_name):
        """
        Remove the index for the field.

        Parameters:
            field_name (str): The name of the field.
        """
        self._matrix_serializer.kv_index.remove_index(field_name)

    def remove_all_field_indices(self):
        """
        Remove all the field indices.
        """
        self._matrix_serializer.kv_index.remove_all_field_indices()

    def list_field_index(self):
        """
        List the field index.

        Returns:
            list: The list of field indices.
        """
        return self._matrix_serializer.kv_index.list_indices()

    def delete(self):
        """
        Delete the database.
        """
        if self._matrix_serializer.IS_DELETED:
            return

        import gc

        self._search.search.clear_cache()
        self._search.delete()
        self._matrix_serializer.delete()

        gc.collect()

    @property
    def search_report_(self):
        """
        Return the most recent search report.
        """
        report = '\n* - MOST RECENT SEARCH REPORT -\n'
        for key, value in self.most_recent_search_report.items():
            if key == 'Collection Shape':
                value = self.shape

            report += f'| - {key}: {value}\n'

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
            f'{name} distance': self._distance,
            f'{name} use_cache': self._use_cache,
            f'{name} status': 'DELETED' if self._matrix_serializer.IS_DELETED else 'ACTIVE'
        }}

        return db_report

    def __repr__(self):
        if self._matrix_serializer.IS_DELETED:
            if self._initialize_as_collection:
                title = "Deleted LynseDB collection with status: \n"
            else:
                title = "Deleted LynseDB object with status: \n"
        else:
            if self._initialize_as_collection:
                title = "LynseDB collection with status: \n"
            else:
                title = "LynseDB object with status: \n"

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
