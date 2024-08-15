import os
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
from spinesUtils.asserts import raise_if, ParameterTypeAssert
from spinesUtils.timer import Timer

from ...configs.parameters_validator import ParametersValidator
from ...core_components.kv_cache import VeloKV, IndexSchema
from ...execution_layer.indexer import Indexer
from ...execution_layer.search import Search
from ...execution_layer.matrix_serializer import MatrixSerializer
from ...utils import copy_doc
from ...utils.utils import unavailable_if_deleted
from ...api import logger
from ...core_components.kv_cache.filter import Filter


class ExclusiveDB:
    """
    A class for managing a vector database stored in chunk files and computing vectors similarity.
    The class is exclusive and cannot be shared with other threads or processes,
    so it is not thread-safe or process-safe.
    """
    name = "Local"

    @ParametersValidator(
        update_configs=['dim', 'chunk_size', 'dtypes'],
        logger=logger
    )
    @ParameterTypeAssert({
        'dim': int,
        'database_path': str,
        'chunk_size': int,
        'dtypes': str,
        'use_cache': bool,
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
            dtypes: str = 'float32',
            use_cache: bool = True,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            initialize_as_collection: bool = False,
            cache_chunks: int = 20
    ):
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str or Path): The path to the database file.
            chunk_size (int): The size of each data chunk. Default is 100,000.
                It's recommended to be between 10,000 and 500,000.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            use_cache (bool): Whether to use cache for search. Default is True.
            n_threads (int): The number of threads to use for parallel processing. Default is 10.
            warm_up (bool): Whether to warm up the database. Default is False.
            initialize_as_collection (bool): Whether to initialize the database as a collection.
            cache_chunks (int): The buffer size for reading and writing data. Default is 20.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        raise_if(ValueError, chunk_size <= 1, 'chunk_size must greater than 1')
        raise_if(ValueError, dtypes not in ('float16', 'float32', 'float64'),
                 'dtypes must be "float16", "float32" or "float64')

        if chunk_size <= 1:
            raise ValueError('chunk_size must be greater than 1')

        if not 10000 <= chunk_size <= 500000:
            logger.warning('The recommended chunk size is between 10,000 and 500,000.')

        self._database_path = database_path

        self._matrix_serializer = MatrixSerializer(
            dim=dim,
            collection_path=self._database_path,
            chunk_size=chunk_size,
            logger=logger,
            dtypes=dtypes,
            warm_up=warm_up,
            cache_chunks=cache_chunks
        )
        self._data_loader = self._matrix_serializer.dataloader
        self._id_filter = self._matrix_serializer.id_filter

        self._timer = Timer()
        self._use_cache = use_cache

        raise_if(TypeError, n_threads is not None and not isinstance(n_threads, int), "n_threads must be an integer.")
        raise_if(ValueError, n_threads is not None and n_threads <= 0, "n_threads must be greater than 0.")

        self._indexer = Indexer(
            logger=logger,
            dataloader=self._data_loader,
            storage_worker=self._matrix_serializer.storage_worker,
            collections_path_parent=self._matrix_serializer.collections_path_parent
        )

        self._search = Search(
            matrix_serializer=self._matrix_serializer,
            n_threads=n_threads if n_threads else min(32, os.cpu_count() + 4),
        )

        self._search._single_search.clear_cache()

        self._initialize_as_collection = initialize_as_collection

        # Set the docstrings
        copy_doc(self.add_item, MatrixSerializer.add_item)
        copy_doc(self.bulk_add_items, MatrixSerializer.bulk_add_items)
        copy_doc(self.build_index, Indexer.build_index)
        copy_doc(self.remove_index, Indexer.remove_index)
        copy_doc(self.search, Search.search)
        copy_doc(self.query, VeloKV.query)

        # pre_build_index if the ExclusiveDB instance is reloaded
        if self.shape[0] > 0:
            self._pre_build_index()

    @unavailable_if_deleted
    def add_item(self, vector: Union[np.ndarray, list], id: int, *, field: dict = None,
                 buffer_size: Union[int, None, bool] = None):
        return self._matrix_serializer.add_item(vector, id=id, field=field,
                                                buffer_size=buffer_size)

    @unavailable_if_deleted
    def bulk_add_items(
            self, vectors: Union[List[Tuple[np.ndarray, int, dict]], List[Tuple[np.ndarray, int]]],
            **kwargs
    ):
        return self._matrix_serializer.bulk_add_items(vectors)

    @unavailable_if_deleted
    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        """
        self._matrix_serializer.commit()
        self._pre_build_index()

    @unavailable_if_deleted
    def _pre_build_index(self):
        if not hasattr(self._matrix_serializer, "indexer"):
            logger.info("Pre-building the index...")
            self.build_index(index_mode="Flat-IP")  # Default index mode

    @unavailable_if_deleted
    def build_index(self, index_mode='IVF-IP', rebuild=False, **kwargs):
        self._indexer.build_index(index_mode, rebuild=rebuild, **kwargs)
        self._matrix_serializer.indexer = self._indexer

    @unavailable_if_deleted
    def remove_index(self):
        self._indexer.remove_index()
        logger.info("Fallback to default index mode: `Flat-IP`.")
        self.build_index(index_mode='Flat-IP')  # Default index mode

    @unavailable_if_deleted
    def search(self, vector: Union[np.ndarray, list], k: int = 12, *,
               search_filter: Filter = None, return_fields=False, **kwargs):
        raise_if(ValueError, not isinstance(vector, (np.ndarray, list)),
                 'vector must be np.ndarray or list.')

        logger.debug(f'Search vector: {vector.tolist() if isinstance(vector, np.ndarray) else vector}')
        logger.debug(f'Search k: {k}')
        logger.debug(f'Search filter: {search_filter.to_dict() if search_filter else None}')
        logger.debug(f'Search return_fields: {return_fields}')

        raise_if(TypeError, not isinstance(k, int) and not (isinstance(k, str) and k != 'all'),
                 'k must be int or "all".')
        raise_if(ValueError, k <= 0, 'k must be greater than 0.')
        raise_if(ValueError, not isinstance(search_filter, (Filter, type(None))),
                 'search_filter must be Filter or None.')

        # Convert the vector to np.ndarray
        vector = np.atleast_2d(np.asarray(vector))

        raise_if(ValueError, vector.shape[1] != self._matrix_serializer.shape[1],
                 'vector must be same dim with database.')

        rescore = kwargs.setdefault('rescore', False)
        rescore_multiplier = kwargs.setdefault('rescore_multiplier', 2)
        raise_if(TypeError, not isinstance(rescore, bool), 'rescore must be bool.')
        raise_if(TypeError, not isinstance(rescore_multiplier, int), 'rescore_multiplier must be int.')

        if self._matrix_serializer.shape[0] == 0:
            raise ValueError('database is empty.')

        if k > self._matrix_serializer.shape[0]:
            k = self._matrix_serializer.shape[0]

        if not self._use_cache:
            self._search._single_search.clear_cache()

        res = self._search.search(vector=vector, k=k, search_filter=search_filter,
                                  return_fields=return_fields, **kwargs)

        return res

    @unavailable_if_deleted
    def is_id_exists(self, id):
        return id in self._matrix_serializer.id_filter

    @unavailable_if_deleted
    def max_id(self):
        return self._matrix_serializer.id_filter.find_max_value()

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
        from ...execution_layer.session import DataOpsSession

        return DataOpsSession(self)

    @unavailable_if_deleted
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
        dtypes = self._matrix_serializer.dtypes

        for filename in filenames:
            data_chunk, index_chunk = self._matrix_serializer.dataloader(filename)
            dtypes = data_chunk.dtype
            data.extend(data_chunk)
            indices.extend(index_chunk)
            count += len(index_chunk)
            if count >= n:
                break

        if data:
            return (np.vstack(data, dtype=dtypes)[:n], np.array(indices, dtype=np.uint64)[:n],
                    self._matrix_serializer.kv_index.retrieve_ids(indices[:n], include_external_id=True))
        return np.array(data, dtype=dtypes), np.array(indices, dtype=np.uint64), []

    @unavailable_if_deleted
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
        dtypes = self._matrix_serializer.dtypes

        for filename in filenames[::-1]:
            data_chunk, index_chunk = self._matrix_serializer.dataloader(filename)
            dtypes = data_chunk.dtype
            data.insert(0, data_chunk)
            indices = index_chunk.tolist() + indices
            count += len(index_chunk)
            if count >= n:
                break

        if data:
            return (np.vstack(data, dtype=dtypes)[-n:], np.array(indices, dtype=np.uint64)[-n:],
                    self._matrix_serializer.kv_index.retrieve_ids(indices[-n:], include_external_id=True))
        return np.array(data, dtype=dtypes), np.array(indices, dtype=np.uint64), []

    @unavailable_if_deleted
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
            return np.array(data, dtype=data.dtypes), np.array(ids, dtype=np.uint64), \
                self._matrix_serializer.kv_index.retrieve_ids(ids, include_external_id=True)
        return np.array(data, dtype=data.dtypes), np.array(ids, dtype=np.uint64), []

    @unavailable_if_deleted
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

    @unavailable_if_deleted
    def remove_field_index(self, field_name):
        """
        Remove the index for the field.

        Parameters:
            field_name (str): The name of the field.
        """
        self._matrix_serializer.kv_index.remove_index(field_name)

    @unavailable_if_deleted
    def remove_all_field_indices(self):
        """
        Remove all the field indices.
        """
        self._matrix_serializer.kv_index.remove_all_field_indices()

    @unavailable_if_deleted
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

        self._search._single_search.clear_cache()
        self._search.delete()
        self._matrix_serializer.delete()

        gc.collect()

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
            f'{name} use_cache': self._use_cache,
            f'{name} status': 'DELETED' if self._matrix_serializer.IS_DELETED else 'ACTIVE'
        }}

        return db_report

    @property
    def index_mode(self):
        if not hasattr(self._matrix_serializer, "indexer"):
            return None

        return self._matrix_serializer.indexer.index_mode

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
