import os
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
from spinesUtils.asserts import raise_if, ParameterTypeAssert
from spinesUtils.timer import Timer

from ...configs.parameters_validator import ParametersValidator
from ...core_components.fields_cache import IndexSchema
from ...core_components.locks import ThreadLock
from ...execution_layer.indexer import Indexer
from ...execution_layer.search import Search
from ...execution_layer.matrix_serializer import MatrixSerializer
from ...utils.utils import unavailable_if_deleted, unavailable_if_empty, collection_repr
from ...api import logger
from ...core_components.fields_cache.filter import Filter


class ExclusiveDB:
    """
    A class for managing a vector database stored in chunk files and computing vectors similarity.
    The class is exclusive and cannot be shared with other threads or processes,
    so it is not thread-safe or process-safe.

    Note:
        This class is not called at the top level, but is called through the LocalClient class.
        When directly operating on data by calling this class, users need to be clear about what they are doing.
    """
    name = "Local"

    @ParametersValidator(
        immutable_param=['dim', 'chunk_size', 'dtypes'],
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
            cache_chunks (int): The number of chunks to cache in memory. Default is 20.

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

        # In order to have enough data to index the segment
        raise_if(ValueError, chunk_size < 1000, 'chunk_size must greater than or equal to 1000.')

        self._database_path = database_path
        self._database_name = Path(database_path).parent.name
        self._collection_name = Path(database_path).name

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

        self._search.single_search.clear_cache()

        # initial lock
        self._lock = ThreadLock()

        # pre_build_index if the ExclusiveDB instance is reloaded
        if self.shape[0] > 0:
            self._pre_build_index()

    @unavailable_if_deleted
    def add_item(self, vector: Union[np.ndarray, list], id: int, *, field: dict = None,
                 buffer_size: Union[int, None, bool] = True):
        """
        Add a single vector to the collection.

        It is recommended to use incremental ids for best performance.

        Parameters:
            vector (np.ndarray): The vector to be added.
            id (int): The ID of the vector.
            field (dict, optional, keyword-only): The field of the vector. Default is None.
                If None, the field will be set to an empty string.
            buffer_size (int or bool or None): The buffer size for the storage worker. Default is True.

                - If None, the vector will be directly written to the disk.
                - If True, the buffer_size will be set to chunk_size,
                    and the vectors will be written to the disk when the buffer is full.
                - If False, the vector will be directly written to the disk.
                - If int, when the buffer is full, the vectors will be written to the disk.

        Returns:
            (int): The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        return self._matrix_serializer.add_item(vector, id=id, field=field,
                                                buffer_size=buffer_size)

    @unavailable_if_deleted
    def bulk_add_items(
            self, vectors: Union[List[Tuple[np.ndarray, int, dict]], List[Tuple[np.ndarray, int]]],
            **kwargs
    ):
        """
        Bulk add vectors to the collection in batches.

        It is recommended to use incremental ids for best performance.

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved.
                Each vector can be a tuple of (vector, id, field).

        Returns:
            List[int]: A list of indices where the vectors are stored.
        """
        return self._matrix_serializer.bulk_add_items(vectors)

    @unavailable_if_deleted
    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        """
        with self._lock:
            self._matrix_serializer.commit()
            self._pre_build_index()

    @unavailable_if_deleted
    def _pre_build_index(self):
        if not hasattr(self._matrix_serializer, "indexer"):
            logger.info("Pre-building the index...")
            self.build_index(index_mode="Flat-IP")  # Default index mode

    @unavailable_if_deleted
    @unavailable_if_empty
    def build_index(self, index_mode, rebuild=False, **kwargs):
        """
        Build the index for clustering.

        Parameters:
            index_mode (str): The index mode, must be one of the following:

                - 'IVF-IP-SQ8': IVF index with inner product and scalar quantizer with 8 bits.
                    The distance is inner product.
                - 'IVF-IP': IVF index with inner product. (Alias: 'IVF')
                - 'IVF-L2sq-SQ8': IVF index with squared L2 distance and scalar quantizer with 8 bits.
                    The distance is squared L2 distance.
                - 'IVF-L2sq': IVF index with squared L2 distance.
                - 'IVF-Cos-SQ8': IVF index with cosine similarity and scalar quantizer with 8 bits.
                    The distance is cosine similarity.
                - 'IVF-Cos': IVF index with cosine similarity.
                - 'IVF-Jaccard-Binary': IVF index with binary quantizer. The distance is Jaccard distance.
                - 'IVF-Hamming-Binary': IVF index with binary quantizer. The distance is Hamming distance.
                - 'Flat-IP-SQ8': Flat index with inner product and scalar quantizer with 8 bits.
                - 'Flat-IP': Flat index with inner product. (Alias: 'FLAT')
                - 'Flat-L2sq-SQ8': Flat index with squared L2 distance and scalar quantizer with 8 bits.
                - 'Flat-L2sq': Flat index with squared L2 distance.
                - 'Flat-Cos-SQ8': Flat index with cosine similarity and scalar quantizer with 8 bits.
                - 'Flat-Cos': Flat index with cosine similarity.
                - 'Flat-Jaccard-Binary': Flat index with binary quantizer. The distance is Jaccard distance.
                - 'Flat-Hamming-Binary': Flat index with binary quantizer. The distance is Hamming distance.
            rebuild (bool): Whether to rebuild the index.
            kwargs: Additional keyword arguments. The following are available:

                - 'n_clusters' (int): The number of clusters. It is only available when the index_mode including 'IVF'.

        Returns:
            None
        """
        with self._lock:
            self._indexer.build_index(index_mode, rebuild=rebuild, **kwargs)
            self._matrix_serializer.indexer = self._indexer

    @unavailable_if_deleted
    @unavailable_if_empty
    def remove_index(self):
        """
        Remove the vector index.

        Returns:
            None
        """
        self._indexer.remove_index()
        logger.info("Fallback to default index mode: `Flat-IP`.")
        self.build_index(index_mode='Flat-IP')  # Default index mode

    @unavailable_if_deleted
    @unavailable_if_empty
    def search(self, vector: Union[np.ndarray, list], k: int = 10, *,
               search_filter: Filter = None, return_fields=False, **kwargs):
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

                - rescore (bool): Whether to rescore the results of binary or scaler quantization searches.
                    Default is False. It is recommended to set it to True when the index mode is 'Binary'.
                - rescore_multiplier (int): The multiplier for the rescore operation.
                    It is only available when rescore is True.
                    If 'Binary' is in the index mode, the default is 10. Otherwise, the default is 2.

        Returns:
            (Tuple[List[int], List[float], List[Dict]] or Tuple[List[int], List[float], None]):
                If return_fields is True, the indices, similarity scores,
                and fields of the nearest vectors in the database.
                Otherwise, the indices and similarity scores of the nearest vectors in the database.

        Raises:
            ValueError: If the database is empty.
        """
        raise_if(ValueError, not isinstance(vector, (np.ndarray, list)),
                 'vector must be np.ndarray or list.')

        raise_if(TypeError, not isinstance(k, int) and not (isinstance(k, str) and k != 'all'),
                 'k must be int or "all".')
        raise_if(ValueError, k <= 0, 'k must be greater than 0.')
        raise_if(ValueError, not isinstance(search_filter, (Filter, type(None), str)),
                 'search_filter must be Filter or None or FieldExpression string.')

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
            self._search.single_search.clear_cache()

        res = self._search.search(vector=vector, k=k, search_filter=search_filter,
                                  return_fields=return_fields, **kwargs)

        return res

    @unavailable_if_deleted
    def is_id_exists(self, id):
        """
        Check if the ID exists in the database.

        Parameters:
            id (int): The ID to check.

        Returns:
            Bool: True if the ID exists, False otherwise.
        """
        return id in self._matrix_serializer.id_filter

    @property
    @unavailable_if_deleted
    @unavailable_if_empty
    def max_id(self):
        """
        Return the maximum ID in the database.

        Returns:
            (int): The maximum ID in the database.
        """
        return self._matrix_serializer.id_filter.find_max_value()

    @unavailable_if_deleted
    @unavailable_if_empty
    def query(self, query_filter, filter_ids=None, return_ids_only=False):
        """
        Query the fields cache.

        Parameters:
            query_filter (str): Filter or dict or FieldExpression string
                The filter object or the specify data to filter.
            filter_ids (List[int]):
                The list of external IDs to filter.
            return_ids_only (bool):
                If True, only the external IDs will be returned.

        Returns:
            (List[Dict]): The records. If not return_ids_only, the records will be returned.
            (List[int]): The external IDs. If return_ids_only, the external IDs will be returned.
        """
        return self._matrix_serializer.field_index.query(query_filter, filter_ids, return_ids_only)

    @unavailable_if_deleted
    @unavailable_if_empty
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
        ids = self._matrix_serializer.field_index.query(query_filter, filter_ids, return_ids_only=True)
        if not ids:
            return np.array([]), np.array([]), []

        return self.read_by_only_id(ids)

    @property
    def shape(self):
        """
        Return the shape of the entire database.

        Returns:
            (Tuple[int, int]): The number of vectors and the dimension of each vector in the database.
        """
        return self._matrix_serializer.shape

    @unavailable_if_deleted
    def insert_session(self):
        """
        Create a session to insert data, which will automatically commit the data when the session ends.

        Returns:
            DataOpsSession (DataOpsSession): The session object.
        """
        from ...execution_layer.session import DataOpsSession

        return DataOpsSession(self)

    @unavailable_if_deleted
    @unavailable_if_empty
    def head(self, n=5):
        """
        Return the first n vectors in the database.

        Parameters:
            n (int): The number of vectors to return. Default is 5.

        Returns:
            (Tuple[List[np.ndarray], List[int], List[Dict]]): The vectors, IDs, and fields of the items.
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
            return (np.asarray(indices)[:n], np.vstack(data, dtype=dtypes)[:n],
                    self._matrix_serializer.field_index.retrieve_ids(indices[:n],
                                                                     include_external_id=True))
        return np.asarray(indices), np.array(data, dtype=dtypes), []

    @unavailable_if_deleted
    @unavailable_if_empty
    def tail(self, n=5):
        """
        Return the last n vectors in the database.

        Parameters:
            n (int): The number of vectors to return. Default is 5.

        Returns:
            (Tuple[List[np.ndarray], List[int], List[Dict]]): The vectors, IDs, and fields of the items.
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
            return (np.asarray(indices)[-n:], np.vstack(data, dtype=dtypes)[-n:],
                    self._matrix_serializer.field_index.retrieve_ids(indices[-n:],
                                                                     include_external_id=True))
        return np.asarray(indices), np.array(data, dtype=dtypes), []

    @unavailable_if_deleted
    @unavailable_if_empty
    def yield_every_single_element(self, limit=None):
        """
        Yield every single element in the database.

        Parameters:
            limit (int): The maximum number of elements to yield. Default is None.

        Returns:
            (Tuple[np.ndarray, int, Dict]): The vectors, IDs, and fields of the items

        """
        raise_if(TypeError, limit is not None and not isinstance(limit, int), "limit must be an integer.")
        filenames = self._matrix_serializer.storage_worker.get_all_files()
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        count = 0
        for filename in filenames:
            data_chunk, index_chunk = self._matrix_serializer.dataloader(filename)
            for i in range(len(index_chunk)):
                yield int(index_chunk[i]), np.asarray(data_chunk[i]), \
                    self._matrix_serializer.field_index.retrieve([index_chunk[i]], include_external_id=True)
                count += 1
                if limit and count >= limit:
                    return

    @unavailable_if_deleted
    @unavailable_if_empty
    def read_by_only_id(self, id: Union[int, list]):
        """
        Read the vector data by the external ID.

        Parameters:
            id (int or list): The external ID or list of external IDs.

        Returns:
            (Tuple[np.ndarray, List[int], List[Dict]], Tuple[np.ndarray, int, Dict]):
                The vectors, IDs, and fields of the items.
        """
        data, ids = self._matrix_serializer.storage_worker.read_by_only_id(id)

        if data.shape[0] > 0:
            return np.asarray(ids), np.asarray(data), \
                self._matrix_serializer.field_index.retrieve_ids(ids, include_external_id=True)
        return np.asarray(ids), np.asarray(data), []

    @unavailable_if_deleted
    @unavailable_if_empty
    def list_fields(self):
        """
        Return all field names and their types.

        Returns:
            Dict: A dictionary with field names as keys and field types as values.
        """
        return self._matrix_serializer.field_index.storage.list_fields()

    @unavailable_if_deleted
    @unavailable_if_empty
    def build_field_index(self, schema, rebuild_if_exists=False):
        """
        Build an index for the field.

        Parameters:
            schema (IndexSchema or Field name string): The schema of the field or the field name string.
                When passing the field name string, the field name must be wrapped with ':', like ':vector:', ':timestamp:'.
            rebuild_if_exists (bool): Whether to rebuild the index if it already exists.

        Returns:
            None

        Note:
            The :id: is a reserved field name and cannot be used.
        """
        if not isinstance(schema, (IndexSchema, str)):
            raise TypeError("schema must be an instance of IndexSchema or a field string.")
        self._matrix_serializer.field_index.build_index(schema, rebuild_if_exists=rebuild_if_exists)

    @unavailable_if_deleted
    @unavailable_if_empty
    def remove_field_index(self, field_name):
        """
        Remove the index for the field.

        Parameters:
            field_name (str): The name of the field.

        Returns:
            None
        """
        if not field_name.startswith(':') or not field_name.endswith(':'):
            raise ValueError("The field name must be wrapped with ':'.")

        field_name = field_name.strip(':')
        self._matrix_serializer.field_index.remove_index(field_name)

    @unavailable_if_deleted
    @unavailable_if_empty
    def remove_all_field_indices(self):
        """
        Remove all the field indices.

        Returns:
            None
        """
        self._matrix_serializer.field_index.remove_all_field_indices()

    @unavailable_if_deleted
    @unavailable_if_empty
    def list_field_index(self):
        """
        List the field index.

        Returns:
            List: The list of field indices.
        """
        return self._matrix_serializer.field_index.list_indices()

    def delete(self):
        """
        Delete the database.

        This API is not part of the Collection in the HTTP API, but rather part of the HTTPClient in the HTTP API
        Users must call this API through the HTTPClient when they want to delete a collection on the server.

        Returns:
            None
        """
        if self._matrix_serializer.IS_DELETED:
            return

        import gc

        with self._lock:
            self._search.single_search.clear_cache()
            self._search.delete()
            self._matrix_serializer.delete()

        gc.collect()

    @property
    @unavailable_if_empty
    def index_mode(self):
        """
        Return the index mode of the database.

        Returns:
            (str or None): The index mode of the database.
        """
        if not hasattr(self._matrix_serializer, "indexer"):
            return None

        return self._matrix_serializer.indexer.index_mode

    def is_deleted(self):
        """
        To check if the database is deleted.

        Returns:
            Bool: True if the database is deleted, False otherwise.
        """
        return self._matrix_serializer.IS_DELETED

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return collection_repr(self)

    def __str__(self):
        return self.__repr__()
