"""api.py - The MinVectorDB API."""
from spinesUtils.asserts import raise_if
from spinesUtils.logging import Logger

from min_vec.configs.config import config
from min_vec.configs.parameters_validator import ParametersValidator


logger = Logger(
    fp=config.MVDB_LOG_PATH,
    name='MinVectorDB',
    truncate_file=config.MVDB_TRUNCATE_LOG,
    with_time=config.MVDB_LOG_WITH_TIME,
    level=config.MVDB_LOG_LEVEL
)


class MinVectorDB:
    """
    A class for managing a vector database stored in .mvdb files and computing vectors similarity.
    """

    from spinesUtils.asserts import ParameterValuesAssert, ParameterTypeAssert

    @ParameterTypeAssert({
        'use_cache': bool, 'reindex_if_conflict': bool
    }, func_name='MinVectorDB')
    @ParametersValidator(
        update_configs=['dim', 'database_path', 'n_cluster', 'chunk_size', 'index_mode', 'dtypes', 'scaler_bits'],
        logger=logger
    )
    def __init__(
            self, dim, database_path, n_cluster=16, chunk_size=100_000, distance='cosine',
            index_mode='IVF-FLAT', dtypes='float32',
            use_cache=True, reindex_if_conflict=False, scaler_bits=8, n_threads=10
    ) -> None:
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str): Path to the database file.
            n_cluster (int): The number of clusters for the IVF-FLAT index. Default is 8.
            chunk_size (int): The size of each data chunk. Default is 100_000.
            distance (str): Method for calculating vector distance.
                Options are 'cosine' or 'L2' for Euclidean distance. Default is 'cosine'.
            index_mode (str): The storage mode of the database.
                Options are 'FLAT' or 'IVF-FLAT'. Default is 'IVF-FLAT'.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            use_cache (bool): Whether to use cache for query. Default is True.
            reindex_if_conflict (bool): Whether to reindex if there is a conflict. Default is False.
            scaler_bits (int): The number of bits for scalar quantization.
                Options are 8, 16, or 32. The default is None, which means no scalar quantization.
                The 8 for 8-bit, 16 for 16-bit, and 32 for 32-bit.
            n_threads (int): The number of threads to use for parallel processing. Default is 10.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """

        from spinesUtils.timer import Timer

        from min_vec.execution_layer.query import Query
        from min_vec.execution_layer.matrix_serializer import MatrixSerializer

        raise_if(NotImplementedError, distance != 'cosine', "Only 'cosine' distance is supported for now.")

        logger.info("Initializing MinVectorDB with: \n "
                    f"\r//    dim={dim}, database_path='{database_path}', \n"
                    f"\r//    n_cluster={n_cluster}, chunk_size={chunk_size},\n"
                    f"\r//    distance='{distance}', index_mode='{index_mode}', \n"
                    f"\r//    dtypes='{dtypes}', use_cache={use_cache}, \n"
                    f"\r//    reindex_if_conflict={reindex_if_conflict}, scaler_bits={scaler_bits}\n"
                    )

        if chunk_size <= 1:
            raise ValueError('chunk_size must be greater than 1')

        self._matrix_serializer = MatrixSerializer(
            dim=dim,
            database_path=database_path,
            n_clusters=n_cluster,
            chunk_size=chunk_size,
            distance=distance,
            index_mode=index_mode,
            logger=logger,
            dtypes=dtypes,
            reindex_if_conflict=reindex_if_conflict,
            scaler_bits=scaler_bits
        )
        self._data_loader = self._matrix_serializer.iterable_dataloader
        self._id_filter = self._matrix_serializer.id_filter

        self._timer = Timer()
        self._use_cache = use_cache
        self._distance = distance

        raise_if(TypeError, not isinstance(n_threads, int), "n_threads must be an integer.")
        raise_if(ValueError, n_threads <= 0, "n_threads must be greater than 0.")

        self._matrix_query = Query(
            matrix_serializer=self._matrix_serializer,
            n_threads=n_threads
        )

        self._matrix_query.query.clear_cache()

        self._most_recent_query_report = {}

    def add_item(self, vector, *, index: int = None, field: str = None) -> int:
        """
        Add a single vector to the database.

        Parameters:
            vector (np.ndarray): The vector to be added.
            index (int, optional, keyword-only): The ID of the vector. If None, a new ID will be generated.
            field (str, optional, keyword-only): The field of the vector. Default is None. If None, the field will be
                set to an empty string.

        Returns:
            int: The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        return self._matrix_serializer.add_item(vector, index=index, field=field)

    def bulk_add_items(self, vectors):
        """
        Bulk add vectors to the database in batches.

        Parameters: vectors (list or tuple): A list or tuple of vectors to be saved. Each vector can be a tuple of (
            vector, id, field).

        Returns:
            list: A list of indices where the vectors are stored.
        """
        return self._matrix_serializer.bulk_add_items(vectors)

    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        This method is required to be called after saving vectors to query them.
        """
        self._matrix_serializer.commit()

    def query(self, vector, k: int | str = 12, *, fields: list = None, subset_indices=None,
              return_similarity=True):
        """
        Query the database for the vectors most similar to the given vector in batches.

        Parameters:
            vector (np.ndarray): The query vector.
            k (int or str): The number of nearest vectors to return. if be 'all', return all vectors.
            fields (list, optional): The target of the vector.
            subset_indices (list, optional): The subset of indices to query.
            return_similarity (bool): Whether to return the similarity scores.Default is True.

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the database is empty.
        """
        import datetime

        self._most_recent_query_report = {}

        self._timer.start()
        if self._use_cache:
            res = self._matrix_query.query(vector=vector, k=k, fields=fields,
                                           subset_indices=subset_indices, index_mode=self._matrix_serializer.index_mode,
                                           distance=self._distance, return_similarity=return_similarity)
        else:
            res = self._matrix_query.query(vector=vector, k=k, fields=fields,
                                           subset_indices=subset_indices, index_mode=self._matrix_serializer.index_mode,
                                           now_time=datetime.datetime.now().timestamp(), distance=self._distance,
                                           return_similarity=return_similarity)

        time_cost = self._timer.last_timestamp_diff()
        self._most_recent_query_report['Database shape'] = self.shape
        self._most_recent_query_report['Query time'] = f"{time_cost :>.5f} s"
        self._most_recent_query_report['Query K'] = k
        self._most_recent_query_report[f'Top {k} results index'] = res[0]
        if return_similarity:
            self._most_recent_query_report[f'Top {k} results similarity'] = res[1]

        return res

    @property
    def shape(self):
        """
        Return the shape of the entire database.

        Returns:
            tuple: The number of vectors and the dimension of each vector in the database.
        """
        return self._matrix_serializer.shape

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
        import gc

        self._matrix_serializer.delete()
        self._matrix_query.query.clear_cache()
        self._matrix_query.delete()

        gc.collect()

    @property
    def query_report_(self):
        """
        Return the most recent query report.
        """
        # print as a pretty string
        # title use bold font
        report = '\n* - MOST RECENT QUERY REPORT -\n'
        for key, value in self._most_recent_query_report.items():
            report += f'| - {key}: {value}\n'

        report += '* - END OF REPORT -\n'

        return report

    @property
    def status_report_(self):
        """
        Return the database report.
        """
        db_report = {'DATABASE STATUS REPORT':{
            'Database shape': (0, self._matrix_serializer.dim) if self._matrix_serializer.IS_DELETED else self.shape,
            'Database last_commit_time': self._matrix_serializer.last_commit_time,
            'Database commit status': self._matrix_serializer.COMMIT_FLAG,
            'Database index_mode': self._matrix_serializer.index_mode,
            'Database distance': self._distance,
            'Database use_cache': self._use_cache,
            'Database reindex_if_conflict': self._matrix_serializer.reindex_if_conflict,
            'Database status': 'DELETED' if self._matrix_serializer.IS_DELETED else 'ACTIVE'
        }}

        return db_report

    def __repr__(self):
        title = "Deleted MinVectorDB object with status: \n"
        report = '\n* - DATABASE STATUS REPORT -\n'
        for key, value in self.status_report_['DATABASE STATUS REPORT'].items():
            report += f'| - {key}: {value}\n'

        return title + report

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.shape[0]

    def is_deleted(self):
        """To check if the database is deleted."""
        return self._matrix_serializer.IS_DELETED
