"""api.py - The MinVectorDB API."""

from min_vec.config import *


class MinVectorDB:
    """
    A class for managing a vector database stored in .mvdb files and computing vectors similarity.
    """

    from spinesUtils.asserts import ParameterValuesAssert, ParameterTypeAssert

    @ParameterTypeAssert({
        'use_cache': bool, 'reindex_if_conflict': bool
    }, func_name='MinVectorDB')
    def __init__(
            self, dim, database_path, n_cluster=8, chunk_size=100_000, distance='cosine',
            index_mode='IVF-FLAT', dtypes='float64',
            use_cache=True, reindex_if_conflict=False, scaler_bits=8
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
            scaler_bits (int): The number of bits for scalar quantization. Default is 8.
                Options are 8, 16, or 32. The default is None, which means no scalar quantization.
                The 8 for 8-bit, 16 for 16-bit, and 32 for 32-bit.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        from spinesUtils.logging import Logger
        from spinesUtils.timer import Timer

        from min_vec.query import DatabaseQuery
        from min_vec.matrix_serializer import MatrixSerializer

        logger = Logger(
            fp=MVDB_LOG_PATH,
            name='MinVectorDB',
            truncate_file=MVDB_TRUNCATE_LOG,
            with_time=MVDB_LOG_WITH_TIME,
            level=MVDB_LOG_LEVEL
        )
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
        # matrix_serializer functions
        self.add_item = self._matrix_serializer.add_item
        self.bulk_add_items = self._matrix_serializer.bulk_add_items
        self.delete = self._matrix_serializer.delete
        self._data_loader = self._matrix_serializer.iterable_dataloader
        self.check_commit = self._matrix_serializer.check_commit
        self._id_filter = self._matrix_serializer.id_filter
        self.commit = self._matrix_serializer.commit

        self._timer = Timer()
        self.use_cache = use_cache
        self.distance = distance

        self._matrix_query = DatabaseQuery(
            matrix_serializer=self._matrix_serializer
        )

        self._matrix_query.query.clear_cache()

        self._most_recent_query_report = {}

    def query(self, vector, k: int | str = 12, *, fields: list = None, normalize: bool = False, subset_indices=None):
        import datetime

        self._most_recent_query_report = {}

        self._timer.start()
        if self.use_cache:
            res = self._matrix_query.query(vector=vector, k=k, fields=fields, normalize=normalize,
                                           subset_indices=subset_indices, index_mode=self._matrix_serializer.index_mode,
                                           distance=self.distance)
        else:
            res = self._matrix_query.query(vector=vector, k=k, fields=fields, normalize=normalize,
                                           subset_indices=subset_indices, index_mode=self._matrix_serializer.index_mode,
                                           now_time=datetime.datetime.now().timestamp(), distance=self.distance)

        time_cost = self._timer.last_timestamp_diff()
        self._most_recent_query_report['Database shape'] = self.shape
        self._most_recent_query_report['Query time'] = f"{time_cost :>.5f} s"
        self._most_recent_query_report['Query vector'] = vector
        self._most_recent_query_report['Query K'] = k
        self._most_recent_query_report['Query fields'] = fields
        self._most_recent_query_report['Query normalize'] = normalize
        self._most_recent_query_report['Query subset_indices'] = subset_indices
        self._most_recent_query_report[f'Top {k} results index'] = res[0]
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

    def _get_n_elements(self, returns, n, from_tail=False):
        """get the first/last n vectors/indices/fields in the database."""
        import numpy as np

        _database = None
        _indices = []
        _fields = []
        for database, indices, fields in self._data_loader(read_chunk_only=False, from_tail=from_tail):
            if _database is None:
                _database = database
            else:
                _database = np.vstack((_database, database))

            _indices.extend(indices)
            _fields.extend(fields)

            if _database.shape[0] >= n:
                break

        if _database is None:
            return None

        if returns == 'database':
            return _database[:n, :]
        elif returns == 'indices':
            return [int(i) for i in _indices[:n]]
        else:
            return _fields[:n]

    @ParameterTypeAssert({'n': int, 'returns': str})
    @ParameterValuesAssert({'returns': ('database', 'indices', 'fields')})
    def head(self, n=10, returns='database'):
        """Return the first n vectors/indices/fields in the database.

        Parameters:
            n (int): The number of vectors to return.
            returns (str): The type of data to return. Options are 'database', 'indices', or 'fields'.

        Returns:
            The first n vectors in the database.
        """
        self.check_commit()

        if self._matrix_serializer.shape[0] == 0:
            return None

        return self._get_n_elements(returns, n, from_tail=False)

    @ParameterTypeAssert({'n': int, 'returns': str})
    @ParameterValuesAssert({'returns': ('database', 'indices', 'fields')})
    def tail(self, n=10, returns='database'):
        """Return the last n vectors/indices/fields in the database.

        Parameters:
            n (int): The number of vectors to return.
            returns (str): The type of data to return. Options are 'database', 'indices', or 'fields'.

        Returns:
            The last n vectors/indices/fields in the database.
        """
        self.check_commit()

        if self._matrix_serializer.shape[0] == 0:
            return None

        return self._get_n_elements(returns, n, from_tail=True)

    def insert_session(self):
        """
        Create a session to insert data, which will automatically commit the data when the session ends.
        """
        from min_vec.session import DatabaseSession

        return DatabaseSession(self)

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
            'Database distance': self.distance,
            'Database use_cache': self.use_cache,
            'Database reindex_if_conflict': self._matrix_serializer.reindex_if_conflict,
            'Database status': 'DELETED' if self._matrix_serializer.IS_DELETED else 'ACTIVE'
        }}

        return db_report

    def __repr__(self):
        title = "MinVectorDB object with status: \n"
        report = '\n* - DATABASE STATUS REPORT -\n'
        for key, value in self.status_report_['DATABASE STATUS REPORT'].items():
            report += f'| - {key}: {value}\n'

        return title + report

    def __str__(self):
        return self.__repr__()
