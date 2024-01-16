import numpy as np
from spinesUtils.asserts import ParameterValuesAssert, ParameterTypeAssert

from min_vec.query import DatabaseQuery
from min_vec.session import DatabaseSession
from min_vec.binary_matrix_serializer import BinaryMatrixSerializer


class MinVectorDB:
    """
    A class for managing a vector database stored in .mvdb files and computing vectors similarity.
    """

    @ParameterTypeAssert({
        'dim': int, 'database_path': str, 'chunk_size': int, 'bloom_filter_size': int, 'device': str
    }, func_name='MinVectorDB')
    @ParameterValuesAssert({
        'database_path': lambda s: s.endswith('.mvdb'),
        'distance': ('cosine', 'L2'),
    }, func_name='MinVectorDB')
    def __init__(self, dim, database_path, n_cluster=8, chunk_size=100_000, dtypes=np.float32, distance='cosine',
                 bloom_filter_size=100_000_000, device='auto') -> None:
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str): Path to the database file.
            chunk_size (int): The size of each data chunk. Default is 100_000.
            dtypes (str): Data type of the vectors.
                Default is 'float32'. Options are 'float32', 'float64', 'float16', 'int32', 'int64', 'int16', 'int8'.
            distance (str): Method for calculating vector distance.
                Options are 'cosine' or 'L2' for Euclidean distance. Default is 'cosine'.
            bloom_filter_size (int): The size of the bloom filter. Default is 100_000_000.
            device (str): The device to use for vector operations.
                Options are 'auto', 'cpu', 'mps', or 'cuda'. Default is 'auto'.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        if chunk_size <= 1:
            raise ValueError('chunk_size must be greater than 1')

        self._binary_matrix_serializer = BinaryMatrixSerializer(
            dim=dim,
            database_path=database_path,
            n_clusters=n_cluster,
            chunk_size=chunk_size,
            dtypes=dtypes,
            bloom_filter_size=bloom_filter_size,
            device=device,
            distance=distance
        )
        # binary_matrix_serializer functions
        self.add_item = self._binary_matrix_serializer.add_item
        self.bulk_add_items = self._binary_matrix_serializer.bulk_add_items
        self.delete = self._binary_matrix_serializer.delete
        self._data_loader = self._binary_matrix_serializer.data_loader
        self.check_commit = self._binary_matrix_serializer.check_commit
        self._id_filter = self._binary_matrix_serializer.id_filter
        self.commit = self._binary_matrix_serializer.commit

        self._matrix_query = DatabaseQuery(
            binary_matrix_serializer=self._binary_matrix_serializer,
            distance=distance,
            device=device,
            dtypes=dtypes,
            chunk_size=chunk_size
        )
        # matrix_query functions
        self.query = self._matrix_query.query

    @property
    def shape(self):
        """
        Return the shape of the entire database.

        Returns:
            tuple: The number of vectors and the dimension of each vector in the database.
        """
        self.check_commit()

        return self._binary_matrix_serializer.database_shape

    def _get_n_elements(self, returns, n, paths):
        _database = None
        _indices = []
        _fields = []
        for database, indices, fields in self._data_loader(paths):
            if _database is None:
                _database = database
            else:
                _database = np.vstack((_database, database))

            _indices.extend(indices)
            _fields.extend(fields)

            if _database.shape[0] >= n:
                break

        if returns == 'database':
            return _database[:n, :]
        elif returns == 'indices':
            return _indices[:n]
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

        if len(self._binary_matrix_serializer.database_cluster_path) == 0 and \
                len(self._binary_matrix_serializer.database_chunk_path) == 0:
            return None

        if len(self._binary_matrix_serializer.database_cluster_path) == 0:
            path = [str(i) for i in self._binary_matrix_serializer.database_chunk_path]
            path = sorted(path)
        else:
            path = [str(i) for i in self._binary_matrix_serializer.database_cluster_path]
            path = sorted(path)

        # print("sorted path: ", path)

        return self._get_n_elements(returns, n, path)

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

        if len(self._binary_matrix_serializer.database_cluster_path) == 0:
            return None

        return self._get_n_elements(returns, n, self._binary_matrix_serializer.database_cluster_path[::-1])

    def insert_session(self):
        """
        Create a session to insert data, which will automatically commit the data when the session ends.
        """
        return DatabaseSession(self)
