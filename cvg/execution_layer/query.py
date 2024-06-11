from concurrent.futures import ThreadPoolExecutor
import numpy as np

from cvg.execution_layer.matrix_serializer import MatrixSerializer


class Query:
    """Search the database for the vectors most similar to the given vector."""

    def __init__(self, matrix_serializer: 'MatrixSerializer', n_threads=10) -> None:
        """
        Query the database for the vectors by given ID or filter.

        Parameters:
            matrix_serializer (MatrixSerializer): The database to be queried.
        """
        self.matrix_serializer = matrix_serializer

        self.logger = self.matrix_serializer.logger

        # attributes
        self.fields_index = self.matrix_serializer.kv_index

        self.n_threads = n_threads

        self.executors = ThreadPoolExecutor(max_workers=self.n_threads)

    def update_scaler(self, scaler):
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = getattr(self.matrix_serializer, 'scaler', None)

        if self.scaler is not None and not self.scaler.fitted:
            self.scaler = None

    def _query_chunk(self, vector, subset_indices, filename, limited_sorted, distance_func, ivf_subset_indices=None):
        """
        Search a single database chunk for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray): The search vector.
            subset_indices (np.ndarray): The indices to filter the numpy array.

        Returns:
            Tuple: The indices and similarity scores of the nearest vectors in the chunk.
        """
        if ivf_subset_indices is not None:
            if subset_indices is None:
                subset_indices = ivf_subset_indices
            else:
                subset_indices = np.intersect1d(subset_indices, ivf_subset_indices)

        if subset_indices is not None:
            database_chunk, index_chunk = self.matrix_serializer.storage_worker.read_by_idx(filename,
                                                                                            idx=subset_indices)
        else:
            database_chunk, index_chunk = self.matrix_serializer.dataloader(filename)

        if len(index_chunk) == 0:
            return [], [], []

        # Distance calculation core code
        scores = distance_func(vector, database_chunk)

        if scores.ndim != 1:
            if scores.ndim == 0:
                scores = np.array([scores])
            elif scores.ndim == 2:
                scores = scores.squeeze()

        limited_sorted.add(scores, index_chunk, database_chunk)

    def query(self, query_filter=None, k=12, **kwargs):
        """
        Search the database for the vectors most similar to the given vector in batches.

        Parameters:
            query_filter (Filter, optional): The field filter to apply to the search.
            k (int): The number of nearest vectors to return.

        Returns:
            dict: The indices and similarity scores of the nearest vectors in the database.

        Raises:
            ValueError: If the database is empty.
        """
        ...

    def __del__(self):
        self.executors.shutdown(wait=True)
        self.logger.debug('Search executor shutdown.')

    def delete(self):
        self.__del__()
