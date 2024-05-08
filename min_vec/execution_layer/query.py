"""query.py: this file is used to query the database for the vectors most similar to the given vector."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from min_vec.computational_layer.engines import to_normalize
from min_vec.configs.config import config
from min_vec.computational_layer.engines import cosine_distance
from min_vec.execution_layer.matrix_serializer import MatrixSerializer
from min_vec.utils.utils import QueryResultsCache
from min_vec.core_components.limited_sort import LimitedSorted


class Query:
    """Query the database for the vectors most similar to the given vector."""

    def __init__(self, matrix_serializer: MatrixSerializer, n_threads=10, distance='cosine') -> None:
        """
        Query the database for the vectors most similar to the given vector.

        Parameters:
            matrix_serializer (MatrixSerializer): The database to be queried.
            n_threads (int): The number of threads to use for querying the database.
            distance (str): The distance metric to use for the query.
                .. versionadded:: 0.2.7
        """
        self.matrix_serializer = matrix_serializer

        self.logger = self.matrix_serializer.logger
        # attributes
        self.dtypes = self.matrix_serializer.dtypes
        self.distance = distance
        self.chunk_size = self.matrix_serializer.chunk_size

        self.fields_index = self.matrix_serializer.kv_index

        self.n_threads = n_threads

        self.scaler = getattr(self.matrix_serializer, 'scaler', None)
        if self.scaler is not None and not self.scaler.fitted:
            self.scaler = None

        self.executors = ThreadPoolExecutor(max_workers=self.n_threads)
        self.limited_sorted = LimitedSorted(dim=self.matrix_serializer.dim, dtype=self.dtypes,
                                            scaler=self.scaler, chunk_size=self.chunk_size)

    def update_scaler(self, scaler):
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = getattr(self.matrix_serializer, 'scaler', None)

        if self.scaler is not None and not self.scaler.fitted:
            self.scaler = None

        self.limited_sorted.scaler = self.scaler

    def _query_chunk(self, vector, subset_indices, dataloader, filename):
        """
        Query a single database chunk for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray): The query vector.
            subset_indices (np.ndarray): The indices to filter the numpy array.

        Returns:
            Tuple: The indices and similarity scores of the nearest vectors in the chunk.
        """
        database_chunk, index_chunk = dataloader(filename)

        if subset_indices is not None:
            database_chunk = database_chunk[np.isin(index_chunk, subset_indices, assume_unique=True)]
            index_chunk = index_chunk[np.isin(index_chunk, subset_indices, assume_unique=True)]

        if len(index_chunk) == 0:
            return [], []

        # Distance calculation core code
        scores = cosine_distance(vector, database_chunk)

        if scores.ndim != 1:
            if scores.ndim == 0:
                scores = np.array([scores])
            elif scores.ndim == 2:
                scores = scores.squeeze()

        self.limited_sorted.add(scores, index_chunk, database_chunk)

        return index_chunk, scores

    @QueryResultsCache(config.MVDB_QUERY_CACHE_SIZE)
    def query(self, vector, k=12, query_filter=None, distance=None, **kwargs):
        """
        Query the database for the vectors most similar to the given vector in batches.

        Parameters:
            vector (np.ndarray or list): The query vector.
            k (int): The number of nearest vectors to return.
            query_filter (Filter, optional): The field filter to apply to the query.
            distance (str): The distance metric to use for the query.
                .. versionadded:: 0.2.7

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the database is empty.
        """
        self.limited_sorted.clear()

        distance = distance or self.distance

        if isinstance(vector, list):
            vector = np.array(vector)

        vector = vector.astype(self.dtypes) if vector.dtype != self.dtypes else vector

        vector = to_normalize(vector)

        # set the number of nearest vectors to return
        self.limited_sorted.set_n(k)

        all_index = []

        subset_indices = self.matrix_serializer.kv_index.query(query_filter) if query_filter is not None else None

        def batch_query(is_ivf=True, cid=None, sort=True):
            nonlocal all_index

            filenames = self.matrix_serializer.storage_worker.get_all_files(
                read_type='chunk' if not is_ivf else 'cluster', cluster_id=cid)

            futures = [i for i in self.executors.map(
                lambda x: self._query_chunk(vector, subset_indices, self.matrix_serializer.dataloader, x),
                filenames)
                       if len(i[0]) != 0]

            if not futures:
                return [], []

            if sort:
                return self.limited_sorted.get_top_n(vector=vector, distance=distance)

            index, scores = zip(*futures)

            # if index[0] is iterable, then index is a tuple of numpy ndarray, so we need to flatten it
            if len(index[0].shape) > 0:
                index = np.concatenate(index).ravel()

            all_index.extend(index)

        # if the index mode is FLAT, use FLAT
        if not (self.matrix_serializer.ann_model is not None and self.matrix_serializer.ann_model.fitted):
            return batch_query(is_ivf=False)

        # otherwise, use IVF-FLAT
        cluster_id_sorted = np.argsort(
            -cosine_distance(vector, self.matrix_serializer.ann_model.cluster_centers_)
        ).tolist()

        if self.scaler is not None:
            d_vector = self.scaler.encode(vector)
        else:
            d_vector = vector

        predict_id = self.matrix_serializer.ann_model.predict(d_vector.reshape(1, -1))[0]

        cluster_id_sorted.remove(predict_id)
        cluster_id_sorted.insert(0, predict_id)

        for cluster_id in cluster_id_sorted:
            batch_query(cid=cluster_id, sort=False)

            if len(all_index) >= k:
                break

        return self.limited_sorted.get_top_n(vector=vector, distance=distance)

    def __del__(self):
        self.executors.shutdown(wait=True)
        self.logger.debug('DatabaseQuery executor shutdown.')

    def delete(self):
        self.__del__()
