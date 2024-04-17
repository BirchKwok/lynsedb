"""query.py: this file is used to query the database for the vectors most similar to the given vector."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from spinesUtils.asserts import raise_if

from min_vec.computational_layer.engines import to_normalize
from min_vec.configs.config import config
from min_vec.computational_layer.engines import argsort_topk, cosine_distance, euclidean_distance
from min_vec.execution_layer.matrix_serializer import MatrixSerializer
from min_vec.utils.utils import QueryVectorCache
from min_vec.data_structures.limited_sort import LimitedSorted


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
        self.is_reversed = -1 if self.distance == 'cosine' else 1
        self.chunk_size = self.matrix_serializer.chunk_size

        self.fields_mapper = self.matrix_serializer.fields_mapper

        self.n_threads = n_threads

        self.scaler = getattr(self.matrix_serializer, 'scaler', None)

        self.executors = ThreadPoolExecutor(max_workers=self.n_threads)

    def _query_chunk(self, database_chunk, index_chunk, vector_field, vector, field, subset_indices, limited_sorted):
        """
        Query a single database chunk for the vectors most similar to the given vector.

        Parameters:
            database_chunk (np.ndarray): The database chunk to be queried.
            index_chunk (np.ndarray): The indices of the vectors in the database chunk.
            vector_field (np.ndarray): The field of the vectors.
            vector (np.ndarray): The query vector.
            field (str or list, optional): The target field for filtering the vectors.
            subset_indices (list, optional): The subset of indices to query.
            limited_sorted (HeapqSorted): The heapq object to store the top k nearest vectors.

        Returns:
            Tuple: The indices and similarity scores of the nearest vectors in the chunk.
        """
        field_condition = None
        si_condition = None

        if field is not None:
            field = [self.fields_mapper.fields_str_mapper.get(f, -1) for f in field]

            field_condition = np.isin(vector_field, field)

        if subset_indices:
            si_condition = np.isin(index_chunk, subset_indices)

        if field_condition is not None and si_condition is not None:
            condition = np.logical_and(field_condition, si_condition)
        elif field_condition is not None:
            condition = field_condition
        elif si_condition is not None:
            condition = si_condition
        else:
            condition = False

        if condition is not False:
            database_chunk = database_chunk[condition]
            index_chunk = index_chunk[condition]

        if len(index_chunk) == 0:
            return [], []

        # Distance calculation core code
        scores = cosine_distance(vector, database_chunk)

        if scores.ndim != 1:
            if scores.ndim == 0:
                scores = np.array([scores])
            elif scores.ndim == 2:
                scores = scores.squeeze()

        if limited_sorted is not None:
            limited_sorted.add(scores, index_chunk, database_chunk)

        return index_chunk, scores

    @QueryVectorCache(config.MVDB_QUERY_CACHE_SIZE)
    def query(self, vector, k=12,
              fields=None, subset_indices=None, distance=None, return_similarity=False, **kwargs):
        """
        Query the database for the vectors most similar to the given vector in batches.

        Parameters:
            vector (np.ndarray): The query vector.
            k (int): The number of nearest vectors to return.
            fields (list, optional): The target of the vector.
            subset_indices (list, optional): The subset of indices to query.
            distance (str): The distance metric to use for the query.
                .. versionadded:: 0.2.7
            return_similarity (bool): Whether to return the similarity scores of the nearest vectors.
                .. versionadded:: 0.2.5

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the database is empty.
        """
        if distance is not None:
            is_reversed = -1 if distance == 'cosine' else 1
        else:
            is_reversed = self.is_reversed
            distance = self.distance

        vector = vector.astype(self.dtypes) if vector.dtype != self.dtypes else vector

        vector = to_normalize(vector)

        if return_similarity:
            limited_sorted = LimitedSorted(vector, k, self.scaler, distance=distance, chunk_size=self.chunk_size)
        else:
            limited_sorted = None

        all_scores = []
        all_index = []

        def sort_results(all_s, all_i):
            return np.array(all_i)[argsort_topk(is_reversed * np.array(all_s), k)], None

        def batch_query(vector, fields=None, subset_indices=None, is_ivf=True, cluster_id=None, sort=True):
            nonlocal all_scores, all_index, limited_sorted

            dataloader = self.matrix_serializer.cluster_dataloader(cluster_id, mode='lazy') \
                if is_ivf and self.matrix_serializer.ann_model \
                else self.matrix_serializer.iterable_dataloader(mode='lazy')

            futures = [i for i in self.executors.map(
                lambda x: self._query_chunk(x[0], x[1], x[2], vector, fields,
                                            subset_indices, limited_sorted), dataloader)
                       if len(i[0]) != 0]

            if return_similarity and sort:
                return limited_sorted.get_top_n()

            if not futures:
                return None, None

            index, scores = zip(*futures)

            # if index[0] is iterable, then index is a tuple of numpy ndarray, so we need to flatten it
            if len(index[0].shape) > 0:
                index = np.concatenate(index).ravel()
                scores = np.concatenate(scores).ravel()

            all_scores.extend(scores)
            all_index.extend(index)

            return sort_results(all_scores, all_index) if sort else None

        # if the index mode is FLAT, use FLAT
        if not (self.matrix_serializer.ann_model is not None and self.matrix_serializer.ann_model.fitted):
            return batch_query(vector, fields, subset_indices, False)

        # otherwise, use IVF-FLAT
        cluster_id_sorted = np.argsort(
            -cosine_distance(vector, self.matrix_serializer.ann_model.cluster_centers_)
        )

        for cluster_id in cluster_id_sorted:
            batch_query(vector, fields, subset_indices, cluster_id=cluster_id, sort=False)

            if len(all_index) >= k:
                break

        if return_similarity:
            return limited_sorted.get_top_n()

        return sort_results(all_scores, all_index)

    def __del__(self):
        self.executors.shutdown(wait=True)
        self.logger.debug('DatabaseQuery executor shutdown.')

    def delete(self):
        self.__del__()
