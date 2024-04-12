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

    def __init__(self, matrix_serializer: MatrixSerializer, n_threads=10) -> None:
        """
        Query the database for the vectors most similar to the given vector.

        Parameters:
            matrix_serializer (MatrixSerializer): The database to be queried.
            n_threads (int): The number of threads to use for querying the database.
        """
        self.matrix_serializer = matrix_serializer

        self.logger = self.matrix_serializer.logger
        # attributes
        self.dtypes = self.matrix_serializer.dtypes
        self.distance = self.matrix_serializer.distance
        self.distance_func = cosine_distance if self.distance == 'cosine' else euclidean_distance
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
            if isinstance(field, str):
                field = [self.fields_mapper.fields_str_mapper.get(field, -1)]
            else:
                field = [self.fields_mapper.fields_str_mapper.get(f, -1) for f in field]

            field_condition = np.isin(vector_field, field)

        if subset_indices is not None:
            subset_indices = sorted(list(set(subset_indices)))
            si_condition = np.isin(index_chunk, subset_indices)

        if field_condition is not None and si_condition is not None:
            condition = np.logical_and(field_condition, si_condition)
            database_chunk = database_chunk[condition]
            index_chunk = index_chunk[condition]
        elif field_condition is not None:
            condition = field_condition
            database_chunk = database_chunk[condition]
            index_chunk = index_chunk[condition]
        elif si_condition is not None:
            condition = si_condition
            database_chunk = database_chunk[condition]
            index_chunk = index_chunk[condition]

        if len(index_chunk) == 0:
            return [], []

        if self.distance != 'cosine' and self.scaler is not None:
            database_chunk = self.scaler.decode(database_chunk)

        # Distance calculation core code
        scores = self.distance_func(vector, database_chunk)

        if scores.ndim == 0:
            scores = np.array([scores])
        elif scores.ndim == 2:
            scores = scores.squeeze()

        if limited_sorted is not None and self.distance == 'cosine':
            limited_sorted.add(scores, index_chunk, database_chunk)

        return index_chunk, scores

    @QueryVectorCache(config.MVDB_QUERY_CACHE_SIZE)
    def query(self, vector, k = 12,
              fields = None, subset_indices=None, return_similarity=False, **kwargs):
        """
        Query the database for the vectors most similar to the given vector in batches.

        Parameters:
            vector (np.ndarray): The query vector.
            k (int or str): The number of nearest vectors to return. if be 'all', return all vectors.
            fields (list, optional): The target of the vector.
            subset_indices (list, optional): The subset of indices to query.
            return_similarity (bool): Whether to return the similarity scores of the nearest vectors.

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the database is empty.
        """
        self.logger.debug(f'Query vector: {vector.tolist()}')
        self.logger.debug(f'Query k: {k}')
        self.logger.debug(f'Query fields: {fields}')
        self.logger.debug(f'Query subset_indices: {subset_indices}')

        raise_if(TypeError, not isinstance(k, int) and not (isinstance(k, str) and k != 'all'),
                 'k must be int or "all".')
        raise_if(ValueError, k <= 0, 'k must be greater than 0.')
        raise_if(ValueError, not isinstance(fields, list) and fields is not None,
                 'fields must be list or None.')
        raise_if(ValueError, not isinstance(subset_indices, list) and subset_indices is not None,
                 'subset_indices must be list or None.')
        raise_if(ValueError, vector is None, 'vector must be not None.')
        raise_if(ValueError, len(vector) != self.matrix_serializer.shape[1],
                 'vector must be same dim with database.')
        raise_if(ValueError, not isinstance(vector, np.ndarray), 'vector must be np.ndarray.')
        raise_if(ValueError, vector.ndim != 1, 'vector must be 1d array.')

        if self.matrix_serializer.shape[0] == 0:
            raise ValueError('database is empty.')

        if k > self.matrix_serializer.shape[0]:
            k = self.matrix_serializer.shape[0]

        vector = vector.astype(self.dtypes) if vector.dtype != self.dtypes else vector

        vector = to_normalize(vector)

        if self.scaler is not None and return_similarity and self.distance == 'cosine':
            limited_sorted = LimitedSorted(vector, k, self.scaler)
        else:
            limited_sorted = None

        all_scores = []
        all_index = []

        def sort_results(all_s, all_i):
            all_scores_i = np.array(all_s)
            all_index_i = np.array(all_i)

            top_k_indices = argsort_topk(self.is_reversed * all_scores_i, k)

            return all_index_i[top_k_indices], all_scores_i[top_k_indices]

        def batch_query(vector, fields=None, subset_indices=None, is_ivf=True, cluster_id=None, sort=True):
            nonlocal all_scores, all_index, limited_sorted

            dataloader = self.matrix_serializer.cluster_dataloader(cluster_id, mode='lazy') \
                if is_ivf and self.matrix_serializer.ann_model \
                else self.matrix_serializer.iterable_dataloader(mode='lazy')

            futures = [i for i in self.executors.map(
                lambda x: self._query_chunk(x[0], x[1], x[2], vector, fields,
                                            subset_indices, limited_sorted), dataloader)
                       if len(i[0]) != 0]

            if not futures:
                return [], []

            if return_similarity and self.scaler is not None and sort and self.distance == 'cosine':
                return limited_sorted.get_top_n()

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
        if self.distance != 'cosine' and self.scaler is not None:
            cluster_distances = self.distance_func(vector,
                                                   self.scaler.decode(
                                                       self.matrix_serializer.ann_model.cluster_centers_)).squeeze()
        else:
            cluster_distances = self.distance_func(vector, self.matrix_serializer.ann_model.cluster_centers_).squeeze()

        cluster_id_sorted = np.argsort(cluster_distances)[::-1] if self.distance == 'cosine' else np.argsort(
            cluster_distances)

        for cluster_id in cluster_id_sorted:
            batch_query(vector, fields, subset_indices, cluster_id=str(cluster_id), sort=False)

            if len(all_index) >= k:
                break

        if return_similarity and self.scaler is not None and self.distance == 'cosine':
            return limited_sorted.get_top_n()

        return sort_results(all_scores, all_index)

    def __del__(self):
        self.executors.shutdown(wait=True)
        self.logger.debug('DatabaseQuery executor shutdown.')

    def delete(self):
        self.__del__()
