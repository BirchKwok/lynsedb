"""query.py: this file is used to query the database for the vectors most similar to the given vector."""


class DatabaseQuery:
    from min_vec.utils import vectors_cache

    def __init__(self, matrix_serializer) -> None:
        """
        Query the database for the vectors most similar to the given vector.

        Parameters:
            matrix_serializer (MatrixSerializer): The database to be queried.
        """
        from min_vec.engines import cosine_distance, euclidean_distance

        self.matrix_serializer = matrix_serializer

        self.logger = self.matrix_serializer.logger
        # attributes
        self.dtypes = self.matrix_serializer.dtypes
        self.distance = self.matrix_serializer.distance
        self.distance_func = cosine_distance if self.distance == 'cosine' else euclidean_distance
        self.is_reversed = -1 if self.distance == 'cosine' else 1
        self.chunk_size = self.matrix_serializer.chunk_size
        self.index_mode = self.matrix_serializer.index_mode if (self.matrix_serializer.ann_model is not None
                                                                and self.matrix_serializer.ann_model.fitted) else 'FLAT'

        self.database = []
        self.index = []
        self.vector_field = []

    def _query_chunk(self, database_chunk, index_chunk, vector_field, vector, field, subset_indices):
        """
        Query a single database chunk for the vectors most similar to the given vector.

        Parameters:
            database_chunk (np.ndarray): The database chunk to be queried.
            index_chunk (np.ndarray): The indices of the vectors in the database chunk.
            vector (np.ndarray): The query vector.
            field (str or list, optional): The target field for filtering the vectors.
            subset_indices (list, optional): The subset of indices to query.
            vector_field (np.ndarray): The field of the vectors.

        Returns:
            Tuple: The indices and similarity scores of the nearest vectors in the chunk.
        """
        import numpy as np

        if field is not None:
            if isinstance(field, str):
                field = [field]

            condition = np.isin(vector_field, field)
            database_chunk = database_chunk[condition]
            index_chunk = np.array(index_chunk)[condition]

        if subset_indices is not None:
            subset_indices = list(set(subset_indices))

            condition = np.isin(index_chunk, subset_indices)
            database_chunk = database_chunk[condition]
            index_chunk = np.array(index_chunk)[condition]

        if len(index_chunk) == 0:
            return [], []

        # Distance calculation core code
        scores = self.distance_func(database_chunk, vector).squeeze()

        if scores.ndim == 0:
            scores = [scores]

        return index_chunk, scores

    @vectors_cache(10000)
    def query(self, vector, k: int | str = 12,
              fields: list = None, normalize: bool = False, subset_indices=None, **kwargs):
        """
        Query the database for the vectors most similar to the given vector in batches.

        Parameters:
            vector (np.ndarray): The query vector.
            k (int or str): The number of nearest vectors to return. if be 'all', return all vectors.
            fields (list, optional): The target of the vector.
            normalize (bool): Whether to normalize the input vector.
            subset_indices (list, optional): The subset of indices to query.

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the database is empty.
        """

        import numpy as np
        from spinesUtils.asserts import raise_if

        from min_vec.engines import to_normalize

        self.logger.debug(f'Query vector: {vector.tolist()}')
        self.logger.debug(f'Query k: {k}')
        self.logger.debug(f'Query fields: {fields}')
        self.logger.debug(f'Query normalize: {normalize}')
        self.logger.debug(f'Query subset_indices: {subset_indices}')

        raise_if(TypeError, not isinstance(k, int) and not (isinstance(k, str) and k != 'all'),
                 'k must be int or "all".')
        raise_if(ValueError, k <= 0, 'k must be greater than 0.')
        raise_if(ValueError, not isinstance(fields, list) and fields is not None,
                 'fields must be list or None.')
        raise_if(ValueError, not isinstance(subset_indices, list) and subset_indices is not None,
                 'subset_indices must be list or None.')
        raise_if(TypeError, not isinstance(normalize, bool), 'normalize must be bool.')
        raise_if(ValueError, vector is None, 'vector must be not None.')
        raise_if(ValueError, len(vector) != self.matrix_serializer.shape[1],
                 'vector must be same dim with database.')
        raise_if(ValueError, not isinstance(vector, np.ndarray), 'vector must be np.ndarray.')
        raise_if(ValueError, vector.ndim != 1, 'vector must be 1d array.')

        # fields = np.array(fields) if fields is not None else fields
        if self.matrix_serializer.shape[0] == 0:
            raise ValueError('database is empty.')

        if k > self.matrix_serializer.shape[0]:
            k = self.matrix_serializer.shape[0]

        vector = vector.astype(self.dtypes) if vector.dtype != self.dtypes else vector

        vector = to_normalize(vector) if normalize else vector

        vector = vector.reshape(1, -1) if vector.ndim == 1 else vector

        all_scores = []
        all_index = []
        unique_indices = set()
        sorted_indices, sorted_scores = [], []

        def sort_results(all_s, all_i):
            all_scores_i = np.array(all_s)
            all_index_i = np.array(all_i)
            top_k_indices = np.argsort(self.is_reversed * all_scores_i)[:k]

            return all_index_i[top_k_indices], all_scores_i[top_k_indices]

        def batch_query(vector, fields=None, subset_indices=None, is_ivf=True, cluster_id=None):
            nonlocal all_scores, all_index

            dataloader = self.matrix_serializer.cluster_dataloader(cluster_id) \
                if is_ivf and self.matrix_serializer.ann_model \
                else self.matrix_serializer.iterable_dataloader(open_for_only_read=True)

            futures = list(filter(lambda x: len(x[0]) != 0, map(
                lambda x: self._query_chunk(x[0], x[1], x[2], vector, fields, subset_indices),
                dataloader
            )))

            if len(futures) == 0:
                return [], []

            index, scores = zip(*futures)
            # if index[0] is iterable, then index is a tuple of numpy ndarray, so we need to flatten it
            if len(index[0].shape) > 0:
                index = [item for sublist in index for item in sublist]
                scores = [item for sublist in scores for item in sublist]

            index = np.array(index)
            scores = np.array(scores)

            if len(index) != 0:
                new_index = index[~np.isin(index, list(unique_indices))]
                new_scores = scores[~np.isin(index, list(unique_indices))]
                all_scores.extend(new_scores)
                all_index.extend(new_index)
                unique_indices.update(new_index)

            del futures

            all_scores_inside = np.array(all_scores)
            all_index_inside = np.array(all_index)

            return sort_results(all_scores_inside, all_index_inside)

        # if the index mode is FLAT, use FLAT
        if self.index_mode == 'FLAT':
            return batch_query(vector, fields, subset_indices, False)

        # otherwise, use IVF-FLAT
        cluster_distances = self.distance_func(vector, self.matrix_serializer.ann_model.cluster_centers_.T).squeeze()

        cluster_id_sorted = np.argsort(cluster_distances)[::-1]

        for cluster_id in cluster_id_sorted:
            batch_query(vector, fields, subset_indices, cluster_id=cluster_id)

            if len(all_index) >= k:
                break

        return sort_results(np.array(all_scores), np.array(all_index))
