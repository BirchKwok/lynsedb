import os

import numpy as np
from spinesUtils.asserts import ParameterValuesAssert, ParameterTypeAssert
from concurrent.futures import ThreadPoolExecutor

from min_vec.engine import cosine_distance, euclidean_distance, to_normalize, get_device


class DatabaseQuery:
    def __init__(self, binary_matrix_serializer, distance='cosine', device='auto', dtypes=np.float32,
                 chunk_size=10000) -> None:
        self.binary_matrix_serializer = binary_matrix_serializer
        
        # attributes
        self.dtypes = dtypes
        self.distance = distance
        self.distance_func = cosine_distance if distance == 'cosine' else euclidean_distance
        self.is_reversed = -1 if distance == 'cosine' else 1
        self.device = get_device(device)
        self.chunk_size = chunk_size

    def _query_chunk(self, database_chunk, index_chunk, vector, field, subset_indices, vector_field):
        """
        Query a single database chunk for the vectors most similar to the given vector.

        Parameters:
            database_chunk (np.ndarray): The database chunk to be queried.
            index_chunk (np.ndarray): The indices of the vectors in the database chunk.
            vector (np.ndarray): The query vector.
            field (str or list, optional): The target field for filtering the vectors.
            subset_indices (list, optional): The subset of indices to query.

        Returns:
            Tuple: The indices and similarity scores of the nearest vectors in the chunk.
        """
        if field is not None:
            if isinstance(field, str):
                field = [field]

            database_chunk = database_chunk[np.isin(vector_field, field)]
            index_chunk = index_chunk[np.isin(vector_field, field)]

        if subset_indices is not None:
            subset_indices = list(set(subset_indices))
            database_chunk = database_chunk[np.isin(index_chunk, subset_indices)]
            index_chunk = index_chunk[np.isin(index_chunk, subset_indices)]

        if len(index_chunk) == 0:
            return [], []

        # Distance calculation core code
        if self.distance == 'cosine':
            scores = cosine_distance(database_chunk, vector, device=self.device).squeeze()
        else:
            scores = euclidean_distance(database_chunk, vector, device=self.device).squeeze()

        if scores.ndim == 0:
            scores = [scores]

        return index_chunk, scores


    @ParameterValuesAssert({'vector': lambda s: s.ndim == 1})
    @ParameterTypeAssert({
        'vector': np.ndarray, 'k': int, 'fields': (None, list),
        'normalize': bool, 'subset_indices': (None, list)
    })
    def query(self, vector, k: int = 12, fields: str | list = None, normalize: bool = False, subset_indices=None):
        """
        Query the database for the vectors most similar to the given vector in batches.

        Parameters:
            vector (np.ndarray): The query vector.
            k (int): The number of nearest vectors to return.
            fields (str or list, optional): The target of the vector.
            normalize (bool): Whether to normalize the input vector.
            subset_indices (list, optional): The subset of indices to query.

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the database is empty.
        """
        self.binary_matrix_serializer.check_commit()

        if (len(self.binary_matrix_serializer.database_cluster_path) == 0 and
                len(self.binary_matrix_serializer.database_chunk_path) == 0):
            raise ValueError('database is empty.')

        if k > self.binary_matrix_serializer.database_shape[0]:
            k = self.binary_matrix_serializer.database_shape[0]

        vector = vector.astype(self.dtypes)

        vector = to_normalize(vector) if normalize else vector
        vector = vector.reshape(1, -1)

        all_scores = []
        all_index = []

        def batch_query(vec_path, vector, k: int = 12, fields: str | list = None, subset_indices=None):
            nonlocal all_scores, all_index
            batch_size = 10 if self.chunk_size > 100000 else 50

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for i in range(0, len(vec_path), batch_size):
                    batch_paths = vec_path[i:i + batch_size]
                    futures = [
                        executor.submit(self._query_chunk, database, index, vector,
                                        fields, subset_indices, vector_field)
                        for database, index, vector_field in self.binary_matrix_serializer.data_loader(batch_paths)]

                    for future in futures:
                        index, scores = future.result()
                        all_scores.extend(scores)
                        all_index.extend(index)

                    del batch_paths, futures

            all_scores_inside = np.asarray(all_scores)
            all_index_inside = np.asarray(all_index)
            top_k_indices = np.argsort(self.is_reversed * all_scores_inside)[:k]

            return all_index_inside, all_scores_inside, top_k_indices

        if len(self.binary_matrix_serializer.database_cluster_path) == 0:
            all_index_inside, all_scores_inside, top_k_indices =\
                batch_query(self.binary_matrix_serializer.database_chunk_path, vector, k, fields, subset_indices)
            if len(top_k_indices) == k:
                return all_index_inside[top_k_indices], all_scores_inside[top_k_indices]
        else:
            vector_cluster_id = self.binary_matrix_serializer.ann_model.predict(vector)[0]

            topk = 0
            searched_id = set()
            while topk < k:
                _, _, vector_cluster_paths = self.binary_matrix_serializer.ivf_index.search(
                    vector_cluster_id, fields=fields, indices=subset_indices
                )
                searched_id.add(vector_cluster_id)

                vector_cluster_paths = list(set(vector_cluster_paths))

                all_index_inside, all_scores_inside, top_k_indices = \
                    batch_query(vector_cluster_paths, vector, k, fields, subset_indices)

                if len(top_k_indices) == k:
                    return all_index_inside[top_k_indices], all_scores_inside[top_k_indices]

                min_dis = 1e10

                for cluster_id, cluster_center in enumerate(self.binary_matrix_serializer.ann_model.cluster_centers_):
                    if cluster_id in searched_id:
                        continue

                    dis = self.distance_func(vector, cluster_center)
                    if dis < min_dis:
                        min_dis = dis
                        vector_cluster_id = cluster_id