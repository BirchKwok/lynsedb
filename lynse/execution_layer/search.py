from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from lynse.computational_layer.engines import to_normalize, inner_product_distance
from lynse.configs.config import config
from lynse.execution_layer.cluster_worker import ClusterWorker
from lynse.execution_layer.matrix_serializer import MatrixSerializer
from lynse.utils.utils import SearchResultsCache
from lynse.core_components.limited_sort import LimitedSorted


class Search:
    """Search the database for the vectors most similar to the given vector."""

    def __init__(self, matrix_serializer: 'MatrixSerializer', cluster_worker: 'ClusterWorker', n_threads=10,
                 distance='IP') -> None:
        """
        Search the database for the vectors most similar to the given vector.

        Parameters:
            matrix_serializer (MatrixSerializer): The database to be queried.
            n_threads (int): The number of threads to use for searching the database.
            distance (str): The distance metric to use for the search.
                .. versionadded:: 0.2.7
        """
        self.matrix_serializer = matrix_serializer
        self.cluster_worker = cluster_worker

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

    def update_scaler(self, scaler):
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = getattr(self.matrix_serializer, 'scaler', None)

        if self.scaler is not None and not self.scaler.fitted:
            self.scaler = None

    def _search_chunk(self, vector, subset_indices, filename, limited_sorted, distance_func, ivf_subset_indices=None):
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

    @SearchResultsCache(config.LYNSE_SEARCH_CACHE_SIZE)
    def search(self, vector, k=12, search_filter=None, distance=None, normalize=True, **kwargs):
        """
        Search the database for the vectors most similar to the given vector in batches.

        Parameters:
            vector (np.ndarray or list): The search vector.
            k (int): The number of nearest vectors to return.
            search_filter (Filter, optional): The field filter to apply to the search.
            distance (str): The distance metric to use for the search.
                .. versionadded:: 0.2.7
            normalize (bool): Whether to normalize the search vector.
                .. versionadded:: 0.3.6

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the database is empty.
        """
        limited_sorted = LimitedSorted(scaler=self.scaler, n=k)

        distance = distance or self.distance

        if isinstance(vector, list):
            vector = np.array(vector)

        vector = vector.astype(self.dtypes) if vector.dtype != self.dtypes else vector

        vector = to_normalize(vector) if normalize else vector

        subset_indices = None if search_filter is None else self.matrix_serializer.kv_index.query(search_filter)

        filenames = self.matrix_serializer.storage_worker.get_all_files()

        def batch_search(is_ivf=True, cid=None, sort=True, use_jax=False):
            nonlocal subset_indices, limited_sorted, filenames

            ivf_subset_indices = None if not is_ivf else self.cluster_worker.ivf_index.get_entries(cid)

            if ivf_subset_indices is not None:
                filenames = [filename for filename in filenames if filename in ivf_subset_indices]

            map_fuc = self.executors.map if not is_ivf else map
            distance_func = partial(inner_product_distance, use='jax' if use_jax else 'np')

            _ = [
                i for i in map_fuc(
                    lambda x: self._search_chunk(
                        vector, subset_indices, x,
                        limited_sorted,
                        distance_func,
                        ivf_subset_indices[x] if ivf_subset_indices is not None else None
                    ), filenames
                )
            ]

            if sort:
                return limited_sorted.get_top_n(vector=vector, distance=distance)

        # if the index mode is FLAT, use FLAT
        if not (self.cluster_worker.ann_model is not None and self.cluster_worker.ann_model.fitted):
            if self.matrix_serializer.storage_worker.get_shape()[0] >= 500_0000:
                return batch_search(is_ivf=False, use_jax=True)
            else:
                return batch_search(is_ivf=False, use_jax=False)

        # otherwise, use IVF-FLAT
        cluster_id_sorted = np.argsort(
            -inner_product_distance(vector, self.cluster_worker.ann_model.cluster_centers_, use='np')
        ).tolist()

        if self.scaler is not None:
            d_vector = self.scaler.encode(vector)
        else:
            d_vector = vector

        predict_id = self.cluster_worker.ann_model.predict(d_vector.reshape(1, -1))[0]

        cluster_id_sorted.remove(predict_id)
        cluster_id_sorted.insert(0, predict_id)

        for cluster_id in cluster_id_sorted:
            batch_search(cid=cluster_id, sort=False, is_ivf=True)

            if len(limited_sorted) >= k:
                break

        return limited_sorted.get_top_n(vector=vector, distance=distance)

    def __del__(self):
        self.executors.shutdown(wait=True)
        self.logger.debug('Search executor shutdown.')

    def delete(self):
        self.__del__()
