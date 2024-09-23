from concurrent.futures import ThreadPoolExecutor

import numpy as np
from lynse.computational_layer.engines import inner_product
from spinesUtils.asserts import check_has_param

from ..configs.config import config
from ..core_components.fields_cache import ExpressionParser
from ..execution_layer.matrix_serializer import MatrixSerializer
from ..utils.utils import SearchResultsCache
from ..core_components.limited_sort import LimitedSorted


class Search:
    """Search the database for the vectors most similar to the given vector."""

    def __init__(self,
                 matrix_serializer: 'MatrixSerializer',
                 n_threads=10) -> None:
        """
        Search the database for the vectors most similar to the given vector.

        Parameters:
            matrix_serializer (MatrixSerializer): The database to be queried.
            n_threads (int): The number of threads to use for searching the database.
        """
        self.matrix_serializer = matrix_serializer
        self.logger = self.matrix_serializer.logger
        # attributes
        self.dtypes = self.matrix_serializer.dtypes
        self.chunk_size = self.matrix_serializer.chunk_size

        self.fields_index = self.matrix_serializer.field_index

        self.n_threads = n_threads

        self.executors = ThreadPoolExecutor(max_workers=self.n_threads)

    def _narrow_down_ids(self, search_filter):
        """
        Narrow down the search results by applying the filter.

        Parameters:
            search_filter (Filter): The filter to apply to the search.

        Returns:
            np.ndarray: The narrowed down indices.
        """
        return None if search_filter is None else self.matrix_serializer.field_index.query(search_filter)

    def _search_chunk(self, vector, subset_indices, filename, limited_sorted, distance_func, ivf_subset_indices=None,
                      topk=10):
        """
        Search a single database chunk for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray): The search vector.
            subset_indices (np.ndarray): The indices to filter the numpy array.
            filename (str): The name of the database chunk to search.
            limited_sorted (LimitedSorted): The object to store the nearest vectors.
            distance_func (function): The distance function to use for the search.
            ivf_subset_indices (np.ndarray): The indices to filter the numpy array in the IVF index.
            topk (int): The number of nearest vectors to return.

        Returns:
            Tuple: The indices and similarity scores of the nearest vectors in the chunk.
        """
        if ivf_subset_indices is not None:
            if subset_indices is None:
                subset_indices = ivf_subset_indices
            else:
                subset_indices = np.intersect1d(subset_indices, ivf_subset_indices)

        cache = self.matrix_serializer.storage_worker.dataloader.cache
        if subset_indices is not None:
            if cache is None or filename not in cache:
                database_chunk, index_chunk = self.matrix_serializer.storage_worker.mmap_read(filename)
                filter_indices = np.isin(index_chunk, subset_indices)
                database_chunk = database_chunk[filter_indices]
                index_chunk = index_chunk[filter_indices]
            else:
                database_chunk, index_chunk = self.matrix_serializer.storage_worker.dataloader.cache[filename]
                filter_indices = np.isin(index_chunk, subset_indices)
                database_chunk = database_chunk[filter_indices]
                index_chunk = index_chunk[filter_indices]
        else:
            if cache is None or filename not in cache:
                database_chunk, index_chunk = self.matrix_serializer.storage_worker.mmap_read(filename)
            else:
                database_chunk, index_chunk = self.matrix_serializer.storage_worker.dataloader.cache[filename]

        if len(index_chunk) == 0:
            return [], [], []

        # Distance calculation core code
        topk = min(topk, len(index_chunk))  # make sure topk is not larger than the chunk size
        res = distance_func(original_vec=vector, encoded_data=database_chunk, top_k=topk)

        if len(res) == 2 and isinstance(res[0], np.ndarray) and isinstance(res[1], np.ndarray):
            ids, scores = res
            index_chunk = index_chunk[ids]
        else:
            scores = res, None

        limited_sorted.add(scores, index_chunk)

    def _flat_search(self, vector, k, search_filter):
        """
        Search the database for the vectors most similar to the given vector in FLAT mode.

        Parameters:
            vector (np.ndarray): The search vector.
            k (int): The number of nearest vectors to return.
            search_filter (Filter): The field filter to apply to the search.

        Returns:
            Tuple: If return_fields is True, the indices, similarity scores,
                    and fields of the nearest vectors in the database.
                Otherwise, the indices and similarity scores of the nearest vectors in the database.
        """
        vector = vector.astype(self.dtypes) if vector.dtype != self.dtypes else vector

        subset_indices = self._narrow_down_ids(search_filter)

        npy_filenames = self.matrix_serializer.storage_worker.get_all_files(separate=False)

        limited_sorted = LimitedSorted(n=k)

        def batch_search(is_ivf=True, cid=None, directly_return=True):
            nonlocal subset_indices, npy_filenames

            ivf_subset_indices = None if not is_ivf else self.matrix_serializer.indexer.ivf.ivf_index.get_entries(cid)

            distance_func = self.matrix_serializer.indexer.index.search

            if is_ivf:
                npy_filenames = list(filter(lambda x: x in ivf_subset_indices, npy_filenames))

            if npy_filenames:
                _ = [
                    i for i in map(
                        lambda x: self._search_chunk(
                            vector, subset_indices, x,
                            limited_sorted,
                            distance_func,
                            ivf_subset_indices[x] if ivf_subset_indices is not None else None,
                            topk=k
                        ), npy_filenames
                    )
                ]

            if directly_return:
                res_ids, res_scores = limited_sorted.get_top_n()

                return (
                    res_ids, res_scores if self.matrix_serializer.indexer.index_mode != 'Flat-IP' else -1 * res_scores
                )

        # if the index mode is FLAT, use FLAT
        if self.matrix_serializer.indexer.index_mode.startswith('Flat'):
            return batch_search(is_ivf=False)

        # otherwise, use IVF-FLAT
        cluster_id_sorted = inner_product(
            vector, self.matrix_serializer.indexer.ivf.ann_model.cluster_centers_,
            n=self.matrix_serializer.indexer.ivf.ann_model.cluster_centers_.shape[0],
            use_simd=False
        )[0].tolist()

        for cluster_id in cluster_id_sorted:
            batch_search(cid=cluster_id, directly_return=False, is_ivf=True)

            if len(limited_sorted) >= k:
                break

        res_ids, res_scores = limited_sorted.get_top_n()
        return (
            res_ids, res_scores if self.matrix_serializer.indexer.index_mode != 'IVF-IP' else -1 * res_scores
        )

    @SearchResultsCache(config.LYNSE_SEARCH_CACHE_SIZE, config.LYNSE_SEARCH_CACHE_EXPIRE_SECONDS)
    def single_search(self, vector, k=12, search_filter=None, return_fields=False, **kwargs):
        """
        Search the database for the vectors most similar to the given vector in batches.
        """
        if self.matrix_serializer.indexer.index_mode.split('-')[-1] not in ['SQ8', 'Binary']:
            res_ids, res_scores = self._flat_search(vector, k, search_filter)
        else:
            params = {
                'original_vec': vector,
                'top_k': k,
                'subset_indices': self._narrow_down_ids(search_filter)
            }

            if check_has_param(self.matrix_serializer.indexer.index.search, 'rescore_multiplier'):
                if 'Binary' in self.matrix_serializer.indexer.index_mode:
                    params['rescore_multiplier'] = kwargs.get('rescore_multiplier', 10)
                else:
                    params['rescore_multiplier'] = kwargs.get('rescore_multiplier', 2)

            if check_has_param(self.matrix_serializer.indexer.index.search, 'rescore'):
                params['rescore'] = kwargs.get('rescore', False)

            res_ids, res_scores = self.matrix_serializer.indexer.index.search(**params)

        if return_fields:
            res_fields = self.fields_index.retrieve_ids(res_ids.tolist(), include_external_id=True)
        else:
            res_fields = None

        return res_ids, res_scores, res_fields

    def multi_search(self, vectors, k=12, search_filter=None, return_fields=False, **kwargs):
        """
        Search the database for the vectors most similar to the given vectors in batches.
        """
        ids = []
        scores = []
        fields = []

        for i, vector in enumerate(vectors):
            res_ids, res_scores, res_fields = self.single_search(vector, k, search_filter, return_fields, **kwargs)
            ids.append(res_ids)
            scores.append(res_scores)
            fields.append(res_fields)

        return np.vstack(ids), np.vstack(scores), fields

    def search(self, vector, k=12, search_filter=None, return_fields=False, **kwargs):
        """
        Search the database for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray or list): The search vectors, it can be a single vector or a list of vectors.
                The vectors must have the same dimension as the vectors in the database,
                and the type of vector can be a list or a numpy array.
            k (int): The number of nearest vectors to return.
            search_filter (Filter or FilterExpression string, optional): The filter to apply to the search.
            return_fields (bool): Whether to return the fields of the search results.
            kwargs: Additional keyword arguments. The following are valid:
                rescore (bool): Whether to rescore the results of binary or scaler quantization searches.
                    Default is False. It is recommended to set it to True when the index mode is 'Binary'.
                rescore_multiplier (int): The multiplier for the rescore operation.
                    It is only available when rescore is True.
                    If 'Binary' is in the index mode, the default is 10. Otherwise, the default is 2.

        Returns:
            Tuple: If return_fields is True, the indices, similarity scores,
                    and fields of the nearest vectors in the database.
                Otherwise, the indices and similarity scores of the nearest vectors in the database.

        Raises:
            ValueError: If the database is empty.
        """
        if not hasattr(self.matrix_serializer, "indexer"):
            raise ValueError('The database has not been indexed.')

        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector).squeeze()

        if isinstance(search_filter, str):
            search_filter = ExpressionParser(search_filter).to_filter()

        if vector.shape[0] > 1:
            return self.multi_search(vector, k, search_filter, return_fields, **kwargs)
        else:
            return self.single_search(vector, k, search_filter, return_fields, **kwargs)

    def __del__(self):
        if self.executors is not None:
            self.executors.shutdown(wait=True)
            self.logger.debug('Search executor shutdown.')

    def delete(self):
        self.__del__()
