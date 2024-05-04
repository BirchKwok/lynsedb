import numpy as np

from min_vec.computational_layer.engines import cosine_distance, argsort_topk, euclidean_distance
from min_vec.core_components.cross_lock import ThreadLock


class LimitedSorted:
    def __init__(self, dim, dtype, scaler, chunk_size):
        """A class to store the top n most similar vectors to a given vector.

        .. versionadded:: 0.2.5

        Parameters:
            dim (int): The dimension of the vectors.
            dtype (type): The data type of the vectors.
            scaler (Scaler, optional): The scaler to decode the vectors.
            chunk_size (int): The maximum number of vectors to store.
                .. versionadded:: 0.2.7
        """
        self.lock = ThreadLock()
        self.dim = dim
        self.n = None
        self.scaler = scaler
        self.max_length = chunk_size
        self.current_length = 0
        self.similarities = np.empty(self.max_length, dtype=dtype)
        self.indices = np.empty(self.max_length, dtype=int)
        self.matrix_subset = np.empty((self.max_length, dim), dtype=dtype)

    def set_n(self, n):
        self.n = n

    def add(self, sim: np.ndarray, indices: np.ndarray, matrix: np.ndarray):
        num_new_items = len(sim)
        with self.lock:
            end_pos = self.current_length + num_new_items
            if end_pos > self.max_length:
                self.max_length = max(self.max_length * 2, end_pos)
                self.similarities = np.resize(self.similarities, self.max_length)
                self.indices = np.resize(self.indices, self.max_length)
                self.matrix_subset = np.resize(self.matrix_subset, (self.max_length, self.dim))

            self.similarities[self.current_length:end_pos] = sim
            self.indices[self.current_length:end_pos] = indices
            self.matrix_subset[self.current_length:end_pos] = matrix

            self.current_length = end_pos

            idx = argsort_topk(-self.similarities[:self.current_length], self.n)

            idx_len = len(idx)
            self.similarities[:idx_len] = self.similarities[idx]
            self.indices[:idx_len] = self.indices[idx]
            self.matrix_subset[:idx_len] = self.matrix_subset[idx]
            self.current_length = idx_len

    def get_top_n(self, vector: np.ndarray, distance='cosine'):
        with self.lock:
            if distance == 'cosine':
                distance_func = cosine_distance
            else:
                distance_func = euclidean_distance

            if self.scaler is None:
                sim = distance_func(vector, self.matrix_subset[:self.current_length])
            else:
                decoded_vectors = self.scaler.decode(self.matrix_subset[:self.current_length])
                sim = distance_func(vector, decoded_vectors)

            sorted_idx = np.argsort(-sim) if distance == 'cosine' else np.argsort(sim)

            return self.indices[sorted_idx], sim[sorted_idx]

    def clear(self):
        with self.lock:
            self.current_length = 0
            self.similarities = np.empty(self.max_length, dtype=self.similarities.dtype)
            self.indices = np.empty(self.max_length, dtype=self.indices.dtype)
            self.matrix_subset = np.empty((self.max_length, self.dim), dtype=self.matrix_subset.dtype)
