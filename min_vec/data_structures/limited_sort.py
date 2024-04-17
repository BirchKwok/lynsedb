import threading
import numpy as np

from min_vec.computational_layer.engines import cosine_distance, argsort_topk, euclidean_distance


class LimitedSorted:
    def __init__(self, vector: np.ndarray, n: int, scaler, chunk_size, distance='cosine'):
        """A class to store the top n most similar vectors to a given vector.

        .. versionadded:: 0.2.5

        Parameters:
            vector (np.ndarray): The vector to compare against.
            n (int): The number of most similar vectors to store.
            scaler (Scaler, optional): The scaler to decode the vectors.
            chunk_size (int): The maximum number of vectors to store.
                .. versionadded:: 0.2.7
            distance (str): The distance metric to use for the query.
                .. versionadded:: 0.2.7
        """
        self.lock = threading.RLock()
        self.vector = vector
        self.n = n
        self.distance = distance
        self.distance_func = cosine_distance if distance == 'cosine' else euclidean_distance
        self.scaler = scaler
        self.max_length = chunk_size
        self.current_length = 0
        self.similarities = np.empty(self.max_length, dtype=vector.dtype)
        self.indices = np.empty(self.max_length, dtype=int)
        self.matrix_subset = np.empty((self.max_length, vector.size), dtype=vector.dtype)

    def add(self, sim: np.ndarray, indices: np.ndarray, matrix: np.ndarray):
        num_new_items = len(sim)
        with self.lock:
            end_pos = self.current_length + num_new_items
            if end_pos > self.max_length:
                self.max_length = max(self.max_length * 2, end_pos)
                self.similarities = np.resize(self.similarities, self.max_length)
                self.indices = np.resize(self.indices, self.max_length)
                self.matrix_subset = np.resize(self.matrix_subset, (self.max_length, self.vector.size))

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

    def get_top_n(self):
        if self.scaler is None:
            sim = self.distance_func(self.vector, self.matrix_subset[:self.current_length])
        else:
            decoded_vectors = self.scaler.decode(self.matrix_subset[:self.current_length])
            sim = self.distance_func(self.vector, decoded_vectors)

        sorted_idx = np.argsort(-sim) if self.distance_func == cosine_distance else np.argsort(sim)

        return self.indices[sorted_idx], sim[sorted_idx]
