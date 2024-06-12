import threading
from functools import partial

import numpy as np

from lynse.computational_layer.engines import inner_product_distance, euclidean_distance, cosine_distance


class LimitedSorted:
    def __init__(self, scaler=None, n=10):
        """A class to store the top n most similar vectors to a given vector.

        .. versionadded:: 0.2.5

        Parameters:
            scaler (Scaler, optional): The scaler to decode the vectors.
            n (int): The number of vectors to store.
        """
        self.scaler = scaler
        self.n = n
        self.similarities = []
        self.indices = []
        self.matrix_subset = []
        self.lock = threading.RLock()

    def add(self, sim: np.ndarray, indices: np.ndarray, matrix: np.ndarray):
        if sim.shape[0] < 100:
            n_idx = np.argsort(-sim)
        else:
            n_idx = np.argpartition(-sim, self.n)[:self.n]

        sim = sim[n_idx]
        indices = indices[n_idx]
        matrix = matrix[n_idx]

        with self.lock:
            self.similarities.append(sim)
            self.indices.append(indices)
            self.matrix_subset.append(matrix)

    def get_top_n(self, vector: np.ndarray, distance='IP', use='np'):
        if distance == 'IP':
            distance_func = partial(inner_product_distance, use=use)
        elif distance == 'cosine':
            distance_func = partial(cosine_distance, use=use)
        else:
            distance_func = partial(euclidean_distance, use=use)

        matrix_subset = np.vstack(self.matrix_subset)
        indices = np.concatenate(self.indices)

        if self.scaler is None:
            sim = distance_func(vector, matrix_subset)
        else:
            decoded_vectors = self.scaler.decode(matrix_subset)
            sim = distance_func(vector, decoded_vectors)

        sorted_idx = np.argsort(-sim) if distance in ['cosine', 'IP'] else np.argsort(sim)
        sorted_idx = sorted_idx[:self.n]

        return indices[sorted_idx], sim[sorted_idx]

    def __len__(self):
        return self.n * len(self.similarities)
