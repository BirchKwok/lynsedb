import threading

import numpy as np


class LimitedSorted:
    def __init__(self, n=10):
        """A class to store the top n most similar vectors to a given vector.


        Parameters:
            n (int): The number of vectors to store.
        """
        self.n = n
        self.similarities = []
        self.indices = []
        self.lock = threading.RLock()

    def add(self, sim: np.ndarray, indices: np.ndarray):
        with self.lock:
            self.similarities.append(sim)
            self.indices.append(indices)

    def get_top_n(self):
        """Get the top n most similar vectors.

        Returns:
            np.ndarray: The indices of the top n most similar vectors.
            np.ndarray: The similarities of the top n most similar vectors.
        """
        if len(self.similarities) == 0:
            return np.array([]), np.array([])

        indices = np.concatenate(self.indices)
        sim = np.concatenate(self.similarities)

        sorted_idx = self._sort_n_idx(sim)
        return indices[sorted_idx], sim[sorted_idx]

    def _sort_n_idx(self, sim):
        if sim.shape[0] <= 1000:
            return np.argsort(sim)[:self.n]

        top_n_idx = np.argpartition(sim, self.n - 1)[:self.n]
        sorted_idx = np.argsort(sim[top_n_idx])
        return top_n_idx[sorted_idx]

    def __len__(self):
        return len(self.similarities)
