from queue import Queue
import numpy as np


class LimitedSorted:
    def __init__(self, n=10):
        """A class to store the top n most similar vectors to a given vector.

        Parameters:
            n (int): The number of vectors to store.
        """
        self.n = n
        self.similarities = Queue()
        self.indices = Queue()
        self.is_2d = None

    def add(self, sim: np.ndarray, indices: np.ndarray):
        """Add similarities and their corresponding indices.

        Parameters:
            sim (np.ndarray): 1D or 2D array of similarities.
            indices (np.ndarray): 1D or 2D array of corresponding indices.
        """
        if sim.ndim != indices.ndim:
            raise ValueError("sim and indices must have the same number of dimensions.")

        if self.is_2d is None:
            self.is_2d = sim.ndim == 2

        if (sim.ndim == 2) != self.is_2d:
            raise ValueError("All added data must have the same number of dimensions.")

        self.similarities.put(sim)
        self.indices.put(indices)

    def get_top_n(self):
        """Get the top n most similar vectors.

        Returns:
            np.ndarray: The indices of the top n most similar vectors.
            np.ndarray: The similarities of the top n most similar vectors.
        """
        if self.similarities.empty():
            return np.array([]), np.array([])

        if self.is_2d:
            indices = np.vstack(np.atleast_2d(self.indices.queue))
            sim = np.vstack(np.atleast_2d(self.similarities.queue))

            sorted_indices, topk_similarities = self._sort_n_idx_2d(sim, indices)
            return sorted_indices, topk_similarities
        else:
            indices = np.concatenate(np.atleast_2d(self.indices.queue))
            sim = np.concatenate(np.atleast_2d(self.similarities.queue))
            sorted_idx = self._sort_n_idx(sim)
            return indices[sorted_idx], sim[sorted_idx]

    def _sort_n_idx(self, sim):
        """Sort and retrieve indices of the top n similarities for 1D case.

        Parameters:
            sim (np.ndarray): 1D array of similarities.

        Returns:
            np.ndarray: Indices of the top n similarities.
        """
        if sim.shape[0] <= 1000:
            return np.argsort(sim)[:self.n]

        top_n_idx = np.argpartition(sim, self.n - 1)[:self.n]
        sorted_idx = np.argsort(sim[top_n_idx])
        return top_n_idx[sorted_idx]

    def _sort_n_idx_2d(self, sim, indices):
        """Sort and retrieve top n similarities for each row in 2D case using vectorized operations.

        Parameters:
            sim (np.ndarray): 2D array of similarities.
            indices (np.ndarray): 2D array of corresponding indices.

        Returns:
            np.ndarray: Sorted indices of the top n similarities for each row.
            np.ndarray: The top n similarities for each row.
        """
        partitioned_indices = np.argpartition(sim, self.n - 1, axis=1)[:, :self.n]
        top_n_similarities = np.take_along_axis(sim, partitioned_indices, axis=1)

        top_n_indices = np.take_along_axis(indices, partitioned_indices, axis=1)

        sorted_order = np.argsort(top_n_similarities, axis=1)

        top_n_similarities_sorted = np.take_along_axis(top_n_similarities, sorted_order, axis=1)
        top_n_indices_sorted = np.take_along_axis(top_n_indices, sorted_order, axis=1)

        return top_n_indices_sorted, top_n_similarities_sorted

    def __len__(self):
        return len(self.similarities.queue)
