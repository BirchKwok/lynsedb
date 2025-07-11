from queue import Queue
import numpy as np


class LimitedSorted:
    def __init__(self, n=10, bigger_is_better=True):
        """A class to store the top n most similar vectors to a given vector.

        Parameters:
            n (int): The number of vectors to store.
            bigger_is_better (bool): If True, larger values are considered better and will be ranked higher.
                                   If False, smaller values are considered better.
        """
        self.n = n
        self.bigger_is_better = bigger_is_better
        self.similarities = Queue()
        self.indices = Queue()
        self.is_2d = None

    def add(self, sim: np.ndarray, indices: np.ndarray):
        """Add similarities and their corresponding indices.

        Parameters:
            sim (np.ndarray): 1D or 2D array of similarities.
            indices (np.ndarray): 1D or 2D array of corresponding indices.
        """
        sim = np.asarray(sim)
        indices = np.asarray(indices)

        if sim.ndim == 0:
            sim = sim.reshape(1)
        if indices.ndim == 0:
            indices = indices.reshape(1)

        if sim.ndim > 2 or indices.ndim > 2:
            raise ValueError("Only 1D or 2D arrays are supported.")

        if self.is_2d is None:
            self.is_2d = sim.ndim == 2
        elif (sim.ndim == 2) != self.is_2d:
            if not self.is_2d:
                sim = sim.reshape(-1, 1) if sim.ndim == 1 else sim
                indices = indices.reshape(-1, 1) if indices.ndim == 1 else indices
            else:
                sim = sim.ravel()
                indices = indices.ravel()

        if sim.shape != indices.shape:
            indices = np.broadcast_to(indices, sim.shape)

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
            indices_list = [np.asarray(idx) for idx in self.indices.queue]
            sim_list = [np.asarray(sim) for sim in self.similarities.queue]

            target_shape = sim_list[0].shape
            indices = np.vstack([np.broadcast_to(idx, target_shape) for idx in indices_list])
            sim = np.vstack(sim_list)

            sorted_indices, topk_similarities = self._sort_n_idx_2d(sim, indices)
            return sorted_indices, topk_similarities
        else:
            indices = np.concatenate(self.indices.queue)
            sim = np.concatenate(self.similarities.queue)
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
            # 根据bigger_is_better决定排序方向
            return np.argsort(sim)[::(-1 if self.bigger_is_better else 1)][:self.n]

        # 对于大数组使用partition
        if self.bigger_is_better:
            top_n_idx = np.argpartition(sim, -self.n)[-self.n:]
            sorted_idx = np.argsort(sim[top_n_idx])[::-1]
        else:
            top_n_idx = np.argpartition(sim, self.n-1)[:self.n]
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
        if self.bigger_is_better:
            partitioned_indices = np.argpartition(sim, -self.n, axis=1)[:, -self.n:]
        else:
            partitioned_indices = np.argpartition(sim, self.n-1, axis=1)[:, :self.n]

        top_n_similarities = np.take_along_axis(sim, partitioned_indices, axis=1)
        top_n_indices = np.take_along_axis(indices, partitioned_indices, axis=1)

        # 根据bigger_is_better决定排序方向
        sorted_order = np.argsort(top_n_similarities, axis=1)[:, ::-1 if self.bigger_is_better else 1]

        top_n_similarities_sorted = np.take_along_axis(top_n_similarities, sorted_order, axis=1)
        top_n_indices_sorted = np.take_along_axis(top_n_indices, sorted_order, axis=1)

        return top_n_indices_sorted, top_n_similarities_sorted

    def __len__(self):
        return len([i.shape[0] for i in self.similarities.queue])
