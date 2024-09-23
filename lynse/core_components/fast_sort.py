import numpy as np


class FastSort:
    def __init__(self, data):
        self.data = np.asarray(data) if not isinstance(data, np.ndarray) else data
        self.data = self.data.squeeze()

    def topk(self, k, ascending=True):
        axis = 0
        k = min(k, self.data.shape[axis])

        if ascending:
            partitioned_indices = np.argpartition(self.data, k-1, axis=axis)
            topk_indices = partitioned_indices[:k]
        else:
            partitioned_indices = np.argpartition(self.data, self.data.shape[1] - k, axis=axis)
            topk_indices = partitioned_indices[-k:]

        topk_values = np.take_along_axis(self.data, topk_indices, axis=axis)
        sorted_indices = np.argsort(topk_values, axis=axis)
        if not ascending:
            sorted_indices = sorted_indices[::-1]

        topk_indices = np.take_along_axis(topk_indices, sorted_indices, axis=axis)

        topk_values = np.take_along_axis(self.data, topk_indices, axis=axis)
        return topk_indices, topk_values
