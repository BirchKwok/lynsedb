import numpy as np


class FastSort:
    def __init__(self, data, backend='numpy'):
        self.data = np.asarray(data) if not isinstance(data, np.ndarray) else data
        self.backend = backend
        if backend == 'jax':
            import jax.numpy as jnp
            self.np = jnp
        else:
            self.np = np

    def topk(self, k, ascending=True):
        if len(self.data) < 10000:
            sorted_indices = self.np.argsort(self.data)
            if ascending:
                topk_indices = sorted_indices[:k]
            else:
                topk_indices = sorted_indices[-k:][::-1]
        else:
            if ascending:
                partitioned_indices = self.np.argpartition(self.data, k-1)
                topk_indices = partitioned_indices[:k]
                topk_indices = topk_indices[self.np.argsort(self.data[topk_indices])]
            else:
                partitioned_indices = self.np.argpartition(self.data, len(self.data) - k)
                topk_indices = partitioned_indices[-k:]
                topk_indices = topk_indices[self.np.argsort(self.data[topk_indices])[::-1]]

        topk_values = self.data[topk_indices]
        return np.array(topk_indices), np.array(topk_values)
