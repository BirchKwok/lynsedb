import numpy as np

from cvg.core_components.locks import ThreadLock


class LimitedArray:
    def __init__(self, dim, n=10):
        """存储缓存数组的类。

        """
        self.dim = dim
        self.n = n if n != -1 else float('inf')
        self.counter = 0
        self.arrays = np.empty((0, dim), dtype=np.float32)
        self.ids = np.empty((0,), dtype=np.int32)
        self.lock = ThreadLock()
        self.filename = {}

    def add(self, filename, array, ids):
        if self.n == 0:
            return

        with self.lock:
            if filename in self.filename:
                start_idx, end_idx = self.filename[filename]
                self.arrays[start_idx:end_idx] = array
                self.ids[start_idx:end_idx] = ids
            else:
                last_row_idx = self.arrays.shape[0] - 1
                if self.counter < self.n:
                    self.arrays = np.vstack((self.arrays, array))
                    self.ids = np.hstack((self.ids, ids))
                    self.counter += 1
                else:
                    self.arrays = np.vstack((self.arrays[1:], array))
                    self.ids = np.hstack((self.ids[1:], ids))

                self.filename[filename] = (last_row_idx, self.arrays.shape[0])

    def get(self, filename):
        if filename not in self.filename:
            return None, None
        return self.arrays, self.ids

    def clear(self):
        with self.lock:
            self.arrays = np.empty((0, self.dim), dtype=np.float32)
            self.ids = np.empty((0,), dtype=np.int32)
            self.counter = 0

    def is_reached_max_size(self):
        return self.counter == self.n

    def __len__(self):
        return self.counter

    def __contains__(self, item):
        return item in self.filename
