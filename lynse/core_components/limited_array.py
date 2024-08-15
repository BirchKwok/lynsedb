import numpy as np

from ..core_components.locks import ThreadLock


class LimitedArray:
    def __init__(self, dim, n=10):
        """This class is used to store a limited number of arrays.

        Parameters:
            dim (int): The dimension of the arrays.
            n (int): The maximum number of arrays to store. If n is -1, there is no limit.
        """
        self.dim = dim
        self.n = n if n != -1 else float('inf')
        self.counter = 0
        self.arrays = None
        self.ids = None
        self.lock = ThreadLock()
        self.filename = {}

    def add(self, filename, array, ids):
        """
        Add an array to the storage.

        Parameters:
            filename (str): The filename of the array.
            array (np.ndarray): The array to add.
            ids (np.ndarray): The IDs of the array.

        Returns:
            None
        """
        if self.n == 0:
            return

        existed_filename = sorted(list(self.filename.keys()), key=lambda x: int(x.split('_')[-1]))

        with self.lock:
            if filename in self.filename:
                if filename != existed_filename[-1]:
                    last_row_idx, end_idx = self.filename[filename]
                    self.arrays[last_row_idx:end_idx] = array
                    self.ids[last_row_idx:end_idx] = ids
                else:
                    last_row_idx, end_idx = self.filename[filename]
                    self.arrays[last_row_idx:end_idx] = array[:end_idx - last_row_idx]
                    self.ids[last_row_idx:end_idx] = ids[:end_idx - last_row_idx]
                    self.arrays = np.vstack((self.arrays, array[end_idx - last_row_idx:]))
                    self.ids = np.hstack((self.ids, ids[end_idx - last_row_idx:]))
                    end_idx = end_idx + array[end_idx - last_row_idx:].shape[0]
            else:
                if self.arrays is None:
                    last_row_idx = 0
                    self.arrays = array.copy()
                    self.ids = ids.copy()
                    end_idx = self.arrays.shape[0]
                else:
                    last_row_idx = self.arrays.shape[0] - 1 if self.arrays.shape[0] > 0 else 0
                    if self.counter < self.n:
                        self.arrays = np.vstack((self.arrays, array))
                        self.ids = np.hstack((self.ids, ids))
                        self.counter += 1
                    else:
                        self.arrays = np.vstack((self.arrays[1:], array))
                        self.ids = np.hstack((self.ids[1:], ids))

                    end_idx = last_row_idx + array.shape[0]

            self.filename[filename] = (last_row_idx, end_idx)

    def get(self, filename):
        """
        Get the arrays by filename.

        Parameters:
            filename (str): The filename of the arrays.

        Returns:
            np.ndarray, np.ndarray: The arrays and the IDs.
        """
        if filename not in self.filename:
            return None, None
        return self.arrays, self.ids

    def clear(self):
        with self.lock:
            self.arrays = np.empty((0, self.dim), dtype=np.float32)
            self.ids = np.empty((0,), dtype=np.int32)
            self.counter = 0
            self.filename = {}

    @property
    def is_reached_max_size(self):
        return self.counter == self.n

    def __len__(self):
        return self.counter

    def __contains__(self, item):
        return item in self.filename
