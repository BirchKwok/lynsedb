from collections import OrderedDict

import numpy as np

from ..core_components.locks import ThreadLock


class LimitedDict:
    def __init__(self, max_size):
        if not isinstance(max_size, int):
            raise ValueError('max_size must be an integer')

        if max_size == -1:
            self.max_size = np.inf
        elif max_size < 0:
            raise ValueError('max_size must be a positive integer or -1')
        else:
            self.max_size = max_size

        self.cache = OrderedDict()
        self.lock = ThreadLock()

    def __setitem__(self, key, value):
        if self.max_size == 0:
            return

        with self.lock:
            if key in self.cache:
                del self.cache[key]
            self.cache[key] = value

            if self.cache is not None and len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def __getitem__(self, key):
        if self.max_size == 0:
            return

        if key in self.cache:
            value = self.cache[key]
        else:
            raise KeyError('Key not found')

        return value

    @property
    def is_reached_max_size(self):
        if self.max_size == 0:
            return True

        with self.lock:
            return len(self.cache) == self.max_size

    def get(self, key, default=None):
        if self.max_size == 0:
            return default

        if key not in self.cache:
            return default
        return self.__getitem__(key)

    def clear(self):
        with self.lock:
            self.cache.clear()

    def keys(self):
        if self.max_size == 0:
            return []

        with self.lock:
            return self.cache.keys()

    def pop(self, key, default=None):
        if self.max_size == 0:
            return default

        with self.lock:
            return self.cache.pop(key, default)

    def __contains__(self, key):
        with self.lock:
            return key in self.cache

    def __len__(self):
        with self.lock:
            return len(self.cache)

    def __repr__(self):
        with self.lock:
            return repr(self.cache)
