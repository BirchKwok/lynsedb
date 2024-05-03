from collections import OrderedDict
import threading


class LimitedDict:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def __setitem__(self, key, value):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def __getitem__(self, key):
        with self.lock:
            try:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            except KeyError:
                raise KeyError('Key not found')

    @property
    def is_reached_max_size(self):
        return len(self.cache) == self.max_size

    def get(self, key, default=None):
        with self.lock:
            if key not in self.cache:
                return default
            return self.__getitem__(key)

    def clear(self):
        with self.lock:
            self.cache.clear()

    def keys(self):
        with self.lock:
            return self.cache.keys()

    def pop(self, key, default=None):
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
