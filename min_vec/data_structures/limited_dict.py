from collections import OrderedDict


class LimitedDict:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = OrderedDict()

    def __setitem__(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def __getitem__(self, key):
        return self.cache[key]

    def clear(self):
        self.cache.clear()

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def keys(self):
        return self.cache.keys()

    def pop(self, key, default=None):
        return self.cache.pop(key, default)

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        return repr(self.cache)

