from collections import OrderedDict


class LimitedDict:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = OrderedDict()

    def __setitem__(self, key, value):
        # 先检查键是否已存在，如果是，则先删除旧键
        if key in self.cache:
            del self.cache[key]
        self.cache[key] = value
        # 如果缓存超出最大大小，移除最老的项
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def __getitem__(self, key):
        try:
            # 移动到末尾，表示最近使用
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            raise KeyError('Key not found')

    def get(self, key, default=None):
        if key not in self.cache:
            return default
        return self.__getitem__(key)

    def clear(self):
        self.cache.clear()

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
