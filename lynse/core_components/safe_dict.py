from ..core_components.locks import ThreadLock


class SafeDict(dict):
    def __init__(self):
        super().__init__()
        self.lock = ThreadLock()

    def __setitem__(self, key, value):
        with self.lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with self.lock:
            super().__delitem__(key)

    def pop(self, key, default=None):
        with self.lock:
            if key not in self:
                return default
            return super().pop(key)

    def clear(self):
        with self.lock:
            super().clear()

    def update(self, other):
        with self.lock:
            super().update(other)
