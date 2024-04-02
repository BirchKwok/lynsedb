

class LimitedDict(dict):
    def __init__(self, max_size):
        self.max_size = max_size
        self.keys = []
        super().__init__()

    def __setitem__(self, key, value):
        if key not in self.keys:
            if len(self.keys) >= self.max_size:
                del self[self.keys[0]]
                self.keys.pop(0)
            self.keys.append(key)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        return super().get(key, None)

    def __delitem__(self, key):
        if key in self.keys:
            self.keys.remove(key)
        super().__delitem__(key)

    def __repr__(self):
        return str({k: self[k] for k in self.keys})