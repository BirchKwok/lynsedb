from min_vec.core_components.cross_lock import ThreadLock


class SafeList(list):
    def __init__(self, *args, **kwargs):
        super(SafeList, self).__init__(*args, **kwargs)
        self.lock = ThreadLock()

    def append(self, item):
        with self.lock:
            super(SafeList, self).append(item)

    def pop(self, index=-1):
        with self.lock:
            return super(SafeList, self).pop(index)

    def __getitem__(self, index):
        with self.lock:
            return super(SafeList, self).__getitem__(index)

    def __setitem__(self, index, value):
        with self.lock:
            return super(SafeList, self).__setitem__(index, value)

    def __delitem__(self, index):
        with self.lock:
            return super(SafeList, self).__delitem__(index)

    def extend(self, iterable):
        with self.lock:
            return super(SafeList, self).extend(iterable)

    def remove(self, item):
        with self.lock:
            return super(SafeList, self).remove(item)

    def __len__(self):
        with self.lock:
            return super(SafeList, self).__len__()

    def __iter__(self):
        with self.lock:
            return iter(self.copy())

    def __str__(self):
        with self.lock:
            return super(SafeList, self).__str__()

    def clear(self):
        with self.lock:
            return super(SafeList, self).clear()
