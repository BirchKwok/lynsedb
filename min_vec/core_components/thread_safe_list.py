import threading


class ThreadSafeList(list):
    def __init__(self, *args, **kwargs):
        super(ThreadSafeList, self).__init__(*args, **kwargs)
        self.lock = threading.RLock()

    def append(self, item):
        with self.lock:
            super(ThreadSafeList, self).append(item)

    def pop(self, index=-1):
        with self.lock:
            return super(ThreadSafeList, self).pop(index)

    def __getitem__(self, index):
        with self.lock:
            return super(ThreadSafeList, self).__getitem__(index)

    def __setitem__(self, index, value):
        with self.lock:
            return super(ThreadSafeList, self).__setitem__(index, value)

    def __delitem__(self, index):
        with self.lock:
            return super(ThreadSafeList, self).__delitem__(index)

    def extend(self, iterable):
        with self.lock:
            return super(ThreadSafeList, self).extend(iterable)

    def remove(self, item):
        with self.lock:
            return super(ThreadSafeList, self).remove(item)

    def __len__(self):
        with self.lock:
            return super(ThreadSafeList, self).__len__()

    def __iter__(self):
        with self.lock:
            return iter(self.copy())

    def __str__(self):
        with self.lock:
            return super(ThreadSafeList, self).__str__()

    def clear(self):
        with self.lock:
            return super(ThreadSafeList, self).clear()
