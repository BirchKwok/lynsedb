import threading


class ThreadLock:
    def __init__(self):
        self._thread_lock = threading.RLock()

    def acquire(self):
        """Acquire the lock based on the running environment."""
        self._thread_lock.acquire()

    def release(self):
        """Release the lock based on the running environment."""
        self._thread_lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
