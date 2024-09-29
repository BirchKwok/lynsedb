import threading


class ThreadLock:
    def __init__(self):
        """Initialize a reentrant lock."""
        self._thread_lock = threading.RLock()

    def acquire(self, blocking=True, timeout=-1):
        """
        Acquire the lock, with optional blocking and timeout.

        Parameters:
            blocking (bool): Whether to block if the lock cannot be acquired.
            timeout (float): The maximum time to wait for the lock.
                             If -1, wait indefinitely.

        Returns:
            bool: True if the lock was acquired, False otherwise.
        """
        return self._thread_lock.acquire(blocking, timeout)

    def release(self):
        """Release the lock."""
        self._thread_lock.release()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        self.release()
