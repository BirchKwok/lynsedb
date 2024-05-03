import threading


class ThreadSafeCounter:
    def __init__(self):
        """
        Initialize the thread-safe counter.
        """
        self.value = 0
        self.lock = threading.RLock()

    def increment(self):
        with self.lock:
            self.value += 1

    def decrement(self):
        with self.lock:
            self.value -= 1

    def get_value(self):
        with self.lock:
            return self.value

    def reset(self):
        with self.lock:
            self.value = 0

    def add(self, value):
        with self.lock:
            self.value += value

    def subtract(self, value):
        with self.lock:
            self.value -= value
