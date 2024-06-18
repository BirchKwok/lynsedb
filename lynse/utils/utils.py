import time
from functools import wraps
from pathlib import Path

import numpy as np


class SearchResultsCache:
    """A decorator that caches the results of a function call with the same arguments."""

    def __init__(self, max_size=1000, expire_seconds=3600):
        from collections import OrderedDict

        self.cache = OrderedDict()
        self.max_size = max_size
        self.expire_seconds = expire_seconds

    def clear_cache(self):
        self.cache.clear()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key_parts = []
            for arg in args[1:]:  # ignore the self parameter
                if hasattr(arg, 'tobytes'):
                    key_parts.append(('vector', arg.tobytes()))
                elif hasattr(arg, '__dict__'):
                    key_parts.append(arg.__dict__)
                elif isinstance(arg, list):
                    key_parts.append(tuple(arg))
                else:
                    key_parts.append(arg)

            for k, v in kwargs.items():
                if hasattr(v, 'tobytes'):
                    key_parts.append((k, v.tobytes()))
                elif hasattr(v, '__dict__'):
                    key_parts.append(v.__dict__)
                elif isinstance(v, list):
                    key_parts.append(tuple(v))
                else:
                    key_parts.append((k, v))

            key = tuple(key_parts)

            current_time = time.mktime(time.gmtime())
            if key in self.cache:
                result, timestamp = self.cache[key]
                if current_time - timestamp < self.expire_seconds:
                    return result

            result = func(*args, **kwargs)
            self.cache[key] = (result, current_time)

            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            return result

        wrapper.clear_cache = self.clear_cache
        return wrapper


def unavailable_if_deleted(func):
    """A decorator that detects if the function is called after the object is deleted."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(args[0], '_initialize_as_collection'):
            # self is the first parameter
            if args[0]._initialize_as_collection:
                unit_name = 'collection'
            else:
                unit_name = 'database'

            db_name = Path(args[0]._database_path).name

            if args[0]._matrix_serializer.IS_DELETED:
                raise ValueError(f"The {unit_name} `{db_name}` has been deleted, and the `{func.__name__}` function "
                                 f"is unavailable.")
        else:
            db_name = Path(args[0].root_path).name

            if args[0].STATUS == 'DELETED':
                raise ValueError(f"The `{db_name}` has been deleted, and the `{func.__name__}` function "
                                 f"is unavailable.")

        return func(*args, **kwargs)

    return wrapper


def load_chunk_file(filename):
    np_array = np.load(filename, mmap_mode='r')
    return np_array
