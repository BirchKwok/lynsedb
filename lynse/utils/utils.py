import os
import re
import time
from functools import wraps
from pathlib import Path

import numpy as np


class OpsError(Exception):
    """An exception that is raised when an error occurs during a database operation."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


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
        if hasattr(args[0], '_search'):
            # self is the first parameter
            db_name = Path(args[0]._database_path).name

            if args[0]._matrix_serializer.IS_DELETED:
                raise OpsError(f"The collection `{db_name}` has been deleted, and the `{func.__name__}` function "
                               f"is unavailable.")
        else:
            db_name = Path(args[0].root_path).name

            if args[0].STATUS == 'DELETED':
                raise OpsError(f"The `{db_name}` has been deleted, and the `{func.__name__}` function "
                               f"is unavailable.")

        return func(*args, **kwargs)

    return wrapper


def unavailable_if_empty(func):
    """A decorator that detects if the database is empty."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args[0].shape[0] == 0:
            raise OpsError(f"The database is empty, and the `{func.__name__}` function is unavailable.")

        return func(*args, **kwargs)

    return wrapper


def drop_duplicated_substr(source, target) -> str:
    """
    Remove all occurrences of the target string from the source string.

    Parameters:
        source (str): The source string.
        target (str): The target string.

    Returns:
        str: The source string with all occurrences of the target string removed
    """
    t_len = len(target)
    s_len = len(source)

    indices_to_remove = []
    i = 0

    while i <= s_len - t_len:
        if source[i:i + t_len] == target:
            indices_to_remove.append(i)
            i += t_len
        else:
            i += 1

    result = []
    last_index = 0

    for start_index in indices_to_remove:
        result.append(source[last_index:start_index])
        last_index = start_index + t_len

    result.append(source[last_index:])

    return ''.join(result)


def find_first_file_with_substr(directory, substr):
    """
    Find the first file with the specified substring or wildcard in the directory.

    Parameters:
        directory (str or Pathlike): The directory to search.
        substr (str): The substring or wildcard pattern of the file to search for.

    Returns:
        path: The path to the first file with the specified substring or wildcard pattern in the directory.
    """
    # Convert wildcard pattern to regular expression
    regex_pattern = re.compile(re.escape(substr).replace(r'\*', '.*'))

    for file in Path(directory).iterdir():
        if regex_pattern.search(file.name):
            return file.absolute()

    return None


def collection_repr(collection):
    """
    Get the string representation of a collection.

    Parameters:
        collection (Collection): The collection to represent.

    Returns:
        str: The string representation of the collection.
    """
    return (f'{collection.name}CollectionInstance(\n'
            f'    database="{collection._database_name}", \n'
            f'    collection="{collection._collection_name}", \n'
            f'    shape={collection.shape}'
            f'\n)')


def sort_and_get_top_k(arr, k):
    """
    Sort each row of a 1D or 2D array and return the top-k indices and values for each row.

    Parameters:
        arr (np.ndarray): 1D or 2D numpy array
        k (int): Number of top elements to return

    Returns:
        tuple: (top_k_indices, top_k_values)
            top_k_indices: Indices of top-k elements for each row
            top_k_values: Values of top-k elements for each row
    """
    arr = np.asarray(arr)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n_rows, n_cols = arr.shape

    k = min(k, n_cols)

    sorted_indices = np.argsort(arr, axis=1)[:, ::-1]

    top_k_indices = sorted_indices[:, :k]
    top_k_values = np.take_along_axis(arr, top_k_indices, axis=1)

    return top_k_indices, top_k_values


def safe_mmap_reader(path, ids=None):
    """
    Open a file in memory-mapped mode.

    Parameters:
        path (str or Pathlike): The path to the file.
        ids (list): The slices to read from the file.

    Returns:
        np.ndarray: If the system is Windows, the file will be directly loaded into memory.
            Otherwise, the file will be memory-mapped.
    """
    if os.name == 'nt':
        mmap_mode = None
    else:
        mmap_mode = 'r'

    if ids is None:
        return np.asarray(memoryview(np.load(path, mmap_mode=mmap_mode)))

    return np.asarray(memoryview(np.load(path, mmap_mode=mmap_mode)[ids]))
