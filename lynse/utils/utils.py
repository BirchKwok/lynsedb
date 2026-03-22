from collections import OrderedDict
import os
import re
import threading
from contextlib import contextmanager
from threading import Lock
import time
from functools import wraps
from pathlib import Path


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
        obj = args[0]
        if hasattr(obj, '_IS_DELETED'):
            # Rust-backed ExclusiveDB
            if obj._IS_DELETED:
                db_name = Path(obj._database_path).name
                raise OpsError(f"The collection `{db_name}` has been deleted, and the `{func.__name__}` function "
                               f"is unavailable.")
        elif hasattr(obj, '_matrix_serializer'):
            # Legacy Python-backed ExclusiveDB
            db_name = Path(obj._database_path).name
            if obj._matrix_serializer.IS_DELETED:
                raise OpsError(f"The collection `{db_name}` has been deleted, and the `{func.__name__}` function "
                               f"is unavailable.")
        elif hasattr(obj, 'STATUS'):
            # LocalClient
            db_name = Path(obj.root_path).name
            if obj.STATUS == 'DELETED':
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


class FileHandlePool:
    def __init__(self, max_open_files=100):
        self.max_open_files = max_open_files
        self.open_files = OrderedDict()
        self.lock = Lock()

    def get_file_handle(self, file_path):
        from ..core_components.io import load_nnp
        with self.lock:
            if file_path in self.open_files:
                # If the file has already been opened, move it to the end of the OrderedDict
                self.open_files.move_to_end(file_path)
                return self.open_files[file_path]

            # If the maximum number of open files has been reached, close the oldest file
            if len(self.open_files) >= self.max_open_files:
                oldest_file, oldest_handle = self.open_files.popitem(last=False)
                self._close_handle(oldest_handle)

            # open new file
            file_handle = load_nnp(file_path, mmap_mode=True)
            self.open_files[file_path] = file_handle
            return file_handle

    def _close_handle(self, handle):
        """安全地关闭文件句柄"""
        if isinstance(handle, dict):
            # 如果是字典（load_nnp 的返回值），关闭字典中的每个 memmap 对象
            for array_name, array in handle.items():
                if hasattr(array, '_mmap') and array._mmap is not None:
                    try:
                        array._mmap.close()
                    except (AttributeError, OSError):
                        # 如果关闭失败，静默忽略
                        pass
        elif hasattr(handle, '_mmap') and handle._mmap is not None:
            # 如果是单个 memmap 对象
            try:
                handle._mmap.close()
            except (AttributeError, OSError):
                # 如果关闭失败，静默忽略
                pass

    def close_all(self):
        with self.lock:
            for handle in self.open_files.values():
                self._close_handle(handle)
            self.open_files.clear()

    def __del__(self):
        try:
            self.close_all()
        except:
            # 在析构函数中，忽略所有错误
            pass


class SafeMmapReader:
    def __init__(self, max_open_files=100):
        from ..core_components.locks import ThreadLock

        self.file_pool = FileHandlePool(max_open_files)
        self.lock = ThreadLock()

    def load_nnp(self, path, ids=None):
        try:
            with self.lock:
                mmap_handle = self.file_pool.get_file_handle(path)

            if ids is None:
                return mmap_handle
            return mmap_handle[ids]
        except Exception as e:
            raise IOError(f"Failed to load file {path}: {e}")

    def close(self):
        self.file_pool.close_all()

    def __del__(self):
        try:
            self.close()
        except:
            # 在析构函数中，忽略所有错误
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Use threading.local to create thread-local storage
thread_local = threading.local()


def inject_caller_info(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        caller_name = getattr(thread_local, 'caller_name', None)

        if caller_name is None:
            # If caller_name is not set, it means that the method is called directly
            with self.global_lock:
                return method(self, *args, **kwargs)
        else:
            # Otherwise, it means that the method is called by insert_session
            return method(self, *args, **kwargs)

    return wrapper


@contextmanager
def get_cursor(conn):
    """
    Get a cursor from a connection.

    Parameters:
        conn: sqlite3.Connection

    Yields:
        cursor: sqlite3.Cursor
    """
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()


def clean_file_when_finished(filename):
    """
    A decorator that cleans up the file when the function is finished.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                os.remove(filename)
        return wrapper
    return decorator
