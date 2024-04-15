"""utils.py: this file contains some useful functions and decorators."""

from functools import wraps


class UnKnownError(Exception):
    pass


def io_checker(func):
    from pathlib import Path
    from spinesUtils.asserts import raise_if

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except FileNotFoundError as e:
            raise_if(FileNotFoundError, True,
                     f"No such file or directory: '{Path(e.filename).absolute()}'")
        except PermissionError as e:
            raise_if(PermissionError, True,
                     f"No permission to read or write the '{Path(e.filename).absolute()}' file.")
        except IOError as e:
            raise_if(IOError, True, f"Encounter IOError "
                                    f"when read or write the '{Path(e.filename).absolute()}' file.")
        except Exception:
            raise_if(UnKnownError, True, f"Encounter Unknown Error "
                                         f"when read or write the file.")

    return wrapper


class QueryVectorCache:
    """A decorator that caches the results of a function call with the same arguments.
        Only use for DatabaseQuery.query function.
    """

    def __init__(self, max_size=1000):
        from collections import OrderedDict

        self.cache = OrderedDict()
        self.max_size = max_size

    def clear_cache(self):
        self.cache.clear()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成基于参数的唯一键，确保所有部分都是可哈希的
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

            # 将参数列表转换为元组，作为键
            key = tuple(key_parts)

            # 检查是否在缓存中找到对应的向量并计算相似度
            if key in self.cache:
                best_result = self.cache[key]
            else:
                best_result = func(*args, **kwargs)
                self.cache[key] = best_result

            return best_result

        wrapper.clear_cache = self.clear_cache
        return wrapper


def unavailable_if_deleted(func):
    """A decorator that detects if the function is called after the object is deleted."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # self is the first parameter
        if args[0]._matrix_serializer.IS_DELETED:
            print("The database has been deleted, and the operation is invalid.")
            return None
        return func(*args, **kwargs)

    return wrapper
