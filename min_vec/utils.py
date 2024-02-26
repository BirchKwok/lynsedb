"""utils.py: this file contains some useful functions and decorators."""

from functools import wraps

from min_vec.config import MVDB_COSINE_SIMILARITY_THRESHOLD


class UnKnownError(Exception):
    pass


def io_checker(func):
    from functools import wraps
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


class VectorCache:
    """A decorator that caches the results of a function call with the same arguments.
        Only use for DatabaseQuery.query function.
    """

    def __init__(self, max_size=1000):
        from collections import OrderedDict
        from min_vec.engines import cosine_distance

        self.cache = OrderedDict()
        self.max_size = max_size
        self._cosine_similarity = cosine_distance

    def clear_cache(self):
        self.cache.clear()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import numpy as np

            if len(args) < 2:
                query_vec = kwargs['vector']
            else:
                query_vec = args[1]  # this argument is the query vector

            best_similarity = 0.0
            best_result = None

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
                return self.cache[key]
            else:
                if MVDB_COSINE_SIMILARITY_THRESHOLD is not None:
                    # 遍历缓存中的向量，找到最相似的向量
                    for cached_key, cached_vec in self.cache.items():
                        for cv in cached_vec:
                            if cv[0] == 'vector':
                                cached_vec = np.frombuffer(cv, dtype=np.float32)

                                similarity = self._cosine_similarity(query_vec, cached_vec)
                                if similarity > MVDB_COSINE_SIMILARITY_THRESHOLD:
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        best_result = self.cache[key]
                                        break

            # 如果没有找到足够相似的结果，则执行原函数获取新的结果并更新缓存
            if best_result is None:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)

                result = func(*args, **kwargs)
                best_result = result
                self.cache[key] = result

            return best_result

        wrapper.clear_cache = self.clear_cache
        return wrapper
