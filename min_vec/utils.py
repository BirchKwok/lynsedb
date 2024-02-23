"""utils.py: this file contains some useful functions and decorators."""


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
        except Exception as e:
            raise_if(UnKnownError, True, f"Encounter Unknown Error "
                                      f"when read or write the file.")

    return wrapper


def get_env_variable(name, default=None, default_type=str):
    import os

    def type_cast(value):
        if default_type == str:
            return value  # include None
        else:
            try:
                return default_type(value)  # will raise Exception if None
            except Exception:
                return default  # include None

    if default is None:
        return type_cast(os.environ.get(name))
    else:
        return type_cast(os.environ.get(name, default))


def vectors_cache(max_size=1000):
    from collections import OrderedDict
    from functools import wraps

    cache = OrderedDict()

    def clear_cache():
        cache.clear()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成基于参数的唯一键，确保所有部分都是可哈希的
            key_parts = []
            for arg in args[1:]:  # 跳过self参数
                if hasattr(arg, 'tobytes'):
                    # 对于numpy数组，使用其字节表示
                    key_parts.append(arg.tobytes())
                elif hasattr(arg, '__dict__'):
                    # 对于对象，使用其字典表示
                    key_parts.append(arg.__dict__)
                elif isinstance(arg, list):
                    key_parts.append(tuple(arg))
                else:
                    # 其他可哈希的参数直接使用
                    key_parts.append(arg)

            for k, v in kwargs.items():
                if hasattr(v, 'tobytes'):
                    key_parts.append((k, v.tobytes()))
                elif hasattr(v, '__dict__'):
                    # 对于对象，使用其字典表示
                    key_parts.append(v.__dict__)
                elif isinstance(v, list):
                    key_parts.append(tuple(v))
                else:
                    key_parts.append((k, v))

            # 将参数列表转换为元组，作为键
            key = tuple(key_parts)

            if key in cache:
                return cache[key]
            else:
                if len(cache) >= max_size:
                    cache.popitem(last=False)
                res = func(*args, **kwargs)
                cache[key] = res
                return res

        wrapper.clear_cache = clear_cache

        return wrapper

    decorator.clear_cache = clear_cache
    return decorator
