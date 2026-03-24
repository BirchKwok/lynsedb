import inspect
from functools import wraps


def raise_if(exception_type, condition, message=''):
    """Raise an exception of the given type if the condition is True.

    Parameters:
        exception_type: The exception class to raise.
        condition (bool): If True, the exception is raised.
        message (str): The error message.
    """
    if condition:
        raise exception_type(message)


def augmented_isinstance(obj, types):
    """Enhanced isinstance that treats None in the type tuple as type(None).

    Parameters:
        obj: The object to check.
        types (tuple or type): The type(s) to check against. None entries are
            converted to ``type(None)``.

    Returns:
        bool: True if obj is an instance of any of the given types.
    """
    if isinstance(types, tuple):
        types = tuple(type(None) if t is None else t for t in types)
    elif types is None:
        types = type(None)
    return isinstance(obj, types)


def generate_function_kwargs(func, *args, **kwargs):
    """Merge positional arguments into keyword arguments based on the function signature.

    Parameters:
        func: The function whose signature is inspected.
        *args: Positional arguments passed to the function.
        **kwargs: Keyword arguments passed to the function.

    Returns:
        dict: A dictionary mapping parameter names to their values.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    merged = {}
    for i, arg in enumerate(args):
        if i < len(params):
            merged[params[i]] = arg
    merged.update(kwargs)
    return merged


class ParameterTypeAssert:
    """Decorator that validates parameter types at call time.

    Parameters:
        type_dict (dict): Mapping of parameter names to expected types.
            Each value can be a single type or a tuple of types.
            ``None`` inside a tuple is treated as ``type(None)``.
        func_name (str, optional): Display name used in error messages.
    """

    _NoneType = type(None)

    def __init__(self, type_dict, func_name=None):
        self.type_dict = type_dict
        self.func_name = func_name

    @staticmethod
    def _normalize_type(t):
        """Normalize a type spec once: None → type(None), ensure tuple."""
        _NoneType = ParameterTypeAssert._NoneType
        if isinstance(t, tuple):
            return tuple(_NoneType if x is None else x for x in t)
        if t is None:
            return _NoneType
        return t

    def __call__(self, func):
        display_name = self.func_name or func.__name__

        # ── Pre-compute at decoration time (once) ──────────────────────
        param_names = tuple(inspect.signature(func).parameters.keys())
        # Map param_name → positional index for fast lookup from *args
        param_index = {name: i for i, name in enumerate(param_names)}

        # Only keep checks for params that exist in the signature,
        # with types already normalized.
        # Each entry: (param_name, positional_index, normalized_type)
        checks = []
        for name, expected in self.type_dict.items():
            if name in param_index:
                checks.append((name, param_index[name], self._normalize_type(expected)))

        # Fast-path: nothing to check
        if not checks:
            return func

        # Freeze as tuple for slightly faster iteration
        checks = tuple(checks)
        _isinstance = isinstance  # local ref avoids global lookup per call

        @wraps(func)
        def wrapper(*args, **kwargs):
            for name, idx, expected in checks:
                # Try positional args first (common path), then kwargs
                if idx < len(args):
                    value = args[idx]
                elif name in kwargs:
                    value = kwargs[name]
                else:
                    continue
                if not _isinstance(value, expected):
                    raise TypeError(
                        f"In `{display_name}`, parameter `{name}` expected "
                        f"{expected}, got {type(value).__name__}."
                    )
            return func(*args, **kwargs)

        return wrapper
