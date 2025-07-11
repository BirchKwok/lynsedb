from typing import Type, Dict, Any
from .base import BaseIndex


class IndexFactory:
    """Factory class for creating index instances."""

    _registry: Dict[str, Type[BaseIndex]] = {}

    @classmethod
    def register(cls, name: str):
        """Register an index class with the given name."""
        def decorator(index_class: Type[BaseIndex]):
            cls._registry[name] = index_class
            return index_class
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseIndex:
        """Create an index instance of the given type."""
        if name not in cls._registry:
            raise ValueError(f"Index type '{name}' not registered")
        return cls._registry[name](**kwargs)

    @classmethod
    def get_registered_types(cls) -> list:
        """Get a list of all registered index types."""
        return list(cls._registry.keys())
