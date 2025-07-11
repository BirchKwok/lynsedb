import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional


class DistanceMetric(ABC):
    """距离度量的抽象基类。"""

    def __init__(self):
        self.name = None  # 子类需要设置这个属性

    @abstractmethod
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算两个向量之间的距离。"""
        pass

    @abstractmethod
    def batch_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """批量计算向量之间的距离。"""
        pass


class L2Distance(DistanceMetric):
    """L2(欧几里得)距离。"""

    def __init__(self):
        super().__init__()
        self.name = 'l2'

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(np.sum((x - y) ** 2))

    def batch_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((x[:, np.newaxis] - y) ** 2, axis=2))


class IPDistance(DistanceMetric):
    """内积距离。"""

    def __init__(self):
        super().__init__()
        self.name = 'ip'

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        return -np.dot(x, y)

    def batch_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -np.dot(x, y.T)


class CosineDistance(DistanceMetric):
    """余弦距离。"""

    def __init__(self):
        super().__init__()
        self.name = 'cosine'

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        return -np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def batch_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        return -np.dot(x, y.T) / (x_norm[:, np.newaxis] * y_norm)


class JaccardDistance(DistanceMetric):
    """Jaccard距离(用于二进制向量)。"""

    def __init__(self):
        super().__init__()
        self.name = 'jaccard'

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        intersection = np.sum(x & y)
        union = np.sum(x | y)
        return 1.0 - intersection / union if union > 0 else 1.0

    def batch_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        intersection = np.dot(x, y.T)
        x_sum = x.sum(axis=1)
        y_sum = y.sum(axis=1)
        union = x_sum[:, np.newaxis] + y_sum - intersection
        return 1.0 - intersection / np.maximum(union, 1)


class HammingDistance(DistanceMetric):
    """汉明距离(用于二进制向量)。"""

    def __init__(self):
        super().__init__()
        self.name = 'hamming'

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(x != y)

    def batch_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sum(x[:, np.newaxis] != y, axis=2)


class DistanceFactory:
    """距离度量工厂类。"""

    _registry: Dict[str, Type[DistanceMetric]] = {
        'l2': L2Distance,
        'ip': IPDistance,
        'cosine': CosineDistance,
        'jaccard': JaccardDistance,
        'hamming': HammingDistance
    }

    @classmethod
    def register(cls, name: str) -> callable:
        """注册新的距离度量类。"""
        def decorator(distance_class: Type[DistanceMetric]) -> Type[DistanceMetric]:
            cls._registry[name] = distance_class
            return distance_class
        return decorator

    @classmethod
    def create(cls, name: str) -> Optional[DistanceMetric]:
        """创建距离度量实例。"""
        if name not in cls._registry:
            raise ValueError(f"Unknown distance metric: {name}")
        return cls._registry[name]()
