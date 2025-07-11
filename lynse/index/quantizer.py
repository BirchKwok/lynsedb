from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Type, Optional, Tuple


class Quantizer(ABC):
    """量化器的抽象基类。"""

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """训练量化器。"""
        pass

    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """将数据编码为量化形式。"""
        pass

    @abstractmethod
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """将量化数据解码回原始形式。"""
        pass


class NoQuantizer(Quantizer):
    """不进行量化的空量化器。"""

    def fit(self, data: np.ndarray) -> None:
        pass

    def encode(self, data: np.ndarray) -> np.ndarray:
        return data

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return codes


class ScalarQuantizer(Quantizer):
    """标量量化器,使用固定位数对每个维度进行量化。"""

    def __init__(self, bits: int = 8):
        """
        初始化标量量化器。

        参数:
            bits: 每个维度使用的位数
        """
        self.bits = bits
        self.scale = None
        self.min_val = None
        self.max_val = None

    def fit(self, data: np.ndarray) -> None:
        """训练量化器,计算缩放因子。"""
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        self.scale = (self.max_val - self.min_val) / (2**self.bits - 1)
        # 处理scale为0的情况
        self.scale[self.scale == 0] = 1

    def encode(self, data: np.ndarray) -> np.ndarray:
        """将数据编码为量化形式。"""
        if self.scale is None:
            raise ValueError("Quantizer must be fitted before encoding")

        normalized = (data - self.min_val) / self.scale
        return np.clip(normalized, 0, 2**self.bits - 1).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """将量化数据解码回原始形式。"""
        if self.scale is None:
            raise ValueError("Quantizer must be fitted before decoding")

        return codes.astype(np.float32) * self.scale + self.min_val


class BinaryQuantizer(Quantizer):
    """二进制量化器,将向量转换为二进制形式。"""

    def __init__(self, threshold: float = 0.5):
        """
        初始化二进制量化器。

        参数:
            threshold: 二值化阈值
        """
        self.threshold = threshold

    def fit(self, data: np.ndarray) -> None:
        pass

    def encode(self, data: np.ndarray) -> np.ndarray:
        """将数据编码为二进制形式。"""
        return (data > self.threshold).astype(np.bool_)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """将二进制数据解码为浮点形式。"""
        return codes.astype(np.float32)


class ProductQuantizer(Quantizer):
    """乘积量化器,将向量分成子空间分别量化。"""

    def __init__(self, n_subspaces: int = 8, n_clusters: int = 256):
        """
        初始化乘积量化器。

        参数:
            n_subspaces: 子空间数量
            n_clusters: 每个子空间的聚类中心数量
        """
        self.n_subspaces = n_subspaces
        self.n_clusters = n_clusters
        self.subspace_size = None
        self.codebooks = None

    def fit(self, data: np.ndarray) -> None:
        """训练量化器,为每个子空间学习聚类中心。"""
        from sklearn.cluster import KMeans

        # Support both 1-D and 2-D inputs gracefully
        n_dims = data.shape[1] if data.ndim > 1 else data.shape[0]
        self.subspace_size = n_dims // self.n_subspaces

        # 初始化codebooks
        self.codebooks = []

        # 对每个子空间进行聚类
        for i in range(self.n_subspaces):
            start_dim = i * self.subspace_size
            end_dim = start_dim + self.subspace_size

            subspace_data = data[:, start_dim:end_dim]
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.fit(subspace_data)

            self.codebooks.append(kmeans.cluster_centers_)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """将数据编码为量化形式。"""
        if self.codebooks is None:
            raise ValueError("Quantizer must be fitted before encoding")

        n_vectors = len(data)
        codes = np.zeros((n_vectors, self.n_subspaces), dtype=np.uint8)

        for i in range(self.n_subspaces):
            start_dim = i * self.subspace_size
            end_dim = start_dim + self.subspace_size

            subspace_data = data[:, start_dim:end_dim]
            distances = np.sum((subspace_data[:, np.newaxis] - self.codebooks[i]) ** 2, axis=2)
            codes[:, i] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """将量化数据解码回原始形式。"""
        if self.codebooks is None:
            raise ValueError("Quantizer must be fitted before decoding")

        n_vectors = len(codes)
        decoded = np.zeros((n_vectors, self.subspace_size * self.n_subspaces))

        for i in range(self.n_subspaces):
            start_dim = i * self.subspace_size
            end_dim = start_dim + self.subspace_size
            decoded[:, start_dim:end_dim] = self.codebooks[i][codes[:, i]]

        return decoded


class QuantizerFactory:
    """量化器工厂类。"""

    _registry: Dict[str, Type[Quantizer]] = {
        'none': NoQuantizer,
        'sq': ScalarQuantizer,
        'sq8': ScalarQuantizer,  # Alias for 8-bit scalar quantizer used in index specs like "-SQ8"
        'binary': BinaryQuantizer,
        'pq': ProductQuantizer
    }

    @classmethod
    def register(cls, name: str) -> callable:
        """注册新的量化器类。"""
        def decorator(quantizer_class: Type[Quantizer]) -> Type[Quantizer]:
            cls._registry[name] = quantizer_class
            return quantizer_class
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[Quantizer]:
        """创建量化器实例。"""
        if name not in cls._registry:
            raise ValueError(f"Unknown quantizer: {name}")
        return cls._registry[name](**kwargs)
