from typing import Optional, Dict, Any

from .factory import IndexFactory
from .distance import DistanceFactory
from .quantizer import QuantizerFactory


class IndexBuilder:
    """构建器类,用于灵活创建索引实例。"""

    def __init__(self):
        """初始化构建器。"""
        self._index_type = None
        self._distance = None
        self._quantizer = None
        self._params = {}

    def with_index_type(self, index_type: str) -> 'IndexBuilder':
        """设置索引类型。"""
        self._index_type = index_type
        return self

    def with_distance(self, distance: str) -> 'IndexBuilder':
        """设置距离度量。"""
        self._distance = distance
        return self

    def with_quantizer(self, quantizer: str) -> 'IndexBuilder':
        """设置量化器。"""
        self._quantizer = quantizer
        return self

    def with_params(self, **params) -> 'IndexBuilder':
        """设置额外参数。"""
        self._params.update(params)
        return self

    def build(self) -> Any:
        """构建索引实例。"""
        if self._index_type is None:
            raise ValueError("Index type must be specified")

        # 创建索引实例
        index = IndexFactory.create(
            self._index_type,
            distance_metric=self._distance or 'l2',
            quantizer=self._quantizer or 'none',
            **self._params
        )

        return index


class IndexConfig:
    """基于配置的索引创建类。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化配置。

        参数:
            config: 包含索引配置的字典
        """
        self.config = config

    def create_index(self) -> Any:
        """根据配置创建索引实例。"""
        builder = IndexBuilder()

        # 设置基本参数
        if 'index_type' in self.config:
            builder.with_index_type(self.config['index_type'])
        if 'distance' in self.config:
            builder.with_distance(self.config['distance'])
        if 'quantizer' in self.config:
            builder.with_quantizer(self.config['quantizer'])

        # 设置额外参数
        if 'params' in self.config:
            builder.with_params(**self.config['params'])

        return builder.build()


def create_index_from_string(spec: str) -> Any:
    """
    从字符串规范创建索引。

    参数:
        spec: 索引规范字符串,格式为: "index_type-distance[-quantizer]"
        例如: "hnsw-l2", "flat-ip-sq8"

    返回:
        创建的索引实例
    """
    parts = spec.lower().split('-')

    if len(parts) < 2:
        raise ValueError("Index specification must include at least index type and distance metric")

    index_type = parts[0]
    distance = parts[1]
    quantizer = parts[2] if len(parts) > 2 else 'none'

    return (
        IndexBuilder()
        .with_index_type(index_type)
        .with_distance(distance)
        .with_quantizer(quantizer)
        .build()
    )
