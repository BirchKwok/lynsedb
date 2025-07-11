from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np
import cloudpickle

from ..computational_layer.engines import ip, l2sq, cosine, hamming, jaccard, get_simsimd_capabilities, set_simsimd_default
from ..core_components.io import save_nnp, load_nnp
from .distance import DistanceFactory
from .quantizer import QuantizerFactory


class BaseIndex(ABC):
    """索引基类,定义通用接口。"""

    def __init__(self, distance_metric: str = 'l2', quantizer: str = 'none', use_simd: Optional[bool] = None, **kwargs):
        """
        初始化索引。

        参数:
            distance_metric: 距离度量类型
            quantizer: 量化器类型
            use_simd: 是否使用SIMD加速，None表示使用默认设置
            **kwargs: 额外参数
        """
        self.distance = DistanceFactory.create(distance_metric)
        self.quantizer = QuantizerFactory.create(quantizer, **kwargs)
        self.use_simd = use_simd  # None表示使用engines中的默认设置

        self.data = None  # 原始数据
        self.encoded_data = None  # 编码后的数据
        self.ids = None  # 数据ID
        self.metadata = {}  # 元数据
        self.is_trained = False  # 是否已训练

        # 保存SimSIMD能力信息
        self.simd_capabilities = get_simsimd_capabilities()

        # 保存构造参数用于序列化
        self._init_params = {
            'distance_metric': distance_metric,
            'quantizer': quantizer,
            'use_simd': use_simd,
            **kwargs
        }

    def _distance_single(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算两个向量间的距离。"""
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        # If encoded data is not a floating type (e.g., uint8 from SQ8) use decoded vectors directly
        if x.dtype not in (np.float32, np.float16) or y.dtype not in (np.float32, np.float16):
            x_dec = self.quantizer.decode(x)
            y_dec = self.quantizer.decode(y)
            return self.distance.compute(x_dec.squeeze(), y_dec.squeeze())

        if self.distance.name == 'l2':
            _, distances = l2sq(x, y, 1, use_simd=self.use_simd)
        elif self.distance.name == 'ip':
            _, distances = ip(x, y, 1, use_simd=self.use_simd)
        elif self.distance.name == 'cosine':
            _, distances = cosine(x, y, 1, use_simd=self.use_simd)
        elif self.distance.name == 'hamming':
            _, distances = hamming(x, y, 1, use_simd=self.use_simd)
        elif self.distance.name == 'jaccard':
            _, distances = jaccard(x, y, 1, use_simd=self.use_simd)
        else:
            # 回退到距离工厂方法
            return self.distance.compute(x.squeeze(), y.squeeze())
        return distances[0]

    def _distance_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算单个查询向量与批量数据的距离。"""
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        if x.dtype not in (np.float32, np.float16) or y.dtype not in (np.float32, np.float16):
            x_dec = self.quantizer.decode(x)
            y_dec = self.quantizer.decode(y)
            return self.distance.batch_compute(x_dec, y_dec)

        if self.distance.name == 'l2':
            _, distances = l2sq(x, y, y.shape[0], use_simd=self.use_simd)
        elif self.distance.name == 'ip':
            _, distances = ip(x, y, y.shape[0], use_simd=self.use_simd)
        elif self.distance.name == 'cosine':
            _, distances = cosine(x, y, y.shape[0], use_simd=self.use_simd)
        elif self.distance.name == 'hamming':
            _, distances = hamming(x, y, y.shape[0], use_simd=self.use_simd)
        elif self.distance.name == 'jaccard':
            _, distances = jaccard(x, y, y.shape[0], use_simd=self.use_simd)
        else:
            # 回退到距离工厂方法
            return self.distance.batch_compute(x, y)
        return distances

    def get_simd_info(self) -> Dict[str, Any]:
        """
        获取SimSIMD相关信息。

        返回:
            包含SimSIMD能力和配置的字典
        """
        return {
            'capabilities': self.simd_capabilities.copy(),
            'use_simd': self.use_simd,
            'distance_metric': self.distance.name,
            'data_type': str(self.encoded_data.dtype) if self.encoded_data is not None else None
        }

    def set_simd_usage(self, use_simd: Optional[bool]) -> None:
        """
        设置SIMD使用方式。

        参数:
            use_simd: True启用，False禁用，None使用默认设置
        """
        self.use_simd = use_simd
        # 更新初始化参数
        self._init_params['use_simd'] = use_simd

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """使用量化器编码向量。"""
        return self.quantizer.encode(vectors)

    def decode(self, encoded_vectors: np.ndarray) -> np.ndarray:
        """使用量化器解码向量。"""
        return self.quantizer.decode(encoded_vectors)

    def fit_transform(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        训练索引并转换数据。

        参数:
            vectors: 输入向量
            ids: 向量ID,如果为None则使用索引作为ID

        返回:
            编码后的数据
        """
        # 确保传入的是 numpy 数组而不是字典
        if isinstance(vectors, dict):
            if "data" in vectors:
                vectors = vectors["data"]
            else:
                # 如果没有 "data" 键，使用第一个数组
                vectors = list(vectors.values())[0]

        # 确保是 numpy 数组
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        if not self.is_trained:
            # 训练量化器
            self.quantizer.fit(vectors)
            self.is_trained = True

        # 编码数据
        encoded = self.encode(vectors)

        # 确保编码数据是合适的数值类型，避免 object 类型
        if hasattr(encoded, 'dtype') and encoded.dtype == object:
            # 如果编码结果是 object 类型，尝试转换为 float32
            try:
                encoded = np.array(encoded, dtype=np.float32)
            except (ValueError, TypeError):
                # 如果无法转换，保持原样但在后续处理中会失败
                pass

        # 更新状态
        if self.encoded_data is None:
            self.encoded_data = encoded.copy()
            self.data = vectors.copy()
            self.ids = np.arange(len(vectors)) if ids is None else np.array(ids, copy=True)
        else:
            # 确保数据类型一致
            if self.encoded_data.dtype != encoded.dtype:
                # 尝试将两个数组转换为公共类型
                common_dtype = np.find_common_type([self.encoded_data.dtype, encoded.dtype], [])
                if common_dtype == object:
                    # 如果公共类型是 object，尝试使用 float32
                    common_dtype = np.float32

                try:
                    self.encoded_data = self.encoded_data.astype(common_dtype)
                    encoded = encoded.astype(common_dtype)
                except (ValueError, TypeError):
                    # 如果类型转换失败，保持原始类型但可能会在 vstack 时出错
                    pass

            # 使用确保类型一致的数组进行 vstack
            self.encoded_data = np.vstack([self.encoded_data, encoded])
            self.data = np.vstack([self.data, vectors])
            new_ids = np.arange(len(self.ids), len(self.ids) + len(vectors)) if ids is None else np.array(ids)
            self.ids = np.hstack([self.ids, new_ids])

        return encoded

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最近邻。

        参数:
            query: 查询向量
            k: 返回的邻居数量
            **kwargs: 额外参数

        返回:
            (indices, distances): 邻居的索引和距离
        """
        pass

    def save(self, filepath: str) -> None:
        """
        保存索引到文件。

        参数:
            filepath: 保存路径
        """
        state = {
            'init_params': self._init_params,
            'is_trained': self.is_trained,
            'metadata': self.metadata,
            'quantizer_state': cloudpickle.dumps(self.quantizer),
            'distance_state': cloudpickle.dumps(self.distance)
        }

        # 子类特定状态
        state.update(self._get_state())

        with open(filepath, 'wb') as f:
            cloudpickle.dump(state, f)

    def load(self, filepath: str) -> 'BaseIndex':
        """
        从文件加载索引。

        参数:
            filepath: 加载路径

        返回:
            加载后的索引实例
        """
        with open(filepath, 'rb') as f:
            state = cloudpickle.load(f)

        # 恢复基本状态
        self._init_params = state['init_params']
        self.is_trained = state['is_trained']
        self.metadata = state['metadata']
        self.quantizer = cloudpickle.loads(state['quantizer_state'])
        self.distance = cloudpickle.loads(state['distance_state'])

        # 恢复SimSIMD能力信息
        self.simd_capabilities = get_simsimd_capabilities()

        # 恢复子类特定状态
        self._set_state(state)

        return self

    def save_data(self, data_path: str, ids_path: str) -> None:
        """
        保存编码数据和ID到单独的文件，使用NumPack格式。

        参数:
            data_path: 数据保存路径
            ids_path: ID保存路径
        """
        # 使用NumPack保存数据，确保文件扩展名为.npk
        data_path = Path(data_path)
        ids_path = Path(ids_path)

        # 强制使用.npk扩展名
        if data_path.suffix != '.npk':
            data_path = data_path.with_suffix('.npk')
        if ids_path.suffix != '.npk':
            ids_path = ids_path.with_suffix('.npk')

        # 检查并转换编码数据类型
        encoded_data_to_save = self.encoded_data
        if encoded_data_to_save.dtype == object:
            try:
                # 尝试转换为 float32
                encoded_data_to_save = np.array(encoded_data_to_save, dtype=np.float32)
            except (ValueError, TypeError):
                try:
                    # 如果 float32 不行，尝试其他数值类型
                    encoded_data_to_save = np.array(encoded_data_to_save, dtype=np.float64)
                except (ValueError, TypeError):
                    raise ValueError(f"Cannot convert encoded_data with dtype {self.encoded_data.dtype} to a NumPack-compatible type")

        # 检查并转换 IDs 数据类型
        ids_to_save = self.ids
        if ids_to_save.dtype == object:
            try:
                # 尝试转换为 int64
                ids_to_save = np.array(ids_to_save, dtype=np.int64)
            except (ValueError, TypeError):
                try:
                    # 如果 int64 不行，尝试 float64
                    ids_to_save = np.array(ids_to_save, dtype=np.float64)
                except (ValueError, TypeError):
                    raise ValueError(f"Cannot convert ids with dtype {self.ids.dtype} to a NumPack-compatible type")

        # 保存编码数据
        save_nnp(str(data_path), encoded_data=encoded_data_to_save)

        # 保存ID数据
        save_nnp(str(ids_path), ids=ids_to_save)

    def load_data(self, data_path: str, ids_path: str) -> None:
        """
        从文件加载编码数据和ID，使用NumPack格式。

        参数:
            data_path: 数据加载路径
            ids_path: ID加载路径
        """
        # 使用NumPack加载数据，确保文件扩展名为.npk
        data_path = Path(data_path)
        ids_path = Path(ids_path)

        # 强制使用.npk扩展名
        if data_path.suffix != '.npk':
            data_path = data_path.with_suffix('.npk')
        if ids_path.suffix != '.npk':
            ids_path = ids_path.with_suffix('.npk')

        # 加载编码数据
        data_arrays = load_nnp(str(data_path))
        self.encoded_data = data_arrays['encoded_data']

        # 加载ID数据
        ids_arrays = load_nnp(str(ids_path))
        self.ids = ids_arrays['ids']

    @abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        """获取子类特定的状态用于序列化。"""
        pass

    @abstractmethod
    def _set_state(self, state: Dict[str, Any]) -> None:
        """从序列化状态恢复子类特定的状态。"""
        pass

    def delete(self, ids: Union[int, List[int], np.ndarray]) -> None:
        """
        删除指定ID的向量。

        参数:
            ids: 要删除的向量ID，可以是单个ID或ID列表
        """
        if isinstance(ids, (int, np.integer)):
            ids = [ids]

        # 转换为numpy数组
        ids = np.array(ids)

        # 找到要保留的索引
        mask = np.isin(self.ids, ids, invert=True)

        # 更新数据
        self.encoded_data = self.encoded_data[mask]
        if self.data is not None:
            self.data = self.data[mask]
        self.ids = self.ids[mask]

        # 子类特定的删除操作
        self._delete_impl(ids)

    def update(self, vectors: np.ndarray, ids: Union[int, List[int], np.ndarray]) -> None:
        """
        更新指定ID的向量。

        参数:
            vectors: 新的向量数据
            ids: 要更新的向量ID，可以是单个ID或ID列表
        """
        # 确保vectors和ids的维度匹配
        vectors = np.atleast_2d(vectors)
        if isinstance(ids, (int, np.integer)):
            ids = [ids]
        ids = np.array(ids)

        if len(vectors) != len(ids):
            raise ValueError("vectors和ids的长度必须相同")

        # 删除旧的向量
        self.delete(ids)

        # 添加新的向量
        self.fit_transform(vectors, ids)

    @abstractmethod
    def _delete_impl(self, ids: np.ndarray) -> None:
        """
        子类需要实现的删除操作。

        参数:
            ids: 要删除的向量ID
        """
        pass
