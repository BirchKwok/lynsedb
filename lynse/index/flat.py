import numpy as np
import cloudpickle
from typing import Optional, Tuple, Dict, Any

from .base import BaseIndex
from .factory import IndexFactory


@IndexFactory.register('flat')
class FlatIndex(BaseIndex):
    """Base class for flat index implementations."""

    def __init__(self, distance_metric: str = 'l2', quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric, quantizer, **kwargs)

    def fit_transform(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Build the flat index."""
        # Call parent's fit_transform to handle quantization
        encoded_data = super().fit_transform(vectors, ids)
        return encoded_data

    def search(self, query: np.ndarray, k: int = 10, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Encode query using the same quantizer
        encoded_query = self.encode(query)

        # Compute distances
        distances = self._distance_batch(encoded_query.squeeze(), self.encoded_data)

        # Get top k
        if k > len(distances):
            k = len(distances)

        indices = np.argpartition(distances, k-1)[:k]
        distances = distances[indices]

        # Sort results
        sorted_idx = np.argsort(distances)
        indices = indices[sorted_idx]
        distances = distances[sorted_idx]

        return self.ids[indices], distances

    def _get_state(self) -> Dict[str, Any]:
        """获取索引特定的状态。"""
        return {
            'data': self.data,
            'encoded_data': self.encoded_data,
            'ids': self.ids
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复索引。"""
        self.data = state.get('data')
        self.encoded_data = state.get('encoded_data')
        self.ids = state.get('ids')

    def _delete_impl(self, ids: np.ndarray) -> None:
        """
        实现基类要求的删除操作。对于Flat索引，基类的delete方法已经完成了所有必要的操作，
        这里不需要额外的实现。

        参数:
            ids: 要删除的向量ID
        """
        pass  # 基类的delete方法已经处理了所有必要的操作


# Register specialized versions with different distance metrics
@IndexFactory.register('flat-l2')
class FlatL2(FlatIndex):
    """Flat index with L2 distance."""
    def __init__(self, quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric='l2', quantizer=quantizer, **kwargs)


@IndexFactory.register('flat-ip')
class FlatIP(FlatIndex):
    """Flat index with Inner Product distance."""
    def __init__(self, quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric='ip', quantizer=quantizer, **kwargs)


@IndexFactory.register('flat-cosine')
class FlatCosine(FlatIndex):
    """Flat index with Cosine distance."""
    def __init__(self, quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric='cosine', quantizer=quantizer, **kwargs)


@IndexFactory.register('flat-jaccard')
class FlatJaccard(FlatIndex):
    """Flat index with Jaccard distance."""
    def __init__(self, quantizer: str = 'binary', **kwargs):
        super().__init__(distance_metric='jaccard', quantizer=quantizer, **kwargs)


@IndexFactory.register('flat-hamming')
class FlatHamming(FlatIndex):
    """Flat index with Hamming distance."""
    def __init__(self, quantizer: str = 'binary', **kwargs):
        super().__init__(distance_metric='hamming', quantizer=quantizer, **kwargs)
