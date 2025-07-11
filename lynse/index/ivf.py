import numpy as np
import cloudpickle
from typing import Optional, Tuple, Dict, Any, List
from sklearn.cluster import KMeans

from .base import BaseIndex
from .factory import IndexFactory


@IndexFactory.register('ivf')
class IVFIndex(BaseIndex):
    """Base class for IVF (Inverted File) index implementations."""

    def __init__(self, distance_metric: str = 'l2', quantizer: str = 'none',
                 n_centroids: int = 256, nprobe: int = 32, **kwargs):
        """
        Initialize the IVF index.

        Parameters:
            distance_metric: The distance metric to use
            quantizer: The quantizer to use
            n_centroids: Number of cluster centroids for IVF
            nprobe: Number of clusters to search during query
            **kwargs: Additional arguments for the quantizer
        """
        super().__init__(distance_metric, quantizer, **kwargs)

        self.n_centroids = n_centroids
        self.nprobe = nprobe

        # IVF specific data structures
        self.centroids = None  # Cluster centroids
        self.inverted_lists = {}  # Maps cluster_id -> list of vector indices
        self.kmeans = None  # KMeans model for clustering

    def fit_transform(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Build the IVF index."""
        # Call parent's fit_transform to handle quantization
        encoded_data = super().fit_transform(vectors, ids)

        # Train centroids if not already trained
        if self.centroids is None:
            self._train_centroids(encoded_data)

        # Assign vectors to clusters
        self._assign_to_clusters(encoded_data)

        return encoded_data

    def _train_centroids(self, data: np.ndarray) -> None:
        """Train centroids using K-means clustering."""
        n_samples = data.shape[0]
        n_centroids = min(self.n_centroids, n_samples)

        self.kmeans = KMeans(n_clusters=n_centroids, random_state=42)
        self.kmeans.fit(data)
        self.centroids = self.kmeans.cluster_centers_

    def _assign_to_clusters(self, data: np.ndarray) -> None:
        """Assign vectors to their nearest clusters."""
        # Predict cluster assignments
        cluster_assignments = self.kmeans.predict(data)

        # Clear existing inverted lists
        self.inverted_lists = {}

        # Build inverted lists
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id not in self.inverted_lists:
                self.inverted_lists[cluster_id] = []
            self.inverted_lists[cluster_id].append(i)

    def search(self, query: np.ndarray, k: int = 10, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors using IVF."""
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Encode query using the same quantizer
        encoded_query = self.encode(query)

        # Find nearest centroids
        centroid_distances = self._distance_batch(encoded_query.squeeze(), self.centroids)
        nearest_centroids = np.argsort(centroid_distances)[:self.nprobe]

        # Collect candidates from nearest clusters
        candidates = []
        for centroid_id in nearest_centroids:
            if centroid_id in self.inverted_lists:
                candidates.extend(self.inverted_lists[centroid_id])

        if not candidates:
            # Fallback: use all vectors if no candidates found
            candidates = list(range(len(self.encoded_data)))

        # Limit candidates to avoid memory issues
        candidates = candidates[:min(len(candidates), k * 100)]

        # Calculate distances to all candidates
        candidate_data = self.encoded_data[candidates]
        distances = self._distance_batch(encoded_query.squeeze(), candidate_data)

        # Get top k
        if k > len(distances):
            k = len(distances)

        top_indices = np.argsort(distances)[:k]
        result_indices = np.array(candidates)[top_indices]
        result_distances = distances[top_indices]

        return self.ids[result_indices], result_distances

    def _get_state(self) -> Dict[str, Any]:
        """获取索引特定的状态。"""
        return {
            'data': self.data,
            'encoded_data': self.encoded_data,
            'ids': self.ids,
            'n_centroids': self.n_centroids,
            'nprobe': self.nprobe,
            'centroids': self.centroids,
            'inverted_lists': self.inverted_lists,
            'kmeans': cloudpickle.dumps(self.kmeans) if self.kmeans else None
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复索引。"""
        self.data = state.get('data')
        self.encoded_data = state.get('encoded_data')
        self.ids = state.get('ids')
        self.n_centroids = state.get('n_centroids')
        self.nprobe = state.get('nprobe')
        self.centroids = state.get('centroids')
        self.inverted_lists = state.get('inverted_lists', {})

        kmeans_data = state.get('kmeans')
        if kmeans_data:
            self.kmeans = cloudpickle.loads(kmeans_data)
        else:
            self.kmeans = None

    def _delete_impl(self, ids: np.ndarray) -> None:
        """
        实现基类要求的删除操作。对于IVF索引，需要重新构建倒排列表。

        参数:
            ids: 要删除的向量ID
        """
        # 重新分配剩余向量到集群
        if self.encoded_data is not None and len(self.encoded_data) > 0:
            self._assign_to_clusters(self.encoded_data)


# Register specialized versions with different distance metrics
@IndexFactory.register('ivf-l2')
class IVFL2(IVFIndex):
    """IVF index with L2 distance."""
    def __init__(self, quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric='l2', quantizer=quantizer, **kwargs)


@IndexFactory.register('ivf-ip')
class IVFIP(IVFIndex):
    """IVF index with Inner Product distance."""
    def __init__(self, quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric='ip', quantizer=quantizer, **kwargs)


@IndexFactory.register('ivf-cosine')
class IVFCosine(IVFIndex):
    """IVF index with Cosine distance."""
    def __init__(self, quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric='cosine', quantizer=quantizer, **kwargs)


@IndexFactory.register('ivf-jaccard')
class IVFJaccard(IVFIndex):
    """IVF index with Jaccard distance."""
    def __init__(self, quantizer: str = 'binary', **kwargs):
        super().__init__(distance_metric='jaccard', quantizer=quantizer, **kwargs)


@IndexFactory.register('ivf-hamming')
class IVFHamming(IVFIndex):
    """IVF index with Hamming distance."""
    def __init__(self, quantizer: str = 'binary', **kwargs):
        super().__init__(distance_metric='hamming', quantizer=quantizer, **kwargs)
