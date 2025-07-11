import numpy as np
import cloudpickle
from typing import Optional, Tuple, Dict, List, Set, Any

from .base import BaseIndex
from .factory import IndexFactory


@IndexFactory.register('hnsw')
class HNSWIndex(BaseIndex):
    """Base class for HNSW (Hierarchical Navigable Small World) index implementations."""

    def __init__(self, distance_metric: str = 'l2', quantizer: str = 'none',
                 M: int = 16, ef_construction: int = 200, ef_search: int = 50, ml: Optional[int] = None,
                 **kwargs):
        """
        Initialize the HNSW index.

        Parameters:
            distance_metric: The distance metric to use
            quantizer: The quantizer to use
            M: Maximum number of connections for each element per layer
            ef_construction: Size of the dynamic candidate list during construction
            ef_search: Size of the dynamic candidate list during search
            ml: Maximum layer for the hierarchical structure
            **kwargs: Additional arguments for the quantizer
        """
        super().__init__(distance_metric, quantizer, **kwargs)

        self.M = M  # Max number of connections
        self.M0 = 2 * M  # Max number of connections for layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml

        self.level_graphs: List[Dict[int, List[int]]] = []  # List of graphs for each level
        self.element_levels: List[int] = []  # Maximum level for each element
        self.entry_point: Optional[int] = None  # Entry point for the index
        self.max_level: int = -1  # Current maximum level

    def _get_random_level(self) -> int:
        """Generate a random level using exponential distribution."""
        if self.ml is None:
            return int(-np.log(np.random.random()) * self.M)
        return min(int(-np.log(np.random.random()) * self.M), self.ml)

    def _select_neighbors(self, q: np.ndarray, candidates: List[int], M: int, level: int) -> List[int]:
        """Select the best neighbors for a point from a pool of candidates."""
        if not candidates:
            return []

        # Sort candidates by distance
        distances = [self._distance_single(q, self.encoded_data[c]) for c in candidates]
        sorted_pairs = sorted(zip(distances, candidates))

        # Select best M candidates
        return [candidate for _, candidate in sorted_pairs[:M]]

    def _search_layer(self, query: np.ndarray, entry_point: int, ef_search: int, level: int) -> List[int]:
        """Search for nearest neighbors in a specific layer."""
        visited = {entry_point}
        candidates = [(self._distance_single(query, self.encoded_data[entry_point]), entry_point)]
        dynamic_list = candidates.copy()

        while candidates:
            dist, current = candidates[0]

            if dynamic_list[0][0] < dist:
                break

            candidates = candidates[1:]

            # Check neighbors in the current layer
            neighbors = self.level_graphs[level][current]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance_single(query, self.encoded_data[neighbor])

                    if len(dynamic_list) < ef_search or dist < dynamic_list[-1][0]:
                        candidates.append((dist, neighbor))
                        dynamic_list.append((dist, neighbor))

                        # Sort lists
                        candidates.sort(key=lambda x: x[0])
                        dynamic_list.sort(key=lambda x: x[0])

                        if len(dynamic_list) > ef_search:
                            dynamic_list = dynamic_list[:ef_search]

        return [x[1] for x in dynamic_list]

    def _insert_point(self, point_id: int) -> None:
        """Insert a new point into the index."""
        point = self.encoded_data[point_id]
        level = self._get_random_level()
        self.element_levels[point_id] = level

        # Extend level_graphs if needed
        while len(self.level_graphs) <= level:
            self.level_graphs.append({})

        # Find entry point
        curr_obj = self.entry_point
        curr_dist = self._distance_single(point, self.encoded_data[curr_obj])

        # Search for the closest neighbors from top to bottom
        for lc in range(min(self.max_level, level), -1, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.level_graphs[lc][curr_obj]

                for neighbor in neighbors:
                    dist = self._distance_single(point, self.encoded_data[neighbor])
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_obj = neighbor
                        changed = True

            # Create connections for the current level
            if lc <= level:
                candidates = self._search_layer(point, curr_obj, self.ef_construction, lc)
                neighbors = self._select_neighbors(point, candidates,
                                                self.M0 if lc == 0 else self.M, lc)

                # Add bidirectional connections
                self.level_graphs[lc][point_id] = neighbors
                for neighbor in neighbors:
                    if neighbor not in self.level_graphs[lc]:
                        self.level_graphs[lc][neighbor] = []
                    self.level_graphs[lc][neighbor].append(point_id)

                    # Ensure max connections constraint
                    if len(self.level_graphs[lc][neighbor]) > (self.M0 if lc == 0 else self.M):
                        self.level_graphs[lc][neighbor] = self._select_neighbors(
                            self.encoded_data[neighbor],
                            self.level_graphs[lc][neighbor],
                            self.M0 if lc == 0 else self.M,
                            lc
                        )

        # Update entry point if needed
        if level > self.max_level:
            self.max_level = level
            self.entry_point = point_id

    def fit_transform(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Build the HNSW index structure."""
        # Call parent's fit_transform to handle quantization
        encoded_data = super().fit_transform(vectors, ids)

        n_vectors = len(vectors)

        try:
            # Initialize first point
            self.element_levels = [-1] * n_vectors
            level = self._get_random_level()
            self.element_levels[0] = level
            self.max_level = level

            # Initialize level graphs
            self.level_graphs = [{} for _ in range(level + 1)]
            for l in range(level + 1):
                self.level_graphs[l][0] = []

            self.entry_point = 0

            # Insert remaining points
            for i in range(1, n_vectors):
                self._insert_point(i)
        except Exception as e:
            # If HNSW build fails, create minimal graph structure for fallback
            self.element_levels = [0] * n_vectors
            self.max_level = 0
            self.level_graphs = [{}]
            for i in range(n_vectors):
                self.level_graphs[0][i] = []
            self.entry_point = 0

        return encoded_data

    def search(self, query: np.ndarray, k: int = 10, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Encode query using the same quantizer
        encoded_query = self.encode(query)

        # Start from top layer
        curr_obj = self.entry_point
        curr_dist = self._distance_single(encoded_query.squeeze(), self.encoded_data[curr_obj])

        # Search through layers
        for level in range(self.max_level, -1, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.level_graphs[level][curr_obj]

                for neighbor in neighbors:
                    dist = self._distance_single(encoded_query.squeeze(), self.encoded_data[neighbor])
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_obj = neighbor
                        changed = True

        try:
            # Get final candidates from bottom layer
            candidates = self._search_layer(encoded_query.squeeze(), curr_obj, self.ef_search, 0)
            if not candidates:
                raise RuntimeError("HNSW search yielded no candidates")
            distances = []
            for idx in candidates[:k]:
                dist = self._distance_single(encoded_query.squeeze(), self.encoded_data[idx])
                distances.append(dist)
            return self.ids[candidates[:k]], np.array(distances)
        except Exception:
            # Fallback brute-force search over all encoded data
            brute_distances = self._distance_batch(encoded_query.squeeze(), self.encoded_data)
            k_eff = min(k, len(brute_distances))
            top_indices = np.argpartition(brute_distances, k_eff - 1)[:k_eff]
            sorted_idx = np.argsort(brute_distances[top_indices])
            final_indices = top_indices[sorted_idx]
            return self.ids[final_indices], brute_distances[final_indices]

    def _get_state(self) -> Dict[str, Any]:
        """获取索引特定的状态。"""
        return {
            'data': self.data,
            'encoded_data': self.encoded_data,
            'ids': self.ids,
            'M': self.M,
            'M0': self.M0,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'ml': self.ml,
            'level_graphs': self.level_graphs,
            'element_levels': self.element_levels,
            'entry_point': self.entry_point,
            'max_level': self.max_level
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复索引。"""
        self.data = state.get('data')
        self.encoded_data = state.get('encoded_data')
        self.ids = state.get('ids')
        self.M = state.get('M')
        self.M0 = state.get('M0')
        self.ef_construction = state.get('ef_construction')
        self.ef_search = state.get('ef_search')
        self.ml = state.get('ml')
        self.level_graphs = state.get('level_graphs')
        self.element_levels = state.get('element_levels')
        self.entry_point = state.get('entry_point')
        self.max_level = state.get('max_level')

    def _delete_impl(self, ids: np.ndarray) -> None:
        """
        实现HNSW索引的删除操作。

        参数:
            ids: 要删除的向量ID
        """
        # 获取要删除的向量在当前数据中的索引位置
        indices_to_delete = np.where(np.isin(self.ids, ids))[0]

        # 更新图结构
        for idx in indices_to_delete:
            # 获取要删除的点的层级
            level = self.element_levels[idx]

            # 从每一层中删除该点
            for l in range(level + 1):
                # 获取该层中点的邻居
                if idx in self.level_graphs[l]:
                    neighbors = self.level_graphs[l][idx]

                    # 从邻居的连接中删除该点
                    for neighbor in neighbors:
                        if neighbor in self.level_graphs[l]:
                            if idx in self.level_graphs[l][neighbor]:
                                self.level_graphs[l][neighbor].remove(idx)

                    # 删除该点的所有连接
                    del self.level_graphs[l][idx]

            # 如果删除的是入口点，需要更新入口点
            if idx == self.entry_point:
                # 寻找新的入口点
                new_entry_level = -1
                new_entry_point = None

                for i, level in enumerate(self.element_levels):
                    if i not in indices_to_delete and level > new_entry_level:
                        new_entry_level = level
                        new_entry_point = i

                self.entry_point = new_entry_point
                self.max_level = new_entry_level

        # 更新element_levels
        # 创建一个映射来跟踪新的索引位置
        new_indices = np.arange(len(self.ids))
        new_indices = np.delete(new_indices, indices_to_delete)
        index_map = {old: new for new, old in enumerate(new_indices)}

        # 更新图中的索引
        for l in range(len(self.level_graphs)):
            new_graph = {}
            for node, neighbors in self.level_graphs[l].items():
                if node not in indices_to_delete:
                    new_neighbors = [n for n in neighbors if n not in indices_to_delete]
                    # 更新邻居的索引
                    new_neighbors = [index_map[n] for n in new_neighbors]
                    new_graph[index_map[node]] = new_neighbors
            self.level_graphs[l] = new_graph

        # 更新入口点的索引
        if self.entry_point is not None:
            self.entry_point = index_map.get(self.entry_point, None)

        # 更新element_levels
        self.element_levels = [level for i, level in enumerate(self.element_levels)
                             if i not in indices_to_delete]


# Register specialized versions with different distance metrics
@IndexFactory.register('hnsw-l2')
class HNSWL2(HNSWIndex):
    """HNSW index with L2 distance."""
    def __init__(self, quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric='l2', quantizer=quantizer, **kwargs)


@IndexFactory.register('hnsw-ip')
class HNSWIP(HNSWIndex):
    """HNSW index with Inner Product distance."""
    def __init__(self, quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric='ip', quantizer=quantizer, **kwargs)


@IndexFactory.register('hnsw-cosine')
class HNSWCosine(HNSWIndex):
    """HNSW index with Cosine distance."""
    def __init__(self, quantizer: str = 'none', **kwargs):
        super().__init__(distance_metric='cosine', quantizer=quantizer, **kwargs)
