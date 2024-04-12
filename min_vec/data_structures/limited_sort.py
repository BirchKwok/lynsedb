import threading
import numpy as np

from min_vec.computational_layer.engines import cosine_distance, argsort_topk


class LimitedSorted:
    def __init__(self, vector: np.ndarray, n: int, scaler):
        self.lock = threading.RLock()
        self.vector = vector
        self.n = n
        self.distance_func = cosine_distance
        self.scaler = scaler
        self.max_length = 1000  # 预设最大长度，应根据实际情况调整
        self.current_length = 0  # 当前有效数据长度
        self.similarities = np.zeros(self.max_length, dtype=vector.dtype)
        self.indices = np.zeros(self.max_length, dtype=int)
        self.matrix_subset = np.zeros((self.max_length, vector.size), dtype=vector.dtype)

    def add(self, sim: np.ndarray, indices: np.ndarray, matrix: np.ndarray):
        num_new_items = len(sim)
        with self.lock:
            end_pos = self.current_length + num_new_items
            if end_pos > self.max_length:
                # 如果超出预分配大小，则需要重新分配，但这种情况应尽量避免
                self.max_length = max(self.max_length * 2, end_pos)
                self.similarities = np.resize(self.similarities, self.max_length)
                self.indices = np.resize(self.indices, self.max_length)
                self.matrix_subset = np.resize(self.matrix_subset, (self.max_length, self.vector.size))

            # 填充新数据
            self.similarities[self.current_length:end_pos] = sim
            self.indices[self.current_length:end_pos] = indices
            self.matrix_subset[self.current_length:end_pos] = matrix

            # 更新当前长度
            self.current_length = end_pos

            # 根据相似度选择最佳的 n 个
            idx = argsort_topk(-self.similarities[:self.current_length], self.n)

            # 仅保留最佳的 n 个数据，其余数据可以丢弃或根据需要保留
            idx_len = len(idx)
            self.similarities[:idx_len] = self.similarities[idx]
            self.indices[:idx_len] = self.indices[idx]
            self.matrix_subset[:idx_len] = self.matrix_subset[idx]
            self.current_length = idx_len  # 更新当前长度为n

    def get_top_n(self):
        # 如果提供了scaler，对选中的向量解码
        decoded_vectors = self.scaler.decode(self.matrix_subset[:self.current_length])
        sim = self.distance_func(self.vector, decoded_vectors)

        # 根据相似度对索引进行排序
        sorted_idx = np.argsort(-sim)

        return self.indices[sorted_idx], sim[sorted_idx]

