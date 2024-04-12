from typing import Callable

from spinesUtils.logging import Logger

from min_vec.storage_layer.storage import StorageWorker
from min_vec.data_structures.kmeans import BatchKMeans


class ClusterWorker:
    def __init__(
            self,
            logger: Logger,
            iterable_dataloader: Callable,
            ann_model: BatchKMeans,
            storage_worker: StorageWorker,
            save_data: Callable,
            n_clusters: int
    ):
        self.logger = logger
        self.iterable_dataloader = iterable_dataloader
        self.ann_model = ann_model
        self.storage_worker = storage_worker
        self.save_data = save_data
        self.n_clusters = n_clusters

    def _kmeans_clustering(self, data):
        """kmeans clustering"""
        self.ann_model = self.ann_model.partial_fit(data)

        return self.ann_model.labels_

    def build_index(self, scaler=None, distance='cosine'):
        """
        Build the IVF index more efficiently.
        """
        max_len = 0
        # 初始化每个聚类的存储列表
        temp_clusters = {i: ([], [], []) for i in range(self.n_clusters)}

        if scaler is not None and distance == 'L2':
            decoder = scaler.decode
        else:
            decoder = None

        for data, indices, fields in self.iterable_dataloader(read_chunk_only=True, mode='lazy'):
            decode_data = data if decoder is None else decoder(data)
            labels = self._kmeans_clustering(decode_data)

            # 直接按标签将数据分配到相应的聚类
            for d, idx, f, label in zip(data, indices, fields, labels):
                temp_clusters[label][0].append(d)
                temp_clusters[label][1].append(idx)
                temp_clusters[label][2].append(f)
                max_len += 1

            if max_len >= 10000:
                # 遍历每个聚类，保存数据
                for cluster_id, (d, idx, f) in temp_clusters.items():
                    if d:  # 检查是否有数据，避免保存空聚类
                        self.save_data(d, idx, f, write_chunk=False, cluster_id=cluster_id)

                # 初始化每个聚类的存储列表
                temp_clusters = {i: ([], [], []) for i in range(self.n_clusters)}
                max_len = 0

        if max_len > 0:
            for cluster_id, (d, idx, f) in temp_clusters.items():
                if d:
                    self.save_data(d, idx, f, write_chunk=False, cluster_id=cluster_id)

        self.storage_worker.delete_chunk()
