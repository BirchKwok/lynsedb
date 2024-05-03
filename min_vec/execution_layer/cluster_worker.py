from typing import Callable

from spinesUtils.asserts import raise_if
from spinesUtils.logging import Logger

from min_vec.storage_layer.storage import PersistentFileStorage
from min_vec.core_components.kmeans import BatchKMeans


class ClusterWorker:
    def __init__(
            self,
            logger: Logger,
            iterable_dataloader: Callable,
            ann_model: BatchKMeans,
            storage_worker: PersistentFileStorage,
            n_clusters: int
    ):
        self.logger = logger
        self.iterable_dataloader = iterable_dataloader
        self.ann_model = ann_model
        self.storage_worker = storage_worker
        self.n_clusters = n_clusters

    def _kmeans_clustering(self, data, refit=False):
        """kmeans clustering"""
        if refit:
            self.ann_model = self.ann_model.partial_fit(data)
            labels = self.ann_model.labels_
        else:
            raise_if(ValueError, not self.ann_model.fitted, "The model has not been fitted yet.")
            labels = self.ann_model.predict(data)

        return labels

    def build_index(self, refit=False):
        """
        Build the IVF index more efficiently.
        """
        max_len = 0
        # 初始化每个聚类的存储列表
        temp_clusters = {i: ([], []) for i in range(self.n_clusters)}

        if refit:
            # move all files to chunk
            self.storage_worker.move_all_files_to_chunk()

        filenames = self.storage_worker.get_all_files(read_type='chunk')

        for filename in filenames:
            data, indices = self.iterable_dataloader(filename)
            labels = self._kmeans_clustering(data, refit=refit)

            # 直接按标签将数据分配到相应的聚类
            for d, idx, label in zip(data, indices, labels):
                temp_clusters[label][0].append(d)
                temp_clusters[label][1].append(idx)
                max_len += 1

            if max_len >= 10000:
                # 遍历每个聚类，保存数据
                for cluster_id, (d, idx) in temp_clusters.items():
                    if d:  # 检查是否有数据，避免保存空聚类
                        self.storage_worker.write(d, idx, write_type='cluster', cluster_id=cluster_id)

                # 初始化每个聚类的存储列表
                temp_clusters = {i: ([], []) for i in range(self.n_clusters)}
                max_len = 0

        if max_len > 0:
            for cluster_id, (d, idx) in temp_clusters.items():
                if d:
                    self.storage_worker.write(d, idx, write_type='cluster', cluster_id=cluster_id)

        self.storage_worker.delete_chunk()
