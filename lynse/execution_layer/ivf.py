from pathlib import Path
from typing import Callable, Union
import logging
import numpy as np
from spinesUtils.asserts import raise_if

from ..configs.config import config
from ..storage_layer.storage import PersistentFileStorage
from ..core_components.kmeans import EnhancedBatchKMeans
from ..core_components.ivf_index import IVFIndex
from ..core_components.locks import ThreadLock


class IVFCreator:
    def __init__(
            self,
            dataloader: Callable,
            storage_worker: PersistentFileStorage,
            collections_path_parent: Union[str, Path],
            drift_threshold: float = 0.3,
            quality_threshold: float = 0.5
    ):
        self.dataloader = dataloader
        self.ann_model = None
        self.storage_worker = storage_worker
        self.ivf_index_path = collections_path_parent
        self.ivf_index = IVFIndex(filepath=self.ivf_index_path)
        self.drift_threshold = drift_threshold
        self.quality_threshold = quality_threshold

        self.index_path = Path(self.ivf_index_path) / 'index'
        self.index_path.mkdir(parents=True, exist_ok=True)

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=str(self.index_path / 'ivf.log')
        )
        self.logger = logging.getLogger('IVFCreator')

    def _kmeans_clustering(self, data, rebuild=False):
        """kmeans clustering with quality monitoring"""
        if rebuild:
            self.ann_model = self.ann_model.partial_fit(data)
            labels = self.ann_model.labels_

            # 记录聚类质量
            quality_scores = self.ann_model.get_quality_trend()
            if quality_scores:
                self.logger.info(f"Current clustering quality score: {quality_scores[-1]:.4f}")
        else:
            raise_if(ValueError, not self.ann_model.fitted, "The model has not been fitted yet.")
            labels = self.ann_model.predict(data)

        return labels

    def _build_index(self, rebuild=False):
        """
        Build the IVF index more efficiently.
        """
        if rebuild:
            self.ivf_index = IVFIndex(self.ivf_index_path)

        filenames = self.storage_worker.get_all_files()
        already_indexed = self.ivf_index.all_external_ids()

        for filename in filenames:
            data_dict = self.dataloader(filename)

            # 从字典中提取实际的数组数据
            if isinstance(data_dict, dict):
                if "data" in data_dict:
                    data = data_dict["data"]
                else:
                    # 如果没有 "data" 键，使用第一个数组
                    data = list(data_dict.values())[0]
            else:
                # 向后兼容：如果直接返回数组
                data = data_dict

            indices = self.storage_worker.id_mapper[filename].generate_ids(as_range=False)

            isin = np.isin(indices, already_indexed)

            if isin.all():
                continue
            else:
                data = data[~isin]
                indices = indices[~isin]

            labels = self._kmeans_clustering(data, rebuild=rebuild)

            for idx, label in zip(indices, labels):
                self.ivf_index.add_entry(label, filename, idx)

    def _check_if_refit(self, n_clusters, always_refit=False):
        if self.ann_model is None or always_refit:
            self.ann_model = EnhancedBatchKMeans(
                n_clusters=n_clusters,
                batch_size=10240,
                epochs=config.LYNSE_KMEANS_EPOCHS,
                drift_threshold=self.drift_threshold,
                quality_threshold=self.quality_threshold
            )
            return True

        REFIT = False

        if (self.ann_model.n_clusters != n_clusters or
                self.ann_model.epochs != config.LYNSE_KMEANS_EPOCHS or
                self.ann_model.batch_size != 10240):
            self.ann_model = EnhancedBatchKMeans(
                n_clusters=n_clusters,
                batch_size=10240,
                epochs=config.LYNSE_KMEANS_EPOCHS,
                drift_threshold=self.drift_threshold,
                quality_threshold=self.quality_threshold
            )
            REFIT = True

        return REFIT

    def build_index(self, n_clusters=32, rebuild=False):
        """
        Build the index for clustering with quality monitoring.

        Parameters:
            n_clusters (int): The number of clusters.
            rebuild (bool): Whether to rebuild the index.

        Returns:
            None
        """
        raise_if(ValueError, not isinstance(n_clusters, int) or n_clusters <= 0,
                 'n_clusters must be int and greater than 0')
        raise_if(TypeError, not isinstance(rebuild, bool), 'rebuild must be bool')

        all_partition_size = self.storage_worker.get_shape()[0]
        if all_partition_size < 100000:
            return

        with (ThreadLock()):
            REBUILD = self._check_if_refit(n_clusters, always_refit=rebuild) or rebuild

            if REBUILD:
                self.logger.info("Rebuilding index from scratch...")
                self.remove_index()
                self._build_index(rebuild=True)
            else:
                already_clustered_rows = len(self.ivf_index)

                if already_clustered_rows == 0:
                    self._build_index(rebuild=True)
                elif already_clustered_rows == all_partition_size:
                    # 检查聚类质量
                    if self.ann_model.quality_scores_history and \
                            self.ann_model.quality_scores_history[-1] < self.quality_threshold:
                        self.logger.warning("Low clustering quality detected. Triggering rebuild...")
                        self._build_index(rebuild=True)
                    return
                else:
                    if all_partition_size - already_clustered_rows >= 100000:
                        self._build_index(rebuild=True)
                    else:
                        self._build_index(rebuild=False)

            # 保存模型和索引
            self.ann_model.save(self.index_path / 'ivf_ann_model')
            self.ivf_index.save(self.index_path / 'ivf_index')

            # 记录最终的聚类质量
            if self.ann_model.quality_scores_history:
                self.logger.info(f"Final clustering quality score: {self.ann_model.quality_scores_history[-1]:.4f}")

    def remove_index(self):
        """
        Remove the index.

        If all indices are removed, the index file will be removed.

        Returns:
            None
        """
        self.ann_model = None
        self.ivf_index.clear()
        if (self.index_path / 'ivf_ann_model').exists():
            (self.index_path / 'ivf_ann_model').unlink()
        if (self.index_path / 'ivf_index').exists():
            (self.index_path / 'ivf_index').unlink()

    def predict(self, data):
        """
        Predict the cluster label for each sample.

        Parameters:
            data (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        return self.ann_model.predict(data)
