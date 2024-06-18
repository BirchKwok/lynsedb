from pathlib import Path
from typing import Callable, Union

import numpy as np
from spinesUtils.asserts import raise_if
from spinesUtils.logging import Logger

from lynse.configs.config import config
from lynse.storage_layer.storage import PersistentFileStorage
from lynse.core_components.kmeans import BatchKMeans
from lynse.core_components.ivf_index import IVFIndex
from lynse.core_components.locks import ThreadLock


class ClusterWorker:
    def __init__(
            self,
            logger: Logger,
            dataloader: Callable,
            storage_worker: PersistentFileStorage,
            collections_path_parent: Union[str, Path],
    ):
        self.logger = logger
        self.dataloader = dataloader
        self.ann_model = None
        self.storage_worker = storage_worker
        self.ivf_index = IVFIndex()
        self.collections_path_parent = Path(collections_path_parent)
        self.index_mode = 'IVF-FLAT'

        if (self.collections_path_parent / 'ivf_ann_model').exists():
            self.ann_model = BatchKMeans(n_clusters=1, random_state=0, batch_size=10240, epochs=100).load(
                self.collections_path_parent / 'ivf_ann_model')

        if (self.collections_path_parent / 'ivf_index').exists():
            self.ivf_index = IVFIndex().load(self.collections_path_parent / 'ivf_index')

    def _kmeans_clustering(self, data, refit=False):
        """kmeans clustering"""
        if refit:
            self.ann_model = self.ann_model.partial_fit(data)
            labels = self.ann_model.labels_
        else:
            raise_if(ValueError, not self.ann_model.fitted, "The model has not been fitted yet.")
            labels = self.ann_model.predict(data)

        return labels

    def _build_index(self, refit=False):
        """
        Build the IVF index more efficiently.
        """
        if refit:
            self.ivf_index = IVFIndex()

        filenames = self.storage_worker.get_all_files()
        already_indexed = self.ivf_index.all_external_ids()

        for filename in filenames:
            data, indices = self.dataloader(filename)

            isin = np.isin(indices, already_indexed)

            if isin.all():
                continue
            else:
                data = data[~isin]
                indices = indices[~isin]

            labels = self._kmeans_clustering(data, refit=refit)

            # 直接按标签将数据分配到相应的聚类
            for idx, label in zip(indices, labels):
                self.ivf_index.add_entry(label, filename, idx)

    def _check_if_refit(self, n_clusters):
        LYNSE_KMEANS_EPOCHS = config.LYNSE_KMEANS_EPOCHS

        if self.ann_model is None:
            self.ann_model = BatchKMeans(n_clusters=n_clusters, random_state=0,
                                         batch_size=10240, epochs=LYNSE_KMEANS_EPOCHS)

        REFIT = False

        if (self.ann_model.n_clusters != n_clusters or
                self.ann_model.epochs != LYNSE_KMEANS_EPOCHS or
                self.ann_model.batch_size != 10240):
            self.ann_model = BatchKMeans(n_clusters=n_clusters, random_state=0,
                                         batch_size=10240, epochs=LYNSE_KMEANS_EPOCHS)
            REFIT = True

        return REFIT

    def build_index(self, index_mode='IVF-FLAT', n_clusters=32):
        """
        Build the index for clustering.

        Parameters:
            index_mode (str): The index mode, either 'IVF-FLAT' or 'FLAT'.
            n_clusters (int): The number of clusters.

        Returns:
            None
        """
        raise_if(ValueError, not isinstance(n_clusters, int) or n_clusters <= 0,
                 'n_clusters must be int and greater than 0')
        raise_if(ValueError, index_mode not in ['IVF-FLAT', 'FLAT'], 'index_mode must be IVF-FLAT or FLAT')

        if index_mode == 'FLAT':
            self.logger.info("FLAT mode doesn't require indexing.")
            return

        self.index_mode = index_mode

        all_partition_size = self.storage_worker.get_shape()[0]
        if all_partition_size < 100000:
            return

        self.logger.info('Building index...')

        with ThreadLock():
            REFIT = self._check_if_refit(n_clusters)

            if REFIT:
                self.logger.info('Refitting the index for cluster...')
                self._build_index(refit=True)
            else:
                already_clustered_rows = len(self.ivf_index)

                if already_clustered_rows == 0:
                    self.logger.info('Start to fit the index for cluster...')
                    self._build_index(refit=True)
                elif already_clustered_rows == all_partition_size:
                    self.logger.info('All data has been clustered.')
                    return
                else:
                    if all_partition_size - already_clustered_rows >= 100000:
                        self.logger.info('Refitting the index for cluster...')
                        self._build_index(refit=True)
                    else:
                        self.logger.info('Incrementally building the index for cluster...')
                        self._build_index(refit=False)

            # save ivf index and k-means model
            self.logger.debug('Saving ann model...')
            self.ann_model.save(self.collections_path_parent / 'ivf_ann_model')

            self.logger.debug('Saving ivf index...')
            self.ivf_index.save(self.collections_path_parent / 'ivf_index')

    def remove_index(self):
        """
        Remove the index.

        If all indices are removed, the index file will be removed.

        Returns:
            None
        """
        self.ann_model = None
        self.ivf_index.clear_all()
        if (self.collections_path_parent / 'ivf_ann_model').exists():
            (self.collections_path_parent / 'ivf_ann_model').unlink()
        if (self.collections_path_parent / 'ivf_index').exists():
            (self.collections_path_parent / 'ivf_index').unlink()
        self.logger.info('Index removed.')
