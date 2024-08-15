from pathlib import Path
from typing import Callable, Union

import numpy as np
from spinesUtils.asserts import raise_if

from ..configs.config import config
from ..storage_layer.storage import PersistentFileStorage
from ..core_components.kmeans import BatchKMeans
from ..core_components.ivf_index import IVFIndex
from ..core_components.locks import ThreadLock


class IVFCreator:
    def __init__(
            self,
            dataloader: Callable,
            storage_worker: PersistentFileStorage,
            collections_path_parent: Union[str, Path],
    ):
        self.dataloader = dataloader
        self.ann_model = None
        self.storage_worker = storage_worker
        self.ivf_index = IVFIndex()

        self.index_path = Path(collections_path_parent) / 'index'
        self.index_path.mkdir(parents=True, exist_ok=True)

        if (self.index_path / 'ivf_ann_model').exists():
            self.ann_model = BatchKMeans(n_clusters=1, batch_size=10240, epochs=100).load(
                self.index_path / 'ivf_ann_model')

        if (self.index_path / 'ivf_index').exists():
            self.ivf_index = IVFIndex().load(self.index_path / 'ivf_index')

    def _kmeans_clustering(self, data, rebuild=False):
        """kmeans clustering"""
        if rebuild:
            self.ann_model = self.ann_model.partial_fit(data)
            labels = self.ann_model.labels_
        else:
            raise_if(ValueError, not self.ann_model.fitted, "The model has not been fitted yet.")
            labels = self.ann_model.predict(data)

        return labels

    def _build_index(self, rebuild=False):
        """
        Build the IVF index more efficiently.
        """
        if rebuild:
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

            labels = self._kmeans_clustering(data, rebuild=rebuild)

            for idx, label in zip(indices, labels):
                self.ivf_index.add_entry(label, filename, idx)

    def _check_if_refit(self, n_clusters, always_refit=False):
        if self.ann_model is None or always_refit:
            self.ann_model = BatchKMeans(n_clusters=n_clusters, batch_size=10240, epochs=config.LYNSE_KMEANS_EPOCHS)

        REFIT = False

        if (self.ann_model.n_clusters != n_clusters or
                self.ann_model.epochs != config.LYNSE_KMEANS_EPOCHS or
                self.ann_model.batch_size != 10240):
            self.ann_model = BatchKMeans(n_clusters=n_clusters, batch_size=10240, epochs=config.LYNSE_KMEANS_EPOCHS)
            REFIT = True

        return REFIT

    def build_index(self, n_clusters=32, rebuild=False):
        """
        Build the index for clustering.

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

        with ThreadLock():
            REBUILD = self._check_if_refit(n_clusters, always_refit=rebuild) or rebuild

            if REBUILD:
                # delete the old index
                self.remove_index()
                self.ann_model = BatchKMeans(n_clusters=n_clusters, batch_size=10240, epochs=config.LYNSE_KMEANS_EPOCHS)
                self._build_index(rebuild=True)
            else:
                already_clustered_rows = len(self.ivf_index)

                if already_clustered_rows == 0:
                    self._build_index(rebuild=True)
                elif already_clustered_rows == all_partition_size:
                    return
                else:
                    if all_partition_size - already_clustered_rows >= 100000:
                        self._build_index(rebuild=True)
                    else:
                        self._build_index(rebuild=False)

            # save ivf index and k-means model
            self.ann_model.save(self.index_path / 'ivf_ann_model')

            self.ivf_index.save(self.index_path / 'ivf_index')

    def fast_insert(self, start_filename):
        """
        Insert the data into the index.

        Parameters:
            start_filename (str): The filename to start with.

        Returns:
            None
        """
        filenames = self.storage_worker.get_all_files()
        filenames = [filename for filename in filenames if int(filename.split('_')[-1]) >=
                     int(start_filename.split('_')[-1])]

        already_indexed = self.ivf_index.all_external_ids()

        only_run_once = False
        for filename in filenames:
            data, indices = self.dataloader(filename)

            if not only_run_once:
                isin = np.isin(indices, already_indexed)

                if isin.all():
                    continue
                else:
                    data = data[~isin]
                    indices = indices[~isin]

                only_run_once = True

            labels = self._kmeans_clustering(data, rebuild=True)

            for idx, label in zip(indices, labels):
                self.ivf_index.add_entry(label, filename, idx)

    def remove_index(self):
        """
        Remove the index.

        If all indices are removed, the index file will be removed.

        Returns:
            None
        """
        self.ann_model = None
        self.ivf_index.clear_all()
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
