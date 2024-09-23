from pathlib import Path
from typing import Callable, Union

import numpy as np
from spinesUtils.asserts import raise_if
from spinesUtils.logging import Logger

from ..core_components.locks import ThreadLock
from ..storage_layer.storage import PersistentFileStorage
from ..index.sq import IndexSQIP, IndexSQL2sq, IndexSQCos
from ..index.binary import IndexBinaryJaccard, IndexBinaryHamming
from ..index.flat import IndexFlatIP, IndexFlatL2sq, IndexFlatCos
from .ivf import IVFCreator
from ..utils.utils import drop_duplicated_substr, find_first_file_with_substr, safe_mmap_reader

_INDEX_ALIAS = {
    'IVF-IP-SQ8': 'IVF-IP-SQ8',
    'IVF': 'IVF-IP',
    'IVF-L2sq-SQ8': 'IVF-L2sq-SQ8',
    'IVF-L2sq': 'IVF-L2sq',
    'IVF-Cos-SQ8': 'IVF-Cos-SQ8',
    'IVF-Cos': 'IVF-Cos',
    'IVF-Jaccard-Binary': 'IVF-Jaccard-Binary',
    'IVF-Hamming-Binary': 'IVF-Hamming-Binary',
    'Flat-IP-SQ8': 'Flat-IP-SQ8',
    'FLAT': 'Flat-IP',
    'Flat-L2sq-SQ8': 'Flat-L2sq-SQ8',
    'Flat-L2sq': 'Flat-L2sq',
    'Flat-Cos-SQ8': 'Flat-Cos-SQ8',
    'Flat-Cos': 'Flat-Cos',
    'Flat-Hamming-Binary': 'Flat-Hamming-Binary',
    'Flat-Jaccard-Binary': 'Flat-Jaccard-Binary'
}

_IndexMapper = {
    'IVF-IP-SQ8': (IndexSQIP, IVFCreator),
    'IVF-IP': (IndexFlatIP, IVFCreator),
    'IVF-L2sq-SQ8': (IndexSQL2sq, IVFCreator),
    'IVF-L2sq': (IndexFlatL2sq, IVFCreator),
    'IVF-Cos-SQ8': (IndexSQCos, IVFCreator),
    'IVF-Cos': (IndexFlatCos, IVFCreator),
    'IVF-Jaccard-Binary': (IndexBinaryJaccard, IVFCreator),
    'IVF-Hamming-Binary': (IndexBinaryHamming, IVFCreator),
    'Flat-IP-SQ8': (IndexSQIP, None),
    'Flat-IP': (IndexFlatIP, None),
    'Flat-L2sq-SQ8': (IndexSQL2sq, None),
    'Flat-L2sq': (IndexFlatL2sq, None),
    'Flat-Cos-SQ8': (IndexSQCos, None),
    'Flat-Cos': (IndexFlatCos, None),
    'Flat-Jaccard-Binary': (IndexBinaryJaccard, None),
    'Flat-Hamming-Binary': (IndexBinaryHamming, None)
}


class Indexer:
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
        self.collections_path_parent = Path(collections_path_parent)
        self.index_data_path = self.collections_path_parent / 'index_data'
        self.index_ids_path = self.collections_path_parent / 'index_ids'
        self.index_path = self.collections_path_parent / 'index'

        for _ in [self.index_data_path, self.index_ids_path, self.index_path]:
            _.mkdir(parents=True, exist_ok=True)

        self.lock = ThreadLock()

        self.index = None
        self.ivf = None
        self.index_mode = None
        self.current_index_mode = None

    def _build_flat_index(self, index_mode: str):
        """
        Build the flat index.

        Parameters:
            index_mode (str): The index mode.
        """
        def _rebuild():
            nonlocal _index

            filenames = self.storage_worker.get_all_files()
            for filename in filenames:
                data, ids = self.dataloader(filename)
                _index.fit_transform(data, ids)

            _index.save(self.index_path / f'{index_mode}.index')
            if hasattr(_index, "save_data"):
                if index_mode.endswith('SQ8'):
                    data_path = self.index_data_path / f'{self.storage_worker.fingerprint}.sqd'
                    ids_path = self.index_ids_path / f'{self.storage_worker.fingerprint}.sqi'
                elif 'Binary' in index_mode:
                    data_path = self.index_data_path / f'{self.storage_worker.fingerprint}.bd'
                    ids_path = self.index_ids_path / f'{self.storage_worker.fingerprint}.bi'
                else:
                    raise NotImplementedError

                _index.save_data(data_path, ids_path)
            return _index

        def build_sq8_index_for_binary():
            index = _IndexMapper['Flat-IP-SQ8'][0]()
            fitted = False
            for index_name in self.index_path.iterdir():
                if 'SQ8' in index_name.name:
                    index = index.load(index_name)
                    fitted = True
                    break

            if not fitted:
                filenames = self.storage_worker.get_all_files()
                _data = []
                _ids = []
                for filename in filenames:
                    data, ids = self.dataloader(filename)
                    data = np.vstack([np.vstack(_data), data]) if _data else data
                    ids = np.hstack([np.hstack(_ids), ids]) if _ids else ids
                    if data.shape[0] < 100000:
                        _data.append(data)
                        _ids.append(ids)
                        continue

                    _data, _ids = [], []
                    index.fit_transform(data, ids)
                index.save(self.index_path / 'Flat-IP-SQ8.index')
                index.save_data(self.index_data_path / f'{self.storage_worker.fingerprint}.sqd',
                                self.index_ids_path / f'{self.storage_worker.fingerprint}.sqi')
                # clear the data and ids, because they are saved in the file
                index.data = None
                index.ids = None

            return index

        def build_binary():
            nonlocal _index
            if not (self.index_data_path / f'{self.storage_worker.fingerprint}.bd').exists():
                _index = _rebuild()
            else:
                binary_data = safe_mmap_reader(self.index_data_path / f'{self.storage_worker.fingerprint}.bd')
                binary_ids = safe_mmap_reader(self.index_ids_path / f'{self.storage_worker.fingerprint}.bi')

                # load sq8 data as a view, used for rescore
                if (not (self.index_data_path / f'{self.storage_worker.fingerprint}.sqd').exists()) or (
                        find_first_file_with_substr(
                            self.index_ids_path,
                            f'{self.storage_worker.fingerprint}.*SQ8.index') is None
                ):
                    sq8_index = build_sq8_index_for_binary()
                else:
                    sq8_index = _IndexMapper['Flat-IP-SQ8'][0]()

                    sq8_index = sq8_index.load(
                        find_first_file_with_substr(self.index_ids_path,
                                                    f'{self.storage_worker.fingerprint}.*SQ8.index')
                    )

                sq8_data = safe_mmap_reader(self.index_data_path / f'{self.storage_worker.fingerprint}.sqd')

                _index.data = binary_data
                _index.ids = binary_ids
                _index.sq8_data = sq8_data
                _index.sq8_encode = sq8_index.encode
                _index.sq8_decode = sq8_index.decode

            return _index

        # ========== main logic ==========
        uninitialized_index, _ = _IndexMapper[index_mode]
        _index = uninitialized_index()

        if (self.index_path / f'{index_mode}.index').exists():
            _index = _index.load(self.index_path / f'{index_mode}.index')
        else:
            _index = _rebuild()

        if 'Binary' in index_mode:
            _index = build_binary()
        elif 'SQ8' in index_mode:
            if (not (self.index_data_path / f'{self.storage_worker.fingerprint}.sqd').exists()) or (
                not (self.index_ids_path / f'{self.storage_worker.fingerprint}.sqi').exists()
            ):
                _index = _rebuild()
            else:
                sq8_data = safe_mmap_reader(self.index_data_path / f'{self.storage_worker.fingerprint}.sqd')
                sq8_ids = safe_mmap_reader(self.index_ids_path / f'{self.storage_worker.fingerprint}.sqi')

                _index.data = sq8_data
                _index.ids = sq8_ids
        else:
            ...  # do nothing

        self.index = _index
        self.index_mode = index_mode

    def _build_ivf_index(self, n_clusters: int, rebuild=False):
        """
        Build the IVF index.

        Parameters:
            n_clusters (int): The number of clusters.
            rebuild (bool): Whether to rebuild the index.
        """
        self.ivf = IVFCreator(
            dataloader=self.dataloader,
            storage_worker=self.storage_worker,
            collections_path_parent=self.collections_path_parent
        )

        self.ivf.build_index(n_clusters=n_clusters, rebuild=rebuild)

    def build_index(self, index_mode='IVF-IP-SQ8', rebuild=False, **kwargs):
        """
        Build the index for clustering.

        Parameters:
            index_mode (str): The index mode, must be one of the following:

                - 'IVF-IP-SQ8': IVF index with inner product and scalar quantizer with 8 bits.
                    The distance is inner product.
                - 'IVF-IP': IVF index with inner product. (Alias: 'IVF')
                - 'IVF-L2sq-SQ8': IVF index with squared L2 distance and scalar quantizer with 8 bits.
                    The distance is squared L2 distance.
                - 'IVF-L2sq': IVF index with squared L2 distance.
                - 'IVF-Cos-SQ8': IVF index with cosine similarity and scalar quantizer with 8 bits.
                    The distance is cosine similarity.
                - 'IVF-Cos': IVF index with cosine similarity.
                - 'IVF-Jaccard-Binary': IVF index with binary quantizer. The distance is Jaccard distance.
                - 'IVF-Hamming-Binary': IVF index with binary quantizer. The distance is Hamming distance.
                - 'Flat-IP-SQ8': Flat index with inner product and scalar quantizer with 8 bits.
                - 'Flat-IP': Flat index with inner product. (Alias: 'FLAT')
                - 'Flat-L2sq-SQ8': Flat index with squared L2 distance and scalar quantizer with 8 bits.
                - 'Flat-L2sq': Flat index with squared L2 distance.
                - 'Flat-Cos-SQ8': Flat index with cosine similarity and scalar quantizer with 8 bits.
                - 'Flat-Cos': Flat index with cosine similarity.
                - 'Flat-Jaccard-Binary': Flat index with binary quantizer. The distance is Jaccard distance.
                - 'Flat-Hamming-Binary': Flat index with binary quantizer. The distance is Hamming distance.
            rebuild (bool): Whether to rebuild the index.
            kwargs: Additional keyword arguments. The following are available:

                    - 'n_clusters': The number of clusters. It is only available when the index_mode including 'IVF'.

        Returns:
            None
        """
        # clear the old data first
        self._remove_old_data()

        n_clusters = kwargs.setdefault('n_clusters', 32)
        raise_if(ValueError, not isinstance(n_clusters, int) or n_clusters <= 0,
                 'n_clusters must be int and greater than 0')

        raise_if(ValueError, index_mode not in _INDEX_ALIAS and index_mode not in _INDEX_ALIAS.values(),
                 'index_mode must be one of the following:'
                 f' {list(_INDEX_ALIAS) + list(_INDEX_ALIAS.values())}')

        if (not rebuild) and self.index_mode == index_mode:
            self.logger.info('Index already exists.')
            return

        if index_mode in _INDEX_ALIAS:
            index_mode = _INDEX_ALIAS[index_mode]

        self.logger.info(f'Building an index using the `{index_mode}` index mode...')
        all_partition_size = self.storage_worker.get_shape()[0]

        if 'Binary' in index_mode or 'SQ8' in index_mode:
            if all_partition_size < 100000:
                substr = '-Binary' if 'Binary' in index_mode else '-SQ8'
                index_mode = drop_duplicated_substr(index_mode, substr).replace("IVF", "Flat")
                self.logger.info('Index is not built because the number of data points is less than 100000.'
                                 f'Continue to build the {index_mode} index.')

        with self.lock:
            self._build_flat_index(index_mode)
        if not index_mode.startswith('Flat'):
            with self.lock:
                self._build_ivf_index(n_clusters, rebuild=rebuild)

        self.current_index_mode = index_mode
        self.logger.info('Index built.')

    def index_insert(self, data, ids):
        """
        Insert the data into the index.

        Parameters:
            data (np.ndarray): The data to insert.
            ids (np.ndarray): The IDs of the data.

        Returns:
            None
        """
        raise_if(ValueError, self.index is None, 'The index must be built before inserting data.')

        with self.lock:
            self.index.fit_transform(data, ids)

    def update_filenames(self):
        # get new fingerprint
        with open(self.storage_worker.fingerprint_path, 'r') as file:
            new_fingerprint = file.readlines()[-1].strip()

        self._remove_old_data()
        if hasattr(self.index, 'save_data'):
            self.index.save_data(self.index_data_path / f'{new_fingerprint}.sqd',
                                 self.index_ids_path / f'{new_fingerprint}.sqi')

    def ivf_insert(self, last_filename):
        self.ivf.fast_insert(last_filename)

    def _remove_old_data(self):
        """
        Remove old data, including the index and the data.
        """
        # read all history fingerprints
        if not self.storage_worker.fingerprint_path.exists():
            return
        with open(self.storage_worker.fingerprint_path, 'r') as file:
            old_fingerprints = file.readlines()[:-1]

        if old_fingerprints:
            for fingerprint in old_fingerprints:
                fingerprint = fingerprint.strip()
                for _ in [self.index_data_path, self.index_ids_path, self.index_path]:
                    for file in _.iterdir():
                        if fingerprint in file.name:
                            file.unlink()

    def remove_index(self):
        """
        Remove the index.

        Returns:
            None
        """
        self.index = None
        self.index_mode = None
        # remove local index
        if self.current_index_mode is not None:
            if self.current_index_mode.startswith('IVF'):
                self.ivf.remove_index()
                self.ivf = None
            else:
                index_mode = self.current_index_mode
                if index_mode in _INDEX_ALIAS:
                    index_mode = _INDEX_ALIAS[index_mode]
                if (self.index_path / f'{index_mode}_index').exists():
                    (self.index_path / f'{index_mode}_index').unlink()

        self.current_index_mode = 'Flat-IP'  # Default index mode

        self.logger.info('Index removed.')
