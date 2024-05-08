from datetime import datetime
from pathlib import Path
import shutil
from typing import Union

import numpy as np
from spinesUtils.asserts import raise_if
from spinesUtils.logging import Logger

from min_vec.core_components.cross_lock import ThreadLock
from min_vec.core_components.id_checker import IDChecker
from min_vec.utils.utils import io_checker
from min_vec.configs.config import config
from min_vec.core_components.kmeans import BatchKMeans
from min_vec.core_components.scaler import ScalarQuantization
from min_vec.core_components.metadata_kv import MetaDataKVCache
from min_vec.storage_layer.storage import PersistentFileStorage, TemporaryFileStorage
from min_vec.execution_layer.cluster_worker import ClusterWorker
from min_vec.core_components.counter import SafeCounter
from min_vec.core_components.thread_safe_list import SafeList


class MatrixSerializer:
    """
    The MatrixSerializer class is used to serialize and deserialize the matrix data.
    """

    def __init__(
            self,
            dim: int,
            collection_path: Union[str, Path],
            logger: Logger,
            n_clusters: int = 16,
            chunk_size: int = 1_000_000,
            index_mode: str = 'IVF-FLAT',
            dtypes: str = 'float32',
            scaler_bits=None,
            warm_up: bool = False
    ) -> None:
        """
        Initialize the vector collection.

        Parameters:
            dim (int): Dimension of the vectors.
            collection_path (str): Path to the collections file.
            logger (Logger): The logger object.
            n_clusters (int): The number of clusters for the IVF-FLAT index. Default is 16.
            chunk_size (int): The size of each data chunk. Default is 1_000_000.
            index_mode (str): The index mode of the collection.
                Options are 'FLAT' or 'IVF-FLAT'. Default is 'IVF-FLAT'.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            scaler_bits (int): The number of bits for scalar quantization. Default is None.
                Options are 8, 16, 32. If None, scalar quantization will not be used.
                The 8 bits for uint8, 16 bits for uint16, 32 bits for uint32.
            warm_up (bool): Whether to warm up the collection. Default is False.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        self.last_commit_time = None
        # set commit flag, if the flag is True, the collection will not be saved
        self.COMMIT_FLAG = True
        self.IS_DELETED = False

        self.logger = logger

        # set parent path
        self._initialize_parent_path(collection_path)

        self._dtypes_map = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}

        # set dtypes
        self.dtypes = self._dtypes_map[dtypes]
        # set index mode
        self.index_mode = index_mode
        # set n_clusters
        self.n_clusters = n_clusters
        # set dim
        self.dim = dim
        # set chunk size
        self.chunk_size = chunk_size
        # set filter path
        self._filter_path = self.collections_path_parent / 'id_filter.mvdb'

        # set collection path
        self.collections_path = self.collections_path_parent

        # set scalar quantization bits
        self.scaler_bits = scaler_bits if scaler_bits is not None else None
        self.scaler = None
        if self.scaler_bits is not None:
            self._initialize_scalar_quantization()

        # initialize the storage worker
        self.storage_worker = PersistentFileStorage(self.collections_path_parent, self.dim,
                                                    self.chunk_size,
                                                    warm_up=warm_up)
        self.tempfile_storage_worker = TemporaryFileStorage(self.chunk_size)

        self.counter = SafeCounter()

        # ============== Loading or create one empty collection ==============
        # first of all, initialize a collection
        self.database = SafeList()
        self.indices = SafeList()
        self.fields = SafeList()

        self._initialize_fields_index()
        self._initialize_ann_model()
        self._initialize_id_checker()

        self.cluster_worker = ClusterWorker(
            logger=self.logger,
            iterable_dataloader=self.dataloader,
            ann_model=self.ann_model,
            storage_worker=self.storage_worker,
            n_clusters=self.n_clusters
        )

        if self._get_cluster_dataset_num() > 0 and self.index_mode == 'FLAT':
            # cause the index mode is FLAT, but the cluster dataset is not empty,
            # so the clustered datasets will also be traversed during querying.
            self.logger.warning('The index mode is FLAT, but the cluster dataset is not empty, '
                                'so the clustered datasets will also be traversed during querying.')

        self.lock = ThreadLock()

    def _initialize_parent_path(self, collections_path):
        """make directory if not exist"""
        self.collections_path_parent = (Path(collections_path).parent.absolute() /
                                        Path(collections_path).absolute().name)

        self.collections_path_parent.mkdir(parents=True, exist_ok=True)

    def _initialize_scalar_quantization(self):
        if Path(self.collections_path_parent / 'sq_model.mvdb').exists():
            self.scaler = ScalarQuantization.load(self.collections_path_parent / 'sq_model.mvdb')
        else:
            self.scaler = ScalarQuantization(bits=self.scaler_bits, decode_dtype=self.dtypes)

    def _initialize_ann_model(self):
        """initialize ann model"""
        if self.index_mode == 'IVF-FLAT':
            MVDB_KMEANS_EPOCHS = config.MVDB_KMEANS_EPOCHS
            self.ann_model = BatchKMeans(n_clusters=self.n_clusters, random_state=0,
                                         batch_size=10240, epochs=MVDB_KMEANS_EPOCHS)

            if Path(self.collections_path_parent / 'ann_model.mvdb').exists() and self.index_mode == 'IVF-FLAT':
                self.ann_model = self.ann_model.load(self.collections_path_parent / 'ann_model.mvdb')
                if (self.ann_model.n_clusters != self.n_clusters or
                        self.ann_model.epochs != MVDB_KMEANS_EPOCHS or
                        self.ann_model.batch_size != 10240):
                    self.ann_model = BatchKMeans(n_clusters=self.n_clusters, random_state=0,
                                                 batch_size=10240, epochs=MVDB_KMEANS_EPOCHS)
        else:
            self.ann_model = None

    def _initialize_fields_index(self):
        """initialize fields index"""
        if Path(self.collections_path_parent / 'fields_index.mvdb').exists():
            self.kv_index = MetaDataKVCache().load(self.collections_path_parent / 'fields_index.mvdb')
        else:
            self.kv_index = MetaDataKVCache()

        self.temp_kv_index = MetaDataKVCache()

    def _initialize_id_checker(self):
        """initialize id checker and shape"""
        self.id_filter = IDChecker()

        if self._filter_path.exists():
            self.id_filter.from_file(self._filter_path)
        else:
            if self.collections_path.exists():
                filenames = self.storage_worker.get_all_files(read_type='all')
                for filename in filenames:
                    database, indices = self.storage_worker.read(filename=filename)
                    self.id_filter.add(indices)

        self.last_id = self.id_filter.find_max_value()

        self.temp_id_filter = IDChecker()

    def reset_collection(self):
        """Reset the database to its initial state with zeros."""
        self.database = SafeList()
        self.indices = SafeList()
        self.fields = SafeList()

    @io_checker
    def dataloader(self, filename):
        """
        Generator for loading the database and index.

        Parameters:
            filename (str): The name of the file to load.

        Yields:
            tuple: A tuple of (database, index, field).

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
            PermissionError: If the file cannot be read due to permission issues.
            UnKnownError: If an unknown error occurs.
        """
        data, indices = self.storage_worker.read(filename=filename)

        return data, indices

    def _is_collection_reset(self):
        """
        Check if the collection is in its reset state (empty list).

        Returns:
            bool: True if the collection is reset, False otherwise.
        """
        return not (self.database and self.indices and self.fields)

    def _length_checker(self):
        """
        Raises:
            ValueError: If the lengths of the database, indices, and fields are not the same.
        """
        if not self._is_collection_reset() and not self.COMMIT_FLAG:
            if not (len(self.database) == len(self.indices) == len(self.fields)):
                raise ValueError('The database, index length and field length in collection are not the same.')

    def save_data(self, data, indices, fields):
        """Optimized method to save data to chunk or cluster group with reusable logic."""

        for _id, _field in zip(indices, fields):
            self.temp_kv_index.store(_field if _field is not None else {}, _id)

        self.tempfile_storage_worker.write_temp_data(data, indices)

    @io_checker
    def save_chunk_immediately(self):
        """
        Save the current state of the collection to a .mvdb file.

        Returns:
            Path: The path of the saved collection file.
        """
        self._length_checker()

        if self._is_collection_reset():
            return

        self.save_data(
            self.database,
            self.indices,
            self.fields
        )

        self.reset_collection()  # reset collection, indices and fields

    def auto_save_chunk(self):
        self._length_checker()
        if len(self.database) >= self.chunk_size:
            self.save_chunk_immediately()

        return

    def _get_cluster_dataset_num(self):
        if not self.collections_path.exists():
            return 0

        return self.storage_worker.get_cluster_dataset_num()

    def rollback(self):
        """
        Rollback the collection to the last commit.
        """
        if not self.COMMIT_FLAG:
            self.reset_collection()
            self.tempfile_storage_worker.reincarnate()
            self.temp_id_filter = IDChecker()
            self.temp_kv_index = MetaDataKVCache()

            self.COMMIT_FLAG = True

    def commit(self):
        """
        Save the collection, ensuring that all data is written to disk.
        This method is required to be called after saving vectors to query them.
        """
        with self.lock:
            if not self.COMMIT_FLAG:
                try:
                    self.logger.debug('Saving chunk immediately...')
                    self.save_chunk_immediately()

                    try:
                        self.logger.debug('Concatenating id filter...')
                        # concat filter
                        self.id_filter.concat(self.temp_id_filter)
                    except Exception as e:
                        self.logger.error(f'Error occurred while concatenating the filter: {e}, rollback...')
                        self.rollback()
                        raise e

                    self.logger.debug('Writing chunk to storage...')
                    for data, indices in self.tempfile_storage_worker.get_file_iterator():
                        self.storage_worker.write(data, indices, write_type='chunk', normalize=True)

                    self.tempfile_storage_worker.reincarnate()

                except Exception as e:
                    self.logger.error(f'Error occurred while saving the collection: {e}, rollback...')
                    self.rollback()
                    raise e

                try:
                    self.logger.debug('Concatenating fields index...')
                    self.kv_index.concat(self.temp_kv_index)
                except Exception as e:
                    self.logger.error(f'Error occurred while concatenating the fields index: {e}, rollback...')
                    self.rollback()
                    raise e

                try:
                    self.logger.debug('Saving id filter...')
                    self.id_filter.to_file(self._filter_path)
                except Exception as e:
                    self.logger.error(f'Error occurred while saving the filter: {e}, rollback...')
                    self.rollback()
                    raise e

                try:
                    # save fields index
                    self.logger.debug('Saving fields index...')
                    self.kv_index.save(self.collections_path_parent / 'fields_index.mvdb')
                except Exception as e:
                    self.logger.error(f'Error occurred while saving the collection: {e}, rollback...')
                    self.rollback()
                    raise e

                chunk_partition_size = self.storage_worker.get_shape(read_type='chunk')[0]
                cluster_partition_size = self.storage_worker.get_shape(read_type='cluster')[0]

                all_partition_size = self.storage_worker.get_shape(read_type='all')[0]

                if all_partition_size >= 100000 and not (self.collections_path_parent / 'scaled_status.json').exists():
                    # only run once
                    self.storage_worker.update_quantizer(self.scaler)
                    filenames = self.storage_worker.get_all_files(read_type='chunk')
                    if filenames:
                        for filename in filenames:
                            self.storage_worker.write(filename=filename)

                    with open(self.collections_path_parent / 'scaled_status.json', 'w') as f:
                        import json
                        json.dump({'status': True}, f)

                if (
                        cluster_partition_size == 0 and (chunk_partition_size >= 100000 and self.index_mode != 'FLAT')
                ) or (
                        cluster_partition_size > 0 and self.index_mode != 'FLAT'
                ):
                    self.logger.info('Building index...')

                    if cluster_partition_size == 0 and (chunk_partition_size >= 100000 and self.index_mode != 'FLAT'):
                        self.logger.info('Start to fit the index for cluster...')
                        self.cluster_worker.build_index(refit=True)
                    else:
                        self.counter.add(chunk_partition_size)
                        if self.counter.get_value() >= 100000:
                            self.logger.info('Refitting the index for cluster...')
                            self.cluster_worker.build_index(refit=True)
                            self.counter.reset()
                        else:
                            self.logger.info('Incrementally building the index for cluster...')
                            self.cluster_worker.build_index(refit=False)

                    # save ivf index and k-means model
                    self.logger.debug('Saving ann model...')
                    self.ann_model.save(self.collections_path_parent / 'ann_model.mvdb')

                self.reset_collection()

                if self.scaler_bits is not None:
                    if self.scaler.fitted:
                        self.scaler.save(self.collections_path_parent / 'sq_model.mvdb')

                self.COMMIT_FLAG = True

                self.last_commit_time = datetime.now()

    def _process_vector_item(self, vector, index, field):
        if index in self.id_filter or index in self.temp_id_filter:
            raise ValueError(f'id {index} already exists.')

        if len(vector) != self.dim:
            raise ValueError(f'vector dim error, expect {self.dim}, got {len(vector)}')

        if vector.dtype != self.dtypes:
            vector = vector.astype(self.dtypes)

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        return vector, index, field if field is not None else {}

    def bulk_add_items(self, vectors):
        """
        Bulk add vectors to the collection in batches.

        Parameters: vectors (list or tuple): A list or tuple of vectors to be saved. Each vector can be a tuple of (
            vector, id, field).

        Returns:
            list: A list of indices where the vectors are stored.
        """
        raise_if(ValueError, not isinstance(vectors, (tuple, list)),
                 f'vectors must be tuple or list, got {type(vectors)}')

        with self.lock:
            new_ids = []

            for i in range(0, len(vectors), self.chunk_size):
                batch = vectors[i:i + self.chunk_size]

                for sample in batch:
                    sample_len = len(sample)

                    if sample_len == 3:
                        vector, index, field = sample
                    elif sample_len == 2:
                        vector, index = sample
                        field = {}
                    else:
                        raise ValueError('Each sample must be a tuple of (vector, id, field[optional]).')

                    raise_if(TypeError, not (isinstance(field, dict) or field is None),
                             f'field must be dict or None, got {type(field)}')

                    if isinstance(vector, list):
                        vector = np.array(vector)

                    raise_if(ValueError, (not isinstance(index, int)) or index < 0,
                             f'id must be integer and greater than 0, got {index}')

                    field = {} if field is None else field
                    vector, index, field = self._process_vector_item(vector, index, field)

                    self.database.append(vector)
                    self.indices.append(index)
                    new_ids.append(index)
                    self.fields.append(field)
                    self.temp_id_filter.add(index)

                self.auto_save_chunk()

            if self.COMMIT_FLAG:
                self.COMMIT_FLAG = False

            return new_ids

    def add_item(self, vector, index: int, *, field: dict = None) -> int:
        """
        Add a single vector to the collection.

        Parameters:
            vector (np.ndarray): The vector to be added.
            index (int): The ID of the vector.
            field (dict, optional, keyword-only): The field of the vector. Default is None. If None, the field will be
                set to an empty string.
        Returns:
            int: The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        if isinstance(vector, list):
            vector = np.array(vector)

        raise_if(ValueError, vector.ndim != 1, f'vector dim error, expect 1, got {vector.ndim}')
        raise_if(ValueError, field is not None and not isinstance(field, dict),
                 f'field must be dict, got {type(field)}')

        raise_if(ValueError, (not isinstance(index, int)) or index < 0,
                 f'id must be integer and greater than 0, got {index}')

        with self.lock:
            vector, index, field = self._process_vector_item(vector, index, field)

            # Add the id to then filter.
            self.temp_id_filter.add(index)

            self.database.append(vector)
            self.indices.append(index)
            self.fields.append(field)

            self.auto_save_chunk()

            if self.COMMIT_FLAG:
                self.COMMIT_FLAG = False

            return index

    def delete(self):
        """Delete collection."""
        with self.lock:
            if not self.collections_path_parent.exists():
                return None

            try:
                shutil.rmtree(self.collections_path_parent)
            except FileNotFoundError:
                pass

            self.IS_DELETED = True
            self.reset_collection()

            # reinitialize
            if self.scaler_bits is not None:
                self._initialize_scalar_quantization()

            self._initialize_fields_index()
            self._initialize_ann_model()
            self._initialize_id_checker()

            # clear cache
            self.storage_worker.clear_cache()

    @property
    def shape(self):
        with self.lock:
            return tuple(self.storage_worker.get_shape(read_type='all'))
