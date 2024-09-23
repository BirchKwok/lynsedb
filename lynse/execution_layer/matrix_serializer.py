from datetime import datetime
from pathlib import Path
import shutil
import time
from typing import Union

import numpy as np
from spinesUtils.asserts import raise_if
from spinesUtils.logging import Logger
from tqdm import tqdm

from ..core_components.locks import ThreadLock
from ..core_components.id_checker import IDChecker
from ..core_components.fields_cache import FieldsCache
from ..storage_layer.storage import PersistentFileStorage
from ..storage_layer.wal import WALStorage


class MatrixSerializer:
    """
    The MatrixSerializer class is used to serialize and deserialize the matrix data.
    """

    def __init__(
            self,
            dim: int,
            collection_path: Union[str, Path],
            logger: Logger,
            chunk_size: int = 100_000,
            dtypes: str = 'float32',
            warm_up: bool = False,
            cache_chunks: int = 20,
    ) -> None:
        """
        Initialize the vector collection.

        Parameters:
            dim (int): Dimension of the vectors.
            collection_path (str): Path to the collections file.
            logger (Logger): The logger object.
            chunk_size (int): The size of each data chunk. Default is 1_000_000.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            warm_up (bool): Whether to warm up the collection. Default is False.
            cache_chunks (int): The buffer size for the storage worker. Default is 20.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        self.last_commit_time = None
        # set commit flag, if the flag is True, the collection will not be saved
        self.COMMIT_FLAG = True
        self.IS_DELETED = False

        self.logger = logger

        self.threadlock = ThreadLock()

        # set parent path
        self._initialize_components_path(collection_path)

        self._dtypes_map = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}

        # set dtypes
        self.dtypes = self._dtypes_map[dtypes]
        # set dim
        self.dim = dim
        # set chunk size
        self.chunk_size = chunk_size

        # initialize the storage worker
        self.storage_worker = PersistentFileStorage(self.collections_path_parent, self.dim,
                                                    self.chunk_size,
                                                    warm_up=warm_up, cache_chunks=cache_chunks)
        self.wal_worker = WALStorage(collection_name=self.collections_path_parent.name,
                                     chunk_size=self.chunk_size, storage_path=self.collections_path_parent,
                                     flush_interval=1)

        self._initialize_fields_index()
        self._initialize_id_checker()

        # log_dir exists and not empty
        if self.wal_worker.log_dir.exists() and any(self.wal_worker.log_dir.iterdir()):
            self.logger.info("Detected uncommitted data, preparing to recover...")
            # replay wal
            self.commit_data(recover=True)

    def _initialize_components_path(self, collections_path):
        """make directory if not exist"""
        self.collections_path_parent = (Path(collections_path).parent.absolute() /
                                        Path(collections_path).absolute().name)

        self.collections_path_parent.mkdir(parents=True, exist_ok=True)

        # sq_model path
        self.sq_model_path = self.collections_path_parent / 'sq_model'
        # field_index path
        self.field_index_path = self.collections_path_parent / 'fields_index'
        # scaled_status path
        self.scaled_status_path = self.collections_path_parent / 'scaled_status.json'
        # set filter path
        self.filter_path = self.collections_path_parent / 'id_filter'

    def _initialize_fields_index(self):
        """initialize fields index"""
        self.field_index = FieldsCache(self.field_index_path)

    def _initialize_id_checker(self):
        """initialize id checker and shape"""
        self.id_filter = IDChecker()

        if self.filter_path.exists():
            self.id_filter.from_file(self.filter_path)
        else:
            if self.collections_path_parent.exists():
                filenames = self.storage_worker.get_all_files()
                for filename in filenames:
                    database, indices = self.storage_worker.read(filename=filename)
                    self.id_filter.add(indices)

        self.last_id = self.id_filter.find_max_value()

    def dataloader(self, filename):
        """
        Generator for loading the database and index.

        Parameters:
            filename (str): The name of the file to load.

        Yields:
            tuple: A tuple of (database, id).

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
            PermissionError: If the file cannot be read due to permission issues.
            UnKnownError: If an unknown error occurs.
        """
        return self.storage_worker.read(filename=filename)

    def commit_data(self, recover=False):
        if not recover:
            start_msg = 'Writing chunk to storage...'
            end_msg = 'Writing chunk to storage done.'
        else:
            start_msg = 'Recovering data...'
            end_msg = 'Recovering data done.'

        self.logger.info(start_msg + '\n')

        if self.shape[0] > 0:
            if self.shape[0] % self.chunk_size == 0:
                last_filename = f"chunk_{int(self.storage_worker.get_all_files()[-1].split('_')[-1]) + 1}"
            else:
                last_filename = self.storage_worker.get_all_files()[-1]
        else:
            last_filename = 'chunk_0'

        # Only print progress bar if the logger level is less than 20, which means INFO or DEBUG
        for data, ids, fields in tqdm(self.wal_worker.get_file_iterator(), desc="Data persisting",
                                      total=self.wal_worker.file_number, unit='chunk', disable=self.logger.level > 20):
            self.storage_worker.write(data, ids)
            # store fields index
            for _id, _field in zip(ids, fields):
                self.field_index.store(_field, int(_id))

            # insert data to indexer
            if hasattr(self, "indexer"):
                self.indexer.index_insert(data, ids)

        if hasattr(self, "indexer"):
            if self.indexer.ivf is not None:
                self.indexer.ivf_insert(last_filename)
            # update filenames
            self.indexer.update_filenames()

        self.logger.info(end_msg)
        self.field_index.commit()
        self.wal_worker.reincarnate()

    def commit(self):
        """
        Save the collection, ensuring that all data is written to disk.
        """
        with self.threadlock:
            if not self.COMMIT_FLAG:
                self.logger.info('Saving data...')
                if hasattr(self, 'buffer'):
                    if len(self.buffer) > 0:
                        self.bulk_add_items(self.buffer)

                self.commit_data()

                # save id filter
                self.logger.debug('Saving id filter...')
                self.id_filter.to_file(self.filter_path)

                # remove buffer
                self._remove_buffer()

                self.COMMIT_FLAG = True

                self.last_commit_time = datetime.now()

    def _process_vector_item(self, vector, index, field):
        if index in self.id_filter:
            raise ValueError(f'id {index} already exists.')

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if vector.shape[1] != self.dim:
            raise ValueError(f'vector dim error, expect {self.dim}, got {vector.shape[1]}')

        if vector.dtype != self.dtypes:
            vector = vector.astype(self.dtypes)

        return vector, index, field if field is not None else {}

    def bulk_add_items(self, vectors):
        """
        Bulk add vectors to the collection in batches.

        It is recommended to use incremental ids for best performance.

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved.
                Each vector can be a tuple of (vector, id, field).

        Returns:
            list: A list of indices where the vectors are stored.
        """
        raise_if(ValueError, not isinstance(vectors, (tuple, list)),
                 f'vectors must be tuple or list, got {type(vectors)}')

        data = []
        indices = []
        fields = []

        for i in range(0, len(vectors), self.chunk_size):
            batch = vectors[i:i + self.chunk_size]

            for sample in batch:
                sample_len = len(sample)

                if sample_len == 3:
                    vector, id, field = sample
                elif sample_len == 2:
                    vector, id = sample
                    field = {}
                else:
                    raise ValueError('Each sample must be a tuple of (vector, id, field[optional]).')

                raise_if(TypeError, not (isinstance(field, dict) or field is None),
                         f'field must be dict or None, got {type(field)}')

                if isinstance(vector, list):
                    vector = np.array(vector)

                raise_if(ValueError, (not isinstance(id, int)) or id < 0,
                         f'id must be integer and greater than 0, got {id}')

                field = {} if field is None else field
                vector, id, field = self._process_vector_item(vector, id, field)

                data.append(vector)
                indices.append(id)
                fields.append(field)

                self.id_filter.add(id)

        self.wal_worker.write_log_data(data, indices, fields)
        self.COMMIT_FLAG = False

        return indices

    def _define_buffer(self, buffer_size: int):
        self.buffer = []
        self.buffer_size = buffer_size

    def _remove_buffer(self):
        if hasattr(self, 'buffer'):
            del self.buffer
        if hasattr(self, 'buffer_size'):
            del self.buffer_size

    def _insert_buffer(self, vector, id: int, field: dict, buffer_size: int):
        if not hasattr(self, 'buffer'):
            self._define_buffer(buffer_size)
        elif getattr(self, 'buffer_size') != buffer_size:
            self._remove_buffer()
            self._define_buffer(buffer_size)

        if len(self.buffer) < self.buffer_size:
            self.buffer.append((vector, id, field))
        else:
            self.bulk_add_items(self.buffer)
            self.buffer = []
            self.buffer.append((vector, id, field))

    def add_item(self, vector, id: int, field: dict = None,
                 buffer_size: Union[None, int, bool] = True) -> int:
        """
        Add a single vector to the collection.

        It is recommended to use incremental ids for best performance.

        Parameters:
            vector (np.ndarray): The vector to be added.
            id (int): The ID of the vector.
            field (dict, optional, keyword-only): The field of the vector. Default is None.
                If None, the field will be set to an empty string.
            buffer_size (int or bool or None): The buffer size for the storage worker. Default is True.
                If None, the vector will be directly written to the disk.
                If True, the buffer_size will be set to chunk_size,
                    and the vectors will be written to the disk when the buffer is full.
                If False, the vector will be directly written to the disk.
                If int, when the buffer is full, the vectors will be written to the disk.

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

        raise_if(ValueError, (not isinstance(id, int)) or id < 0,
                 f'id must be integer and greater than 0, got {id}')

        vector, id, field = self._process_vector_item(vector, id, field)

        if buffer_size is not None and buffer_size is not False:
            if buffer_size is True:
                buffer_size = self.chunk_size

            self._insert_buffer(vector, id, field, buffer_size)
        else:
            self.wal_worker.write_log_data(vector,
                                           id.reshape(1) if isinstance(id, np.ndarray) else np.array([id]), [field])

        self.COMMIT_FLAG = False

        return id

    def delete(self):
        """Delete collection."""
        with self.threadlock:
            if not self.collections_path_parent.exists():
                return None

            # stop wal
            self.wal_worker.stop()

            retries = 3
            delay = 1  # 每次重试的间隔时间（秒）

            for attempt in range(retries):
                try:
                    shutil.rmtree(self.collections_path_parent)
                    self.logger.info(f"Successfully deleted directory {self.collections_path_parent}")
                    break
                except PermissionError as e:
                    if attempt < retries - 1:
                        self.logger.warning(f"Failed to delete directory, attempting {attempt + 1}...")
                        time.sleep(delay)
                    else:
                        self.logger.error(f"Failed to delete directory {self.collections_path_parent}, file may be locked.")
                        raise e
                except Exception as e:
                    self.logger.error(f"Error deleting: {e}")

            self.IS_DELETED = True

            self._initialize_fields_index()
            self._initialize_id_checker()

            # clear cache
            self.storage_worker.clear_cache()

    @property
    def shape(self):
        """
        Get the shape of the collection.
        """
        with self.threadlock:
            return tuple(self.storage_worker.get_shape(read_type='all'))
