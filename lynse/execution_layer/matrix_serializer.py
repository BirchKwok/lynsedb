import copy
import json
from datetime import datetime
from pathlib import Path
import shutil
from typing import Union

import numpy as np
from lynse.computational_layer.engines import to_normalize
from spinesUtils.asserts import raise_if
from spinesUtils.logging import Logger

from lynse.core_components.locks import ThreadLock
from lynse.core_components.id_checker import IDChecker
from lynse.core_components.scaler import ScalarQuantization
from lynse.core_components.kv_cache import VeloKV
from lynse.storage_layer.storage import PersistentFileStorage, TemporaryFileStorage


class MatrixSerializer:
    """
    The MatrixSerializer class is used to serialize and deserialize the matrix data.
    """
    def __init__(
            self,
            dim: int,
            collection_path: Union[str, Path],
            logger: Logger,
            chunk_size: int = 1_000_000,
            dtypes: str = 'float32',
            scaler_bits=None,
            warm_up: bool = False,
            buffer_size: int = 20,
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
            scaler_bits (int): The number of bits for scalar quantization. Default is None.
                Options are 8, 16, 32. If None, scalar quantization will not be used.
                The 8 bits for uint8, 16 bits for uint16, 32 bits for uint32.
            warm_up (bool): Whether to warm up the collection. Default is False.
            buffer_size (int): The buffer size for the storage worker. Default is 20.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        self.last_commit_time = None
        # set commit flag, if the flag is True, the collection will not be saved
        self.COMMIT_FLAG = True
        self.IS_DELETED = False

        self.logger = logger

        # set parent path
        self._initialize_components_path(collection_path)

        self._dtypes_map = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}

        # set dtypes
        self.dtypes = self._dtypes_map[dtypes]
        # set dim
        self.dim = dim
        # set chunk size
        self.chunk_size = chunk_size

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
                                                    warm_up=warm_up, buffer_size=buffer_size)
        self.tempfile_storage_worker = TemporaryFileStorage(self.chunk_size)

        # ============== Loading or create one empty collection ==============
        # first of all, initialize a collection
        self.database = []
        self.indices = []
        self.fields = []

        self._initialize_fields_index()
        self._initialize_id_checker()

        self.threadlock = ThreadLock()

    def _initialize_components_path(self, collections_path):
        """make directory if not exist"""
        self.collections_path_parent = (Path(collections_path).parent.absolute() /
                                        Path(collections_path).absolute().name)

        self.collections_path_parent.mkdir(parents=True, exist_ok=True)

        # sq_model path
        self.sq_model_path = self.collections_path_parent / 'sq_model'
        # kv_index path
        self.kv_index_path = self.collections_path_parent / 'fields_index'
        # scaled_status path
        self.scaled_status_path = self.collections_path_parent / 'scaled_status.json'
        # set filter path
        self.filter_path = self.collections_path_parent / 'id_filter'

    def _initialize_scalar_quantization(self):
        if self.sq_model_path.exists():
            self.scaler = ScalarQuantization.load(self.sq_model_path)
        else:
            self.scaler = ScalarQuantization(bits=self.scaler_bits, decode_dtype=self.dtypes)

    def _initialize_fields_index(self):
        """initialize fields index"""
        self.kv_index = VeloKV(self.kv_index_path)
        self.temp_kv_index = VeloKV(self.kv_index_path.parent / 'temp_kv_index')

    def _initialize_id_checker(self):
        """initialize id checker and shape"""
        self.id_filter = IDChecker()

        if self.filter_path.exists():
            self.id_filter.from_file(self.filter_path)
        else:
            if self.collections_path.exists():
                filenames = self.storage_worker.get_all_files()
                for filename in filenames:
                    database, indices = self.storage_worker.read(filename=filename)
                    self.id_filter.add(indices)

        self.last_id = self.id_filter.find_max_value()

        self.temp_id_filter = IDChecker()

    def reset_collection(self):
        """Reset the database to its initial state with zeros."""
        self.database = []
        self.indices = []
        self.fields = []

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

    def save_chunk_immediately(self):
        """
        Save the current state of the collection to file.

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

    def rollback(self, backups=None):
        """
        Rollback the collection to the last commit.
        """
        if not self.COMMIT_FLAG:
            self.reset_collection()
            self.tempfile_storage_worker.reincarnate()
            self.temp_id_filter = IDChecker()
            self.temp_kv_index = VeloKV(self.kv_index_path.parent / 'temp_kv_index')

            if backups is not None:
                self.id_filter = backups['id_filter']
                self.kv_index = backups['kv_index']

            self.COMMIT_FLAG = True

    def commit_data(self):
        self.logger.debug('Writing chunk to storage...')
        for (data, indices), data_path, indices_path in self.tempfile_storage_worker.get_file_iterator():
            self.storage_worker.write(data, indices)
            # remove temp file
            data_path.unlink()
            indices_path.unlink()

        self.tempfile_storage_worker.reincarnate()

    def backup(self):
        """
        Backup the current state of the collection.

        Returns:
            dict: A dictionary containing the backup data.
        """
        return {
            'id_filter': copy.deepcopy(self.id_filter),
            'kv_index': copy.deepcopy(self.kv_index)
        }

    def commit(self):
        """
        Save the collection, ensuring that all data is written to disk.
        """
        with self.threadlock:
            backups = self.backup()

            if not self.COMMIT_FLAG:
                try:
                    self.logger.debug('Saving chunk immediately...')
                    self.save_chunk_immediately()
                    self.commit_data()

                    self.logger.debug('Concatenating id filter...')
                    # concat filter
                    self.id_filter.concat(self.temp_id_filter)
                    # concat fields index
                    self.logger.debug('Concatenating fields index...')
                    self.kv_index.concat(self.temp_kv_index)
                    # save id filter
                    self.logger.debug('Saving id filter...')
                    self.id_filter.to_file(self.filter_path)
                    # # save fields index
                    self.logger.debug('Remove temp fields index...')
                    self.temp_kv_index.delete()

                except Exception as e:
                    self.logger.error(f'Error occurred while concatenating the fields index: {e}, rollback...')
                    self.rollback(backups)
                    raise e

                if self.storage_worker.get_shape()[0] >= 100000:
                    filenames = self.storage_worker.get_all_files()

                    if not self.scaled_status_path.exists() \
                            and self.scaler_bits is not None:
                        # only run once
                        self.storage_worker.update_quantizer(self.scaler)

                        if filenames:
                            self.storage_worker.write(filenames=filenames)

                        with open(self.scaled_status_path, 'w') as f:
                            json.dump({'status': True}, f)

                if self.scaler_bits is not None:
                    if self.scaler.fitted:
                        self.scaler.save(self.sq_model_path)

                self.COMMIT_FLAG = True

                self.reset_collection()
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

    def bulk_add_items(self, vectors, normalize: bool = False):
        """
        Bulk add vectors to the collection in batches.

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved. Each vector can be a tuple of (
                vector, id, field).
            normalize (bool): Whether to normalize the vectors. Default is False.


        Returns:
            list: A list of indices where the vectors are stored.
        """
        raise_if(ValueError, not isinstance(vectors, (tuple, list)),
                 f'vectors must be tuple or list, got {type(vectors)}')

        with self.threadlock:
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

                    if normalize:
                        vector = to_normalize(vector)

                    self.database.append(vector)
                    self.indices.append(index)
                    new_ids.append(index)
                    self.fields.append(field)
                    self.temp_id_filter.add(index)

                self.auto_save_chunk()

            if self.COMMIT_FLAG:
                self.COMMIT_FLAG = False

            return new_ids

    def add_item(self, vector, index: int, field: dict = None, normalize: bool = False) -> int:
        """
        Add a single vector to the collection.

        Parameters:
            vector (np.ndarray): The vector to be added.
            index (int): The ID of the vector.
            field (dict, optional, keyword-only): The field of the vector. Default is None. If None, the field will be
                set to an empty string.
            normalize (bool): Whether to normalize the vector. Default is False.
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

        vector, index, field = self._process_vector_item(vector, index, field)
        if normalize:
            vector = to_normalize(vector)

        with self.threadlock:
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
        with self.threadlock:
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
            self._initialize_id_checker()

            # clear cache
            self.storage_worker.clear_cache()

    @property
    def shape(self):
        with self.threadlock:
            return tuple(self.storage_worker.get_shape(read_type='all'))
