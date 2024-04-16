from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
from spinesUtils.asserts import raise_if
from spinesUtils.logging import Logger

from min_vec.data_structures.filter import IDFilter
from min_vec.utils.utils import io_checker
from min_vec.configs.config import config
from min_vec.data_structures.kmeans import BatchKMeans
from min_vec.data_structures.scaler import ScalarQuantization
from min_vec.data_structures.fields_mapper import FieldsMapper
from min_vec.storage_layer.storage import StorageWorker
from min_vec.execution_layer.cluster_worker import ClusterWorker


class MatrixSerializer:
    """
    The MatrixSerializer class is used to serialize and deserialize the matrix data.
    """
    def __init__(
            self,
            dim: int,
            database_path: str,
            distance: str,
            logger: Logger,
            n_clusters: int = 16,
            chunk_size: int = 1_000_000,
            index_mode: str = 'IVF-FLAT',
            dtypes: str = 'float32',
            scaler_bits=None,
            warm_up: bool = False
    ) -> None:
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str): Path to the database file.
            distance (str): Method for calculating vector distance.
                Options are 'cosine' or 'L2' for Euclidean distance.
            logger (Logger): The logger object.
            n_clusters (int): The number of clusters for the IVF-FLAT index. Default is 16.
            chunk_size (int): The size of each data chunk. Default is 1_000_000.
            index_mode (str): The index mode of the database.
                Options are 'FLAT' or 'IVF-FLAT'. Default is 'IVF-FLAT'.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            scaler_bits (int): The number of bits for scalar quantization. Default is None.
                Options are 8, 16, 32. If None, scalar quantization will not be used.
                The 8 bits for uint8, 16 bits for uint16, 32 bits for uint32.
            warm_up (bool): Whether to warm up the database. Default is False.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        self.last_commit_time = None
        # set commit flag, if the flag is True, the database will not be saved
        self.COMMIT_FLAG = True
        # set flag for scalar quantization, if the flag is True, the database will be rescanned for scalar quantization
        self.RESCAN_FOR_SQ_FLAG = False

        self.IS_DELETED = False

        self.logger = logger

        # set parent path
        self._initialize_parent_path(database_path)

        self._dtypes_map = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}

        # set distance
        self.distance = distance
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
        self._filter_path = self.database_path_parent / 'id_filter.mvdb'

        # set database path
        self.database_path = self.database_path_parent / Path(database_path).name.split('.mvdb')[0]

        # set scalar quantization bits
        self.scaler_bits = scaler_bits if scaler_bits is not None else None
        self.scaler = None
        if self.scaler_bits is not None:
            self._initialize_scalar_quantization()

        # initialize the storage worker
        self.storage_worker = StorageWorker(self.database_path_parent, self.dim,
                                            self.chunk_size,
                                            quantizer=None if self.scaler_bits is None else self.scaler,
                                            warm_up=warm_up)

        self.logger.info(f"Initializing database folder path: '{'.mvdb'.join(database_path.split('.mvdb')[:-1])}/'")

        # ============== Loading or create one empty database ==============
        # first of all, initialize a database
        self.database = []
        self.indices = []
        self.fields = []

        self._initialize_fields_mapper()
        self._initialize_ann_model()
        self._initialize_id_filter()

        self.cluster_worker = ClusterWorker(
            logger=self.logger,
            iterable_dataloader=self.iterable_dataloader,
            ann_model=self.ann_model,
            storage_worker=self.storage_worker,
            save_data=self.save_data,
            n_clusters=self.n_clusters
        )

        # save initial parameters
        self._write_params(dtypes=dtypes)

        if self._get_cluster_dataset_num() > 0 and self.index_mode == 'FLAT':
            # cause the index mode is FLAT, but the cluster dataset is not empty,
            # so the clustered datasets will also be traversed during querying.
            self.logger.warning('The index mode is FLAT, but the cluster dataset is not empty, '
                                'so the clustered datasets will also be traversed during querying.')

    def _write_params(self, dtypes):
        attrs = {
            'dim': self.dim,
            'chunk_size': self.chunk_size,
            'distance': self.distance,
            'dtypes': dtypes
        }

        if self.scaler_bits is not None:
            attrs['sq_bits'] = self.scaler_bits

        self.storage_worker.write_file_attributes(attrs)

    def _initialize_parent_path(self, database_path):
        """make directory if not exist"""
        self.database_path_parent = Path(database_path).parent.absolute() / Path(
            '.mvdb'.join(Path(database_path).absolute().name.split('.mvdb')[:-1]))

        self.database_path_parent.mkdir(parents=True, exist_ok=True)

    def _initialize_scalar_quantization(self):
        if Path(self.database_path_parent / 'sq_model.mvdb').exists():
            self.scaler = ScalarQuantization.load(self.database_path_parent / 'sq_model.mvdb')
        else:
            self.scaler = ScalarQuantization(bits=self.scaler_bits, decode_dtype=self.dtypes)

    def _initialize_ann_model(self):
        """initialize ann model"""
        if self.index_mode == 'IVF-FLAT':
            MVDB_KMEANS_EPOCHS = config.MVDB_KMEANS_EPOCHS
            self.ann_model = BatchKMeans(n_clusters=self.n_clusters, random_state=0,
                                         batch_size=10240, epochs=MVDB_KMEANS_EPOCHS)

            if Path(self.database_path_parent / 'ann_model.mvdb').exists() and self.index_mode == 'IVF-FLAT':
                self.ann_model = self.ann_model.load(self.database_path_parent / 'ann_model.mvdb')
                if (self.ann_model.n_clusters != self.n_clusters or
                        self.ann_model.epochs != MVDB_KMEANS_EPOCHS or
                        self.ann_model.batch_size != 10240):
                    self.ann_model = BatchKMeans(n_clusters=self.n_clusters, random_state=0,
                                                 batch_size=10240, epochs=MVDB_KMEANS_EPOCHS)
        else:
            self.ann_model = None

    def _initialize_fields_mapper(self):
        """initialize fields mapper"""
        if Path(self.database_path_parent / 'fields_mapper.mvdb').exists():
            self.fields_mapper = FieldsMapper().load(self.database_path_parent / 'fields_mapper.mvdb')
        else:
            self.fields_mapper = FieldsMapper()

    def _initialize_id_filter(self):
        """initialize id filter and shape"""
        self.id_filter = IDFilter()

        if self._filter_path.exists():
            self.id_filter.from_file(self._filter_path)
        else:
            if self.database_path.exists():
                for database, indices, fields in self.storage_worker.read(read_type='all'):
                    self.id_filter.add(indices)

        self.last_id = self.id_filter.find_max_value()

    def check_commit(self):
        if not self.COMMIT_FLAG:
            raise ValueError("Did you forget to run `commit()` function ? Try to run `commit()` first.")

    def reset_database(self):
        """Reset the database to its initial state with zeros."""
        self.database = []
        self.indices = []
        self.fields = []

    @io_checker
    def iterable_dataloader(self, read_chunk_only=False, from_tail=False, mode='eager'):
        """
        Generator for loading the database and index.

        Parameters:
            read_chunk_only (bool): Whether to read only the chunk.
            from_tail (bool): Whether to read from the end of the file.
            mode (str): The mode of the generator. Options are 'eager' or 'lazy'. Default is 'eager'.

        Yields:
            tuple: A tuple of (database, index, field).

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
            PermissionError: If the file cannot be read due to permission issues.
            UnKnownError: If an unknown error occurs.
        """
        read_type = 'all' if not read_chunk_only else 'chunk'

        for data, indices, fields in self.storage_worker.read(read_type=read_type, reverse=from_tail):
            if data is None:
                continue

            if mode == 'lazy':
                yield data, indices, fields
            else:
                yield data, indices, self.fields_mapper.decode(fields)

    def cluster_dataloader(self, cluster_id, mode='eager'):
        """
        Generator for loading the database and index. Only used for querying when index_mode is IVF-FLAT.

        Parameters:
            cluster_id (int): The cluster id.
            mode (str): The mode of the generator. Options are 'eager' or 'lazy'. Default is 'eager'.

        Yields:
            tuple: A tuple of (database, index, field).

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
            PermissionError: If the file cannot be read due to permission issues.
            UnKnownError: If an unknown error occurs.
        """
        for data, indices, fields in self.storage_worker.read(read_type='cluster', cluster_id=str(cluster_id)):
            if mode == 'lazy':
                yield data, indices, fields
            else:
                yield data, indices, self.fields_mapper.decode(fields)

    def _is_database_reset(self):
        """
        Check if the database is in its reset state (empty list).

        Returns:
            bool: True if the database is reset, False otherwise.
        """
        return not (self.database and self.indices and self.fields)

    def _length_checker(self):
        """
        Check if the length of the database and index are equal.

        Raises:
            ValueError: If the lengths of the database, indices, and fields are not the same.
        """
        if not self._is_database_reset() and not self.COMMIT_FLAG:
            if not (len(self.database) == len(self.indices) == len(self.fields)):
                raise ValueError('The database, index length and field length not the same.')

    def save_data(self, data, indices, fields, write_chunk=True, cluster_id=None, normalize=False):
        """Optimized method to save data to chunk or cluster group with reusable logic."""

        if isinstance(fields, str) or all(isinstance(i, str) for i in fields):
            fields_indices = self.fields_mapper.encode(fields)
        else:
            fields_indices = fields

        self.storage_worker.write(data, indices, fields_indices,
                                  write_type='chunk' if write_chunk else 'cluster',
                                  cluster_id=str(cluster_id) if cluster_id is not None else cluster_id,
                                  normalize=normalize)

    @io_checker
    def save_chunk_immediately(self, normalize=True):
        """
        Save the current state of the database to a .mvdb file.

        Returns:
            Path: The path of the saved database file.
        """
        self._length_checker()

        if self._is_database_reset():
            return []

        self.save_data(
            self.database,
            self.indices,
            self.fields,
            write_chunk=True,
            cluster_id=None,
            normalize=normalize
        )

        self.reset_database()  # reset database, indices and fields

    def auto_save_chunk(self, normalize=True):
        self._length_checker()
        if len(self.database) >= self.chunk_size:
            self.save_chunk_immediately(normalize=normalize)

        return

    def _get_cluster_dataset_num(self):
        if not self.database_path.exists():
            return 0

        return self.storage_worker.get_cluster_dataset_num()

    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        This method is required to be called after saving vectors to query them.
        """
        if not self.COMMIT_FLAG:
            self.logger.debug('Saving chunk immediately...')
            self.save_chunk_immediately(normalize=True)

            self.logger.debug('Saving id filter...')
            # save filter
            self.id_filter.to_file(self._filter_path)

            # save fields mapper
            self.logger.debug('Saving fields mapper...')
            self.fields_mapper.save(self.database_path_parent / 'fields_mapper.mvdb')

            chunk_partition_size = self.storage_worker.get_shape(read_type='chunk')[0]
            if chunk_partition_size >= 100000 and self.index_mode != 'FLAT':
                self.logger.debug('Building index...')
                # build index
                self.build_index()

                # save ivf index and k-means model
                self.logger.debug('Saving ann model...')
                self.ann_model.save(self.database_path_parent / 'ann_model.mvdb')

            self.reset_database()

            # save params
            self.storage_worker.write_file_attributes({'index_mode': self.index_mode})

            if self.scaler_bits is not None:
                self.scaler.save(self.database_path_parent / 'sq_model.mvdb')

            self.COMMIT_FLAG = True

            self.last_commit_time = datetime.now()

    def _generate_new_id(self):
        """
        Generate a new ID for the vector.
        """
        while True:
            self.last_id += 1
            if self.last_id not in self.id_filter:
                break

        return self.last_id

    def _process_vector_item(self, vector, index, field):
        if index in self.id_filter:
            raise ValueError(f'id {index} already exists')

        if len(vector) != self.dim:
            raise ValueError(f'vector dim error, expect {self.dim}, got {len(vector)}')

        if vector.dtype != self.dtypes:
            vector = vector.astype(self.dtypes)

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        return vector, index, field if field is not None else ''

    def bulk_add_items(self, vectors):
        """
        Bulk add vectors to the database in batches.

        Parameters: vectors (list or tuple): A list or tuple of vectors to be saved. Each vector can be a tuple of (
            vector, id, field).

        Returns:
            list: A list of indices where the vectors are stored.
        """
        raise_if(ValueError, not isinstance(vectors, (tuple, list)),
                 f'vectors must be tuple or list, got {type(vectors)}')
        raise_if(ValueError, not all(isinstance(i, tuple) for i in vectors),
                 f'vectors must be tuple of (vector, id[optional], field[optional]).')

        new_ids = []

        for i in range(0, len(vectors), self.chunk_size):
            batch = vectors[i:i + self.chunk_size]

            for sample in batch:
                if len(sample) == 1:
                    vector = sample[0]
                    index = None
                    field = ''
                elif len(sample) == 2:
                    vector, index = sample
                    field = ''
                else:
                    raise_if(ValueError, len(sample) != 3,
                             f'vectors must be tuple of (vector, id[optional], field[optional]).')
                    vector, index, field = sample

                index = self._generate_new_id() if index is None else index

                raise_if(ValueError, index < 0, f'id must be greater than 0, got {index}')

                field = '' if field is None else field
                vector, index, field = self._process_vector_item(vector, index, field)

                self.database.append(vector)
                internal_id = index if index is not None else self._generate_new_id()
                self.indices.append(internal_id)
                new_ids.append(internal_id)
                self.fields.append(field)
                self.id_filter.add(index)

            self.auto_save_chunk(normalize=True)

        if self.COMMIT_FLAG:
            self.COMMIT_FLAG = False

        return new_ids

    def add_item(self, vector, *, index: int = None, field: str = None) -> int:
        """
        Add a single vector to the database.

        Parameters:
            vector (np.ndarray): The vector to be added.
            index (int, optional, keyword-only): The ID of the vector. If None, a new ID will be generated.
            field (str, optional, keyword-only): The field of the vector. Default is None. If None, the field will be
                set to an empty string.
        Returns:
            int: The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        raise_if(ValueError, len(vector) != self.dim,
                 f'vector dim error, expect {self.dim}, got {len(vector)}')
        raise_if(ValueError, vector.ndim != 1, f'vector dim error, expect 1, got {vector.ndim}')
        raise_if(ValueError, field is not None and not isinstance(field, str),
                 f'field must be str, got {type(field)}')
        raise_if(ValueError, index is not None and not isinstance(index, int), f'id must be int, got {type(index)}')
        raise_if(ValueError, index is not None and index < 0, f'id must be greater than 0, got {index}')
        raise_if(ValueError, field is not None and not isinstance(field, str),
                 f'field must be str, got {type(field)}')

        index = self._generate_new_id() if index is None else index

        raise_if(ValueError, index in self.id_filter, f'id {index} already exists')

        vector, index, field = self._process_vector_item(vector, index, field)

        # Add the id to then filter.
        self.id_filter.add(index)

        self.database.append(vector)
        self.indices.append(index)
        self.fields.append(field)

        self.auto_save_chunk(normalize=True)

        if self.COMMIT_FLAG:
            self.COMMIT_FLAG = False

        return index

    def delete(self):
        """Delete database."""
        if not self.database_path_parent.exists():
            return None

        try:
            shutil.rmtree(self.database_path_parent)
        except FileNotFoundError:
            pass

        self.IS_DELETED = True
        self.reset_database()

        # reinitialize
        if self.scaler_bits is not None:
            self._initialize_scalar_quantization()

        self._initialize_fields_mapper()
        self._initialize_ann_model()
        self._initialize_id_filter()

        # clear cache
        self.storage_worker.clear_cache()

    def build_index(self):
        """
        Build the IVF index.
        """
        self.cluster_worker.build_index(self.scaler, self.distance)

    @property
    def shape(self):
        self.check_commit()
        return tuple(self.storage_worker.get_shape(read_type='all'))
