import os.path
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import shutil

import numpy as np
from spinesUtils.asserts import raise_if

from min_vec.engines import to_normalize
from min_vec.filter import IDFilter
from min_vec.utils import io_checker
from min_vec.config import *
from min_vec.kmeans import BatchKMeans
from min_vec.scaler import ScalarQuantization
from min_vec.fields_mapper import FieldsMapper
from min_vec.storage import StorageWorker


class MatrixSerializer:
    def __init__(
            self, dim, database_path, distance, logger, n_clusters=16, chunk_size=1_000_000,
            index_mode='IVF-FLAT', dtypes='float64',
            reindex_if_conflict=False, scaler_bits=8
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
            reindex_if_conflict (bool): Whether to reindex the database if there is an index mode conflict.
                Default is False.
            scaler_bits (int): The number of bits for scalar quantization. Default is 8.
                Options are 8, 16, 32. If None, scalar quantization will not be used.
                The 8 bits for uint8, 16 bits for uint16, 32 bits for uint32.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        raise_if(ValueError, not isinstance(dim, int), 'dim must be int')
        raise_if(ValueError, not str(database_path).endswith('mvdb'), 'database_path must end with .mvdb')
        raise_if(ValueError, not isinstance(chunk_size, int) or chunk_size <= 1,
                 'chunk_size must be int and greater than 1')
        raise_if(ValueError, distance not in ('cosine', 'L2'), 'distance must be "cosine" or "L2"')
        raise_if(ValueError, index_mode not in ('FLAT', 'IVF-FLAT'), 'index_mode must be "FLAT" or "IVF-FLAT"')
        raise_if(ValueError, dtypes not in ('float16', 'float32', 'float64'),
                 'dtypes must be "float16", "float32" or "float64')
        raise_if(ValueError, not isinstance(n_clusters, int) or n_clusters <= 0,
                 'n_clusters must be int and greater than 0')
        raise_if(ValueError, not isinstance(reindex_if_conflict, bool), 'reindex_if_conflict must be bool')
        raise_if(ValueError, scaler_bits not in (8, 16, 32, None), 'sq_bits must be 8, 16, 32 or None')

        self.last_commit_time = None
        self.reindex_if_conflict = reindex_if_conflict
        # set commit flag, if the flag is True, the database will not be saved
        self.COMMIT_FLAG = True
        # set flag for scalar quantization, if the flag is True, the database will be rescanned for scalar quantization
        self.RESCAN_FOR_SQ_FLAG = False

        self.IS_DELETED = None

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
        # set device
        self.device = torch.device(MVDB_COMPUTE_DEVICE)
        # set scalar quantization bits
        self.scaler_bits = scaler_bits if scaler_bits is not None else None

        # set database path
        self.database_path = self.database_path_parent / Path(database_path).name

        # initialize the storage worker
        self.storage_worker = StorageWorker(self.database_path, self.dim, self.chunk_size)

        self.logger.info(f"Initializing database folder path: '{'.mvdb'.join(database_path.split('.mvdb')[:-1])}/',"
                         " database file path: "
                         f"'{'.mvdb'.join(database_path.split('.mvdb')[:-1]) + '/' + database_path}'")

        # ============== Loading or create one empty database ==============
        # first of all, initialize a database
        self.database = []
        self.indices = []
        self.fields = []

        # check initialize params
        self._check_initialize_params(dtypes, reindex_if_conflict)

        if self.scaler_bits is not None:
            self._initialize_scalar_quantization()

        self._initialize_fields_mapper()
        self._initialize_ann_model()
        self._initialize_id_filter()

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

    def _check_initialize_params(self, dtypes, reindex_if_conflict):
        """check initialize params"""
        write_params_flag = False
        if self.database_path.exists():
            self.logger.info('Database file exists, reading parameters...')

            attrs = self.storage_worker.read_file_attributes()
            dim = attrs.get('dim', self.dim)
            chunk_size = attrs.get('chunk_size', self.chunk_size)
            distance = attrs.get('distance', self.distance)
            file_dtypes = attrs.get('dtypes', dtypes)
            old_index_mode = attrs.get('index_mode', None)
            old_sq_bits = attrs.get('sq_bits', self.scaler_bits)

            if dim != self.dim:
                self.logger.warning(
                    f'* dim={dim} in the file is not equal to the dim={self.dim} in the parameters, '
                    f'the parameter dim will be covered by the dim in the file.')
                self.dim = dim
                write_params_flag = True

            if chunk_size != self.chunk_size:
                self.logger.warning(
                    f'* chunk_size={chunk_size} in the file is not '
                    f'equal to the chunk_size={self.chunk_size}'
                    f'in the parameters, the parameter chunk_size will be covered by the chunk_size '
                    f'in the file.'
                )
                self.chunk_size = chunk_size
                write_params_flag = True

            if distance != self.distance:
                self.logger.warning(f'* distance=\'{distance}\' in the file is not '
                                    f'equal to the distance=\'{self.distance}\' '
                                    f'in the parameters, the parameter distance will be covered by the distance '
                                    f'in the file.')
                self.distance = distance
                write_params_flag = True

            if file_dtypes != dtypes:
                self.logger.warning(
                    f'* dtypes=\'{file_dtypes}\' in the file is not equal to the dtypes=\'{dtypes}\' '
                    f'in the parameters, the parameter dtypes will be covered by the dtypes '
                    f'in the file.')
                self.dtypes = self._dtypes_map[attrs.get('dtypes', dtypes)]
                write_params_flag = True

            if old_index_mode != self.index_mode and self.index_mode == 'IVF-FLAT':
                if reindex_if_conflict:
                    self.logger.warning(
                        f'* index_mode=\'{old_index_mode}\' in the file is not equal to the index_mode='
                        f'\'{self.index_mode}\' in the parameters, if you really want to change the '
                        f'index_mode to \'IVF-FLAT\', you need to run `commit()` function after initializing '
                        f'the database.')

                    self.COMMIT_FLAG = False
                else:
                    self.logger.warning(
                        f'* index_mode=\'{old_index_mode}\' in the file is not equal to the index_mode='
                        f'\'{self.index_mode}\' in the parameters, if you really want to change the '
                        f'index_mode to \'IVF-FLAT\', you need to set `reindex_if_conflict=True` first '
                        f'and run `commit()` function after initializing the database.')

            if self.scaler_bits is not None and old_sq_bits != self.scaler_bits:
                self.logger.warning(f'* sq_bits={old_sq_bits} in the file is not equal to the sq_bits='
                                    f'{self.scaler_bits} in the parameters, '
                                    f'the sq_bits in the file will be covered by the sq_bits in the parameters.')
                self.RESCAN_FOR_SQ_FLAG = True

            self.logger.info('Reading parameters done.')

        if write_params_flag or not self.database_path.exists():
            self._write_params(dtypes)

    def _initialize_parent_path(self, database_path):
        """make directory if not exist"""
        self.database_path_parent = Path(database_path).parent.absolute() / Path(
            '.mvdb'.join(Path(database_path).absolute().name.split('.mvdb')[:-1]))

        if not self.database_path_parent.exists():
            self.database_path_parent.mkdir(parents=True)

    def _initialize_scalar_quantization(self):
        if Path(self.database_path_parent / 'sq_model.mvdb').exists():
            self.scaler = ScalarQuantization.load(self.database_path_parent / 'sq_model.mvdb')
        else:
            self.scaler = ScalarQuantization(bits=self.scaler_bits, decode_dtype=self.dtypes)
            # if the database file exists, the model will be fitted
            if self.database_path.exists() and self.RESCAN_FOR_SQ_FLAG:
                for i in self.iterable_dataloader(read_chunk_only=False):
                    self.scaler.partial_fit(i[0])

    def _initialize_ann_model(self):
        """initialize ann model"""
        if self.index_mode == 'IVF-FLAT':
            self.ann_model = BatchKMeans(n_clusters=self.n_clusters, random_state=0,
                                         batch_size=10240, epochs=MVDB_KMEANS_EPOCHS, distance=self.distance)

            if Path(self.database_path_parent / 'ann_model.mvdb').exists() and self.index_mode == 'IVF-FLAT':
                self.ann_model = self.ann_model.load(self.database_path_parent / 'ann_model.mvdb')
                if (self.ann_model.n_clusters != self.n_clusters or self.ann_model.distance != self.distance or
                        self.ann_model.epochs != MVDB_KMEANS_EPOCHS or
                        self.ann_model.batch_size != 10240):
                    self.ann_model = BatchKMeans(n_clusters=self.n_clusters, random_state=0,
                                                 batch_size=10240, epochs=MVDB_KMEANS_EPOCHS, distance=self.distance)
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
        if self.scaler_bits is not None:
            decoder = self.scaler.decode
        else:
            decoder = lambda x: x

        read_type = 'all' if not read_chunk_only else 'chunk'

        for data, indices, fields in self.storage_worker.read(read_type=read_type, reverse=from_tail):
            if mode == 'lazy':
                yield decoder(data), indices, fields
            else:
                yield decoder(data), indices, self.fields_mapper.decode(fields)

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
        if self.scaler_bits is not None:
            raise_if(ValueError, not self.scaler.fitted, 'The model must be fitted before decoding.')
            decoder = self.scaler.decode
        else:
            decoder = lambda x: x

        for data, indices, fields in self.storage_worker.read(read_type='cluster', cluster_id=str(cluster_id)):
            if mode == 'lazy':
                yield decoder(data), indices, fields
            else:
                yield decoder(data), indices, self.fields_mapper.decode(fields)

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

    def reset_chunk_partition(self):
        """
        Reset the chunk partition.
        """
        self.storage_worker.delete_chunk()

    def save_data(self, data, indices, fields, to_chunk=True, cluster_id=None):
        """Optimized method to save data to chunk or cluster group with reusable logic."""
        data_type = self.dtypes

        if self.scaler_bits is not None:
            self.scaler.partial_fit(data)
            data = self.scaler.encode(data)
            data_type = self.scaler.bits

        if isinstance(fields, str) or all(isinstance(i, str) for i in fields):
            fields_indices = self.fields_mapper.encode(fields)
        else:
            fields_indices = fields

        self.storage_worker.write(data, indices, fields_indices,
                                  write_type='chunk' if to_chunk else 'cluster',
                                  cluster_id=str(cluster_id) if cluster_id is not None else cluster_id,
                                  data_dtype=data_type)

    @io_checker
    def save_chunk_immediately(self):
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
            to_chunk=True,
            cluster_id=None
        )

        self.reset_database()  # reset database, indices and fields

    def auto_save_chunk(self):
        self._length_checker()
        if len(self.database) >= self.chunk_size:
            self.save_chunk_immediately()

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
            self.logger.info('Saving chunk immediately...')
            self.save_chunk_immediately()

            self.logger.info('Saving id filter...')
            # save filter
            self.id_filter.to_file(self._filter_path)

            # save scalar quantization model
            if self.scaler_bits is not None:
                self.logger.info('Saving scalar quantization model...')
                self.scaler.save(self.database_path_parent / 'sq_model.mvdb')

            # save fields mapper
            self.logger.info('Saving fields mapper...')
            self.fields_mapper.save(self.database_path_parent / 'fields_mapper.mvdb')

            chunk_partition_size = self.storage_worker.get_shape(read_type='chunk')[0]
            if chunk_partition_size >= 100000 and self.index_mode != 'FLAT':
                self.logger.info('Building index...')
                # build index
                self.build_index()

                # save ivf index and k-means model
                self.logger.info('Saving ann model...')
                self.ann_model.save(self.database_path_parent / 'ann_model.mvdb')

            self.reset_database()

            # save params
            self.storage_worker.write_file_attributes({'index_mode': self.index_mode})

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

    def _process_vector_item(self, vector, index, field, normalize):
        if index in self.id_filter:
            raise ValueError(f'id {index} already exists')

        if len(vector) != self.dim:
            raise ValueError(f'vector dim error, expect {self.dim}, got {len(vector)}')

        vector = vector.astype(self.dtypes)

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if normalize:
            vector = to_normalize(vector)

        return vector, index, field if field is not None else ''

    def bulk_add_items(self, vectors, normalize: bool = False, save_immediately=False):
        """
        Bulk add vectors to the database in batches.

        Parameters: vectors (list or tuple): A list or tuple of vectors to be saved. Each vector can be a tuple of (
            vector, id, field).
        normalize (bool): Whether to normalize the input vector.
        save_immediately (bool): Whether to save the database immediately.

        Returns:
            list: A list of indices where the vectors are stored.
        """
        raise_if(ValueError, not isinstance(vectors, (tuple, list)),
                 f'vectors must be tuple or list, got {type(vectors)}')
        raise_if(ValueError, not isinstance(normalize, bool),
                 f'normalize must be bool, got {type(normalize)}')
        raise_if(ValueError, not isinstance(save_immediately, bool),
                 f'save_immediately must be bool, got {type(save_immediately)}')
        raise_if(ValueError, not all(1 <= len(i) <= 3 and isinstance(i, tuple) for i in vectors),
                 f'vectors must be tuple of (vector, id, field), got {vectors}')

        batch_size = MVDB_BULK_ADD_BATCH_SIZE

        new_ids = []

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            # temp_vectors, temp_indices, temp_fields = [], [], []

            for sample in batch:
                if len(sample) == 1:
                    vector = sample[0]
                    index = None
                    field = ''
                elif len(sample) == 2:
                    vector, index = sample
                    field = ''
                else:
                    vector, index, field = sample

                index = self._generate_new_id() if index is None else index

                raise_if(ValueError, index < 0, f'id must be greater than 0, got {index}')

                field = '' if field is None else field
                vector, index, field = self._process_vector_item(vector, index, field, normalize)

                self.database.append(vector)
                internal_id = index if index is not None else self._generate_new_id()
                self.indices.append(internal_id)
                new_ids.append(internal_id)
                self.fields.append(field)
                self.id_filter.add(index)

            self.save_chunk_immediately() if save_immediately else self.auto_save_chunk()

        if self.COMMIT_FLAG:
            self.COMMIT_FLAG = False

        return new_ids

    def add_item(self, vector, index: int = None, field: str = None, normalize: bool = False,
                 save_immediately=False) -> int:
        """
        Add a single vector to the database.

        Parameters:
            vector (np.ndarray): The vector to be added.
            index (int, optional): Optional ID for the vector.
            field (str, optional): Optional field for the vector.
            normalize (bool): Whether to normalize the input vector.
            save_immediately (bool): Whether to save the database immediately.
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
        raise_if(ValueError, not isinstance(normalize, bool),
                 f'normalize must be bool, got {type(normalize)}')
        raise_if(ValueError, not isinstance(save_immediately, bool),
                 f'save_immediately must be bool, got {type(save_immediately)}')

        index = self._generate_new_id() if index is None else index

        raise_if(ValueError, index in self.id_filter, f'id {index} already exists')

        vector, index, field = self._process_vector_item(vector, index, field, normalize)

        # Add the id to then filter.
        self.id_filter.add(index)

        self.database.append(vector)
        self.indices.append(index)
        self.fields.append(field)

        self.save_chunk_immediately() if save_immediately else self.auto_save_chunk()

        if self.COMMIT_FLAG:
            self.COMMIT_FLAG = False

        return index

    def delete(self):
        """Delete database."""
        if not self.database_path_parent.exists():
            return None

        try:
            shutil.rmtree(self.database_path_parent)
            self.IS_DELETED = True
        except FileNotFoundError:
            pass

        self.reset_database()

        # reinitialize
        if self.scaler_bits is not None:
            self._initialize_scalar_quantization()

        self._initialize_fields_mapper()
        self._initialize_ann_model()
        self._initialize_id_filter()

    def _kmeans_clustering(self, data):
        """kmeans clustering"""
        self.ann_model = self.ann_model.partial_fit(data)

        return self.ann_model.cluster_centers_, self.ann_model.labels_

    def _clustering_for_ivf(self, data, indices, fields):
        # Perform K-means clustering.
        self.cluster_centers, labels = self._kmeans_clustering(data)

        # Current batch's IVF index.
        ivf_index = {i: [] for i in range(self.n_clusters)}
        # Allocate data to various cluster centers.
        for idx, label in enumerate(labels):
            try:
                ivf_index[label].append([idx, indices[idx], fields[idx]])
            except Exception as e:
                self.logger.error(f"Error in adding to cluster: {e}")
                raise e

        return ivf_index

    def _build_index(self):
        for database, indices, fields in self.iterable_dataloader(read_chunk_only=True, mode='lazy'):
            # Create an IVF index.
            ivf_index = self._clustering_for_ivf(database, indices, fields)
            # reorganize files
            self.save_cluster_partitions(database, ivf_index)

    def build_index(self):
        """
        Build the IVF index.
        """
        self._build_index()

        self.reset_chunk_partition()

    def save_cluster_partitions(self, database, ivf_index):
        """
        Rearrange files according to the IVF index.
        """
        for cluster_id in range(self.n_clusters):
            all_vectors, all_indices, all_fields = [], [], []

            for cid, _ in ivf_index.items():
                if len(_) == 0:
                    continue
                vec_idx, index, field = zip(*_)
                if cid == cluster_id:
                    # Extract all vectors, indices, and fields from the current cluster center.
                    all_vectors.extend([database[i] for i in vec_idx])
                    all_indices.extend(index)
                    all_fields.extend(field)

            if len(all_vectors) == 0:
                continue

            self.save_data(all_vectors, all_indices, all_fields, to_chunk=False,
                           cluster_id=cluster_id)

    @lru_cache(maxsize=1)
    def _get_shape(self, mtime):
        """
        Return the shape of the entire database.

        Returns:
            tuple: The number of vectors and the dimension of each vector in the database.
        """
        return self.storage_worker.get_shape(read_type='all')

    @property
    def shape(self):
        self.check_commit()
        return self._get_shape(int(os.path.getmtime(self.database_path) * 1000))
