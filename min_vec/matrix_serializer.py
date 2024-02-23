import os.path
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import shutil

import numpy as np
import h5py
from spinesUtils.asserts import raise_if

from min_vec.engines import to_normalize, silhouette_score
from min_vec.filter import BloomTrie
from min_vec.utils import io_checker, get_env_variable
from min_vec.ivf_index import IVFIndex
from min_vec.kmeans import KMeans

MVDB_KMEANS_EPOCHS = get_env_variable('MVDB_KMEANS_EPOCHS', 100, int)
MVDB_CLUSTERING_QUALITY_THRESHOLD = get_env_variable("MVDB_CLUSTERING_QUALITY_THRESHOLD", 0.3, float)
MVDB_BULK_ADD_BATCH_SIZE = get_env_variable('MVDB_BULK_ADD_BATCH_SIZE', 100000, int)
MVDB_REINDEX_CHECKING_SAMPLES = get_env_variable('MVDB_REINDEX_CHECKING_SAMPLES', 10000, int)


class MatrixSerializer:
    def __init__(
            self, dim, database_path, distance, logger, n_clusters=8, chunk_size=1_000_000,
            bloom_filter_size=100_000_000, index_mode='IVF-FLAT', dtypes='float32',
            reindex_if_conflict=False
    ) -> None:
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str): Path to the database file.
            distance (str): Method for calculating vector distance.
                Options are 'cosine' or 'L2' for Euclidean distance.
            logger (Logger): The logger object.
            n_clusters (int): The number of clusters for the IVF-FLAT index. Default is 8.
            chunk_size (int): The size of each data chunk. Default is 1_000_000.
            bloom_filter_size (int): The size of the bloom filter. Default is 100_000_000.
            index_mode (str): The index mode of the database.
                Options are 'FLAT' or 'IVF-FLAT'. Default is 'IVF-FLAT'.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            reindex_if_conflict (bool): Whether to reindex the database if there is a index mode conflict.

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
        raise_if(ValueError, not isinstance(bloom_filter_size, int) or bloom_filter_size <= 0,
                 'bloom_filter_size must be int and greater than 0')
        raise_if(ValueError, not isinstance(reindex_if_conflict, bool), 'reindex_if_conflict must be bool')

        self.last_commit_time = None
        self._reindex_if_conflict = reindex_if_conflict
        self._REINDEX_FLAG = False
        # set commit flag, if the flag is True, the database will not be saved
        self._COMMIT_FLAG = True

        self.logger = logger

        # set parent path
        self._initialize_parent_path(database_path)

        self._dtypes_map = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}

        # set added samples
        self._added_samples = 0
        # set distance
        self.distance = distance
        # set bloom filter size
        self.bloom_filter_size = bloom_filter_size
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
        # set bloom filter path
        self._bloom_filter_path = self.database_path_parent / 'id_filter'
        # set string dtype
        self._str_dtype = h5py.special_dtype(vlen=str)
        # set open file handler
        self._open_file_handler = None

        # set database path
        self.database_path = self.database_path_parent / Path(database_path).name

        # ============== Loading or create one empty database ==============
        # first of all, initialize a database
        self.reset_database()

        # initialize the database shape ( not the real shape, only for the first time )
        self._added_data_rows = self.shape[0] if self.database_path.exists() else 0

        if self._added_data_rows > 10000:
            self._added_data_rows = 'fulfilled'

        # check initialize params
        self._check_initialize_params(dtypes, reindex_if_conflict)

        self._initialize_ann_model()
        self._initialize_id_filter()
        self._initialize_ivf_index()

        if self._get_cluster_dataset_num() > 0 and self.index_mode == 'FLAT':
            # cause the index mode is FLAT, but the cluster dataset is not empty,
            # so the clustered datasets will also be traversed during querying.
            self.logger.warning('The index mode is FLAT, but the cluster dataset is not empty, '
                                'so the clustered datasets will also be traversed during querying.')

    def _check_initialize_params(self, dtypes, reindex_if_conflict):
        """check initialize params"""

        def _write_params(f):
            f.attrs['dim'] = self.dim
            f.attrs['chunk_size'] = self.chunk_size
            f.attrs['distance'] = self.distance
            f.attrs['dtypes'] = dtypes
            f.attrs['bloom_filter_size'] = self.bloom_filter_size

        if self.database_path.exists():
            self.logger.info('Database file exists, reading parameters...')
            with self._get_h5py_file_handler() as f:
                dim = f.attrs.get('dim', self.dim)
                chunk_size = f.attrs.get('chunk_size', self.chunk_size)
                distance = f.attrs.get('distance', self.distance)
                file_dtypes = f.attrs.get('dtypes', dtypes)
                bloom_filter_size = f.attrs.get('bloom_filter_size', self.bloom_filter_size)
                old_index_mode = f.attrs.get('index_mode', self.index_mode)

                if dim != self.dim:
                    self.logger.warning(
                        f'* dim={dim} in the file is not equal to the dim={self.dim} in the parameters, '
                        f'the parameter dim will be covered by the dim in the file.')
                    self.dim = dim

                if chunk_size != self.chunk_size:
                    self.logger.warning(f'* chunk_size={chunk_size} in the file is not '
                                        f'equal to the chunk_size={self.chunk_size}'
                                        f'in the parameters, the parameter chunk_size will be covered by the chunk_size '
                                        f'in the file.')
                    self.chunk_size = chunk_size

                if distance != self.distance:
                    self.logger.warning(f'* distance=\'{distance}\' in the file is not '
                                        f'equal to the distance=\'{self.distance}\' '
                                        f'in the parameters, the parameter distance will be covered by the distance '
                                        f'in the file.')
                    self.distance = distance

                if file_dtypes != dtypes:
                    self.logger.warning(
                        f'* dtypes=\'{file_dtypes}\' in the file is not equal to the dtypes=\'{dtypes}\' '
                        f'in the parameters, the parameter dtypes will be covered by the dtypes '
                        f'in the file.')
                    self.dtypes = self._dtypes_map[f.attrs.get('dtypes', dtypes)]

                if bloom_filter_size != self.bloom_filter_size:
                    self.logger.warning(f'* bloom_filter_size={bloom_filter_size} in the file is not equal to the '
                                        f'bloom_filter_size={self.bloom_filter_size} in the parameters, the parameter '
                                        f'bloom_filter_size will be covered by the bloom_filter_size in the file.')
                    self.bloom_filter_size = bloom_filter_size

                if old_index_mode != self.index_mode and self.index_mode == 'IVF-FLAT':
                    if reindex_if_conflict:
                        self.logger.warning(
                            f'* index_mode=\'{old_index_mode}\' in the file is not equal to the index_mode='
                            f'\'{self.index_mode}\' in the parameters, if you really want to change the '
                            f'index_mode to \'IVF-FLAT\', you need to run `commit()` function after initializing '
                            f'the database.')

                        # self._CHANGE_INDEX_MODE_FLAG = True
                        self._COMMIT_FLAG = False
                    else:
                        self.logger.warning(
                            f'* index_mode=\'{old_index_mode}\' in the file is not equal to the index_mode='
                            f'\'{self.index_mode}\' in the parameters, if you really want to change the '
                            f'index_mode to \'IVF-FLAT\', you need to set `reindex_if_conflict=True` first '
                            f'and run `commit()` function after initializing the database.')

                _write_params(f)

            self.logger.info('Reading parameters done.')
        else:
            with self._get_h5py_file_handler() as f:
                _write_params(f)

    def _initialize_parent_path(self, database_path):
        """make directory if not exist"""
        self.database_path_parent = Path(database_path).parent.absolute() / Path(
            '.mvdb'.join(Path(database_path).absolute().name.split('.mvdb')[:-1]))

        if not self.database_path_parent.exists():
            self.database_path_parent.mkdir(parents=True)

    def _initialize_ann_model(self):
        """initialize ann model"""
        if self.index_mode == 'IVF-FLAT':
            self.ann_model = KMeans(n_clusters=self.n_clusters, random_state=0,
                                    batch_size=10240, epochs=MVDB_KMEANS_EPOCHS, distance=self.distance)

            if Path(self.database_path_parent / 'ann_model.mvdb').exists() and self.index_mode == 'IVF-FLAT':
                self.ann_model = self.ann_model.load(self.database_path_parent / 'ann_model.mvdb')
                if (self.ann_model.n_clusters != self.n_clusters or self.ann_model.distance != self.distance or
                        self.ann_model.epochs != MVDB_KMEANS_EPOCHS or
                        self.ann_model.batch_size != 10240):
                    self.ann_model = KMeans(n_clusters=self.n_clusters, random_state=0,
                                            batch_size=10240, epochs=MVDB_KMEANS_EPOCHS, distance=self.distance)
        else:
            self.ann_model = None

    def _initialize_ivf_index(self):
        """initialize ivf index"""
        if Path(self.database_path_parent / 'ivf_index.mvdb').exists():
            self.ivf_index = (IVFIndex(n_clusters=self.n_clusters).
                              load(self.database_path_parent / 'ivf_index.mvdb'))
        else:
            self.ivf_index = IVFIndex(n_clusters=self.n_clusters)

    def _initialize_id_filter(self):
        """initialize id filter and shape"""
        self.id_filter = BloomTrie(bloom_size=self.bloom_filter_size, bloom_hash_count=5)

        if self._bloom_filter_path.exists():
            self.id_filter.from_file(self._bloom_filter_path)
        else:
            if self.database_path.exists():
                for database, indices, fields in self.iterable_dataloader(read_chunk_only=False,
                                                                          open_for_only_read=True):
                    self.id_filter.add(tuple(indices))

        self.last_id = self.id_filter.find_max_value()

        # save id filter
        self.id_filter.to_file(self._bloom_filter_path)

    def check_commit(self):
        if not self._COMMIT_FLAG:
            raise ValueError("Did you forget to run `commit()` function ? Try to run `commit()` first.")

    def reset_database(self):
        """Reset the database to its initial state with zeros."""
        self.database = []
        self.indices = []
        self.fields = []

    def _get_h5py_file_handler(self, open_for_only_read=False):
        read_method = 'r' if open_for_only_read else 'a'
        # None or closed
        if self._open_file_handler is None or not self._open_file_handler.id.valid:
            self._open_file_handler = h5py.File(self.database_path, read_method)

        return self._open_file_handler

    def _reset_h5py_file_handler(self):
        if self._open_file_handler is not None:
            self._open_file_handler.close()
            self._open_file_handler = None

    @io_checker
    def iterable_dataloader(self, read_chunk_only=False, from_tail=False, open_for_only_read=False):
        """
        Generator for loading the database and index.

        Parameters:
            read_chunk_only (bool): Whether to read only the chunk.
            from_tail (bool): Whether to read from the end of the file.
            open_for_only_read (bool): Whether to open the file for only reading.

        Yields:
            tuple: A tuple of (database, index, field).

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
            PermissionError: If the file cannot be read due to permission issues.
            UnKnownError: If an unknown error occurs.
        """

        def yield_data_chunking(f, key, msg_grp, index_key, field_key, reverse=False):
            dataset = f[key]
            num_chunks = dataset.chunks[0]
            total_size = dataset.shape[0]
            chunk_ranges = range(0, total_size, num_chunks)

            if reverse:
                chunk_ranges = reversed(chunk_ranges)

            for start in chunk_ranges:
                end = min(start + num_chunks, total_size)

                data_slice = dataset[start:end]
                indices_slice = msg_grp[index_key][start:end]
                fields_slice = msg_grp[field_key][start:end]

                # yield np.array(data_slice), np.array(indices_slice), list(
                #     map(lambda s: s.decode('utf-8'), fields_slice))

                yield data_slice, indices_slice, list(
                    map(lambda s: s.decode('utf-8'), fields_slice))

        with (self._get_h5py_file_handler(open_for_only_read=open_for_only_read) as f):
            if not from_tail:
                if 'chunk' in f and 'chunk_msg' in f:
                    if (self.index_mode == 'FLAT' or read_chunk_only) or \
                            (self.index_mode == 'IVF-FLAT' and self._added_data_rows != 'fulfilled'):
                        yield from yield_data_chunking(f, 'chunk', f['chunk_msg'], 'indices', 'fields')

                if not read_chunk_only:
                    if 'cluster' in f:
                        for j in f['cluster']:
                            yield from yield_data_chunking(f['cluster'], j, f['cluster_msg'], f'{j}_indices',
                                                           f'{j}_fields')
            else:
                if not read_chunk_only and 'cluster' in f:
                    for j in reversed(list(f['cluster'].keys())):
                        yield from yield_data_chunking(f['cluster'], j, f['cluster_msg'], f'{j}_indices', f'{j}_fields',
                                                       reverse=True)

                if 'chunk' in f and 'chunk_msg' in f:
                    if (self.index_mode == 'FLAT' or read_chunk_only) or \
                            (self.index_mode == 'IVF-FLAT' and self._added_data_rows != 'fulfilled'):
                        yield from yield_data_chunking(f, 'chunk', f['chunk_msg'], 'indices', 'fields', reverse=True)

        self._reset_h5py_file_handler()

    def cluster_dataloader(self, cluster_id):
        """
        Generator for loading the database and index. Only used for querying when index_mode is IVF-FLAT.

        Parameters:
            cluster_id (int): The cluster id.

        Yields:
            tuple: A tuple of (database, index, field).

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
            PermissionError: If the file cannot be read due to permission issues.
            UnKnownError: If an unknown error occurs.
        """
        with self._get_h5py_file_handler(open_for_only_read=True) as f:
            datashape = f['cluster'][f'cluster_id_{cluster_id}'].shape[0]
            if datashape // self.chunk_size == 0:
                data = f['cluster'][f'cluster_id_{cluster_id}']
                indices = f['cluster_msg'][f'cluster_id_{cluster_id}_indices']
                fields = f['cluster_msg'][f'cluster_id_{cluster_id}_fields']

                yield data, indices, [i.decode('utf-8') for i in fields]
            else:
                for i in range(datashape // self.chunk_size + 1):
                    data = f['cluster'][f'cluster_id_{cluster_id}'][i * self.chunk_size:(i + 1) * self.chunk_size]
                    indices = f['cluster_msg'][f'cluster_id_{cluster_id}_indices'][
                              i * self.chunk_size:(i + 1) * self.chunk_size]
                    fields = f['cluster_msg'][f'cluster_id_{cluster_id}_fields'][
                             i * self.chunk_size:(i + 1) * self.chunk_size]

                    yield data, indices, [i.decode('utf-8') for i in fields]

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
        if not self._is_database_reset() and not self._COMMIT_FLAG:
            if not (len(self.database) == len(self.indices) == len(self.fields)):
                raise ValueError('The database, index length and field length not the same.')

    def reset_chunk_partition(self):
        """
        Reset the chunk partition.
        """
        f = self._get_h5py_file_handler()
        del f['chunk']
        del f['chunk_msg']

    def save_data(self, data, indices, fields, to_chunk=True, cluster_id=None):
        """Optimized method to save data to chunk or cluster group with reusable logic."""

        def update_dataset(dataset, data):
            """Resize and update dataset with new data."""
            dataset.resize((dataset.shape[0] + len(data), self.dim))
            dataset[-len(data):, :] = np.vstack(data)

        def update_metadata(group, name, data, maxshape, dtype):
            """Create or update metadata dataset."""
            if name in group:
                dataset = group[name]
                dataset.resize((dataset.shape[0] + len(data),))
                dataset[-len(data):] = data
            else:
                group.create_dataset(name, shape=(len(data),), maxshape=maxshape, dtype=dtype, data=data,
                                     chunks=(self.chunk_size,), compression='gzip')

        f = self._get_h5py_file_handler()
        if to_chunk:
            dataset_path = 'chunk'
            metadata_group_path = 'chunk_msg'
        else:
            group_name = 'cluster' if not self._REINDEX_FLAG else 'cluster_new'
            dataset_path = f"{group_name}/cluster_id_{str(cluster_id)}"
            metadata_group_path = "cluster_msg" if not self._REINDEX_FLAG else "cluster_msg_new"

        # Update or create dataset
        if dataset_path in f:
            update_dataset(f[dataset_path], data)
        else:
            f.create_dataset(dataset_path, shape=(len(data), self.dim), maxshape=(None, self.dim), dtype=np.float32,
                             data=np.vstack(data), chunks=(self.chunk_size, self.dim))

        # Ensure the metadata group exists
        metadata_group = f.require_group(metadata_group_path)

        # Update indices and fields metadata
        update_metadata(metadata_group,
                        'indices' if to_chunk else f"cluster_id_{str(cluster_id)}_indices",
                        indices, (None,), np.uint64)

        update_metadata(metadata_group,
                        'fields' if to_chunk else f"cluster_id_{str(cluster_id)}_fields",
                        fields, (None,), self._str_dtype)

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

        cluster_dataset_num = 0
        with self._get_h5py_file_handler() as f:
            if 'cluster' not in f:
                return 0
            return len([i for i in f['cluster'] if isinstance(f['cluster'][i], h5py.Dataset)])

    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        This method is required to be called after saving vectors to query them.
        """
        if not self._COMMIT_FLAG:
            self.logger.info('Saving chunk immediately...')
            self.save_chunk_immediately()

            self.logger.info('Saving id filter...')
            # save bloom filter
            self.id_filter.to_file(self._bloom_filter_path)

            f = self._get_h5py_file_handler()
            if 'chunk' in f:
                chunk_partition_size = f['chunk'].shape[0]
            else:
                chunk_partition_size = 0

            if chunk_partition_size >= 100000 and self.index_mode != 'FLAT':
                self.logger.info('Building index...')
                # build index
                self.build_index()

                # del old cluster files
                if self._REINDEX_FLAG:
                    f = self._get_h5py_file_handler()
                    if 'cluster' in f:
                        del f['cluster']
                        del f['cluster_msg']

                    # rename cluster_new to cluster
                    f.move('cluster_new', 'cluster')
                    f.move('cluster_msg_new', 'cluster_msg')
                    self._REINDEX_FLAG = False

                # save ivf index and k-means model
                self.logger.info('Saving ann model...')
                self.ann_model.save(self.database_path_parent / 'ann_model.mvdb')

                self.logger.info('Saving ivf index...')
                self.ivf_index.save(self.database_path_parent / 'ivf_index.mvdb')

            self.reset_database()

            # save params
            f.attrs['index_mode'] = self.index_mode
            self._reset_h5py_file_handler()

            self._COMMIT_FLAG = True

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

        if self._COMMIT_FLAG:
            self._COMMIT_FLAG = False

        self._added_samples += len(new_ids)

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
        raise_if(ValueError, index in self.id_filter, f'id {index} already exists')
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

        vector, index, field = self._process_vector_item(vector, index, field, normalize)

        # Add the id to then bloom filter.
        self.id_filter.add(index)

        self.database.append(vector)
        self.indices.append(index)
        self.fields.append(field)

        self.save_chunk_immediately() if save_immediately else self.auto_save_chunk()

        if self._COMMIT_FLAG:
            self._COMMIT_FLAG = False

        self._added_samples += 1

        return index

    def delete(self):
        """Delete database."""
        if not self.database_path_parent.exists():
            return None

        try:
            shutil.rmtree(self.database_path_parent)

        except FileNotFoundError:
            pass

        self.id_filter = BloomTrie(bloom_size=self.bloom_filter_size, bloom_hash_count=5)
        self.reset_database()

    def _kmeans_clustering(self, data):
        """kmeans clustering"""
        self.ann_model = self.ann_model.partial_fit(data)

        return self.ann_model.cluster_centers_, self.ann_model.labels_

    def _insert_ivf_index(self, data, indices, fields, reindex=False):
        # Perform K-means clustering.
        self.cluster_centers, labels = self._kmeans_clustering(data)

        current_file_rows = {}
        for i in range(self.n_clusters):
            current_file_rows[i] = 0

        f = self._get_h5py_file_handler()
        if not reindex:
            if 'cluster' in f:
                for j in f['cluster']:
                    current_file_rows[int(j.split('_')[-1])] = f['cluster'][j].shape[0] - 1
        else:
            if 'cluster_new' in f:
                for j in f['cluster_new']:
                    current_file_rows[int(j.split('_')[-1])] = f['cluster_new'][j].shape[0] - 1

        # Current batch's IVF index.
        ivf_index = {i: [] for i in range(self.n_clusters)}
        # Allocate data to various cluster centers.
        for idx, label in enumerate(labels):
            current_file_rows[label] += 1
            try:
                self.ivf_index.add_to_cluster(label, indices[idx], fields[idx], current_file_rows[label])
                ivf_index[label].append([idx, indices[idx], fields[idx]])
            except Exception as e:
                self.logger.error(f"Error in adding to cluster: {e}")
                current_file_rows[label] -= 1
                raise e

        return ivf_index

    def _build_index(self):
        _database = []
        for database, indices, fields in self.iterable_dataloader(read_chunk_only=True):
            # Create an IVF index.
            ivf_index = self._insert_ivf_index(database, indices, fields)
            # reorganize files
            self.reorganize_cluster_partitions(database, ivf_index)

            if len(database) > 100:
                random_length = int(len(database) * 0.1)
            elif 10 < len(database) <= 100:
                random_length = 10
            else:
                random_length = 1

            # Randomly generate a portion of data indices to evaluate the clustering quality.
            random_indices = np.random.choice(len(database), size=random_length, replace=False)

            _database.append(database[random_indices, :])

        return np.vstack(_database)

    def build_index(self):
        """
        Build the IVF index.
        """
        sample_data = self._build_index()

        # delete chunk_id.mvdb files
        self.reset_chunk_partition()

        is_quality_good = self.evaluate_clustering(sample_data)

        if not is_quality_good and self._added_samples >= MVDB_REINDEX_CHECKING_SAMPLES:
            self.logger.info('The clustering quality is not good, reindexing...')
            self._REINDEX_FLAG = True
            self.reindex_if_appropriate()
            if self._added_data_rows != 'fulfilled' and isinstance(self._added_data_rows, int):
                self._added_data_rows += self._added_samples

                if self._added_data_rows > 10000:
                    self._added_data_rows = 'fulfilled'

            self._added_samples = 0  # reset added samples

    def evaluate_clustering(self, data):
        """
        Evaluate the quality of clustering using the silhouette coefficient.
        """
        threshold = MVDB_CLUSTERING_QUALITY_THRESHOLD

        labels = self.ann_model.predict(data)
        score = silhouette_score(data, labels)
        self.logger.info(f"The clustering quality is: {score}")
        return score > threshold

    def reindex_if_appropriate(self):
        """
        If the quality of clustering is poor, re-establish the index.
        """
        self.ann_model = KMeans(n_clusters=self.n_clusters, random_state=0,
                                batch_size=10240, epochs=MVDB_KMEANS_EPOCHS, distance=self.distance)

        self.ivf_index = IVFIndex(n_clusters=self.n_clusters)

        _database = []
        for database, indices, fields in self.iterable_dataloader(read_chunk_only=False):
            # Create an IVF index.
            ivf_index = self._insert_ivf_index(database, indices, fields, reindex=True)
            # reorganize files
            self.reorganize_cluster_partitions(database, ivf_index)

    def reorganize_cluster_partitions(self, database, ivf_index):
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
        self.check_commit()

        dim = self.dim
        total = 0

        if not self.database_path.exists():
            return total, dim

        with self._get_h5py_file_handler() as f:
            if 'chunk' in f:
                total += f['chunk'].shape[0]

            if 'cluster' in f:
                grp = f['cluster']
                for i in grp:
                    total += grp[i].shape[0]

        return total, dim

    @property
    def shape(self):
        self.check_commit()
        return self._get_shape(int(os.path.getmtime(self.database_path) * 1000))
