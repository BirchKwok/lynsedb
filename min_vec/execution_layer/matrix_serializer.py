from datetime import datetime
from pathlib import Path
import shutil
from typing import Union

import numpy as np
from spinesUtils.asserts import raise_if
from spinesUtils.logging import Logger

from min_vec.structures.id_checker import IDChecker
from min_vec.utils.utils import io_checker
from min_vec.configs.config import config
from min_vec.structures.kmeans import BatchKMeans
from min_vec.structures.scaler import ScalarQuantization
from min_vec.structures.fields_filter import FieldIndex
from min_vec.storage_layer.storage import StorageWorker
from min_vec.execution_layer.cluster_worker import ClusterWorker


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
        self.collections_path = self.collections_path_parent  # / Path(collection_path).name

        # set scalar quantization bits
        self.scaler_bits = scaler_bits if scaler_bits is not None else None
        self.scaler = None
        if self.scaler_bits is not None:
            self._initialize_scalar_quantization()

        # initialize the storage worker
        self.storage_worker = StorageWorker(self.collections_path_parent, self.dim,
                                            self.chunk_size,
                                            quantizer=None if self.scaler_bits is None else self.scaler,
                                            warm_up=warm_up)

        # ============== Loading or create one empty collection ==============
        # first of all, initialize a collection
        self.database = []
        self.indices = []
        self.fields = []

        self._initialize_fields_index()
        self._initialize_ann_model()
        self._initialize_id_checker()

        self.cluster_worker = ClusterWorker(
            logger=self.logger,
            iterable_dataloader=self.dataloader,
            ann_model=self.ann_model,
            storage_worker=self.storage_worker,
            save_data=self.save_data,
            n_clusters=self.n_clusters
        )

        if self._get_cluster_dataset_num() > 0 and self.index_mode == 'FLAT':
            # cause the index mode is FLAT, but the cluster dataset is not empty,
            # so the clustered datasets will also be traversed during querying.
            self.logger.warning('The index mode is FLAT, but the cluster dataset is not empty, '
                                'so the clustered datasets will also be traversed during querying.')

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
            self.fields_index = FieldIndex().load(self.collections_path_parent / 'fields_index.mvdb')
        else:
            self.fields_index = FieldIndex()

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

    def check_commit(self):
        if not self.COMMIT_FLAG:
            raise ValueError("Did you forget to run `commit()` function ? Try to run `commit()` first.")

    def reset_collection(self):
        """Reset the database to its initial state with zeros."""
        self.database = []
        self.indices = []
        self.fields = []

    @io_checker
    def dataloader(self, filename, mode='eager'):
        """
        Generator for loading the database and index.

        Parameters:
            filename (str): The name of the file to load.
            mode (str): The mode of the generator. Options are 'eager' or 'lazy'. Default is 'eager'.

        Yields:
            tuple: A tuple of (database, index, field).

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
            PermissionError: If the file cannot be read due to permission issues.
            UnKnownError: If an unknown error occurs.
        """
        data, indices = self.storage_worker.read(filename=filename)

        if mode == 'lazy':
            return data, indices
        else:
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

    def save_data(self, data, indices, fields, write_chunk=True, cluster_id=None, normalize=False):
        """Optimized method to save data to chunk or cluster group with reusable logic."""
        if fields is not None and cluster_id is None:
            for id, field in zip(indices, fields):
                self.fields_index.store(field, id)

        self.storage_worker.write(data, indices,
                                  write_type='chunk' if write_chunk else 'cluster',
                                  cluster_id=str(cluster_id) if cluster_id is not None else cluster_id,
                                  normalize=normalize)

    @io_checker
    def save_chunk_immediately(self, normalize=True):
        """
        Save the current state of the collection to a .mvdb file.

        Returns:
            Path: The path of the saved collection file.
        """
        self._length_checker()

        if self._is_collection_reset():
            return []

        self.save_data(
            self.database,
            self.indices,
            self.fields,
            write_chunk=True,
            cluster_id=None,
            normalize=normalize
        )

        self.reset_collection()  # reset collection, indices and fields

    def auto_save_chunk(self, normalize=True):
        self._length_checker()
        if len(self.database) >= self.chunk_size:
            self.save_chunk_immediately(normalize=normalize)

        return

    def _get_cluster_dataset_num(self):
        if not self.collections_path.exists():
            return 0

        return self.storage_worker.get_cluster_dataset_num()

    def commit(self):
        """
        Save the collection, ensuring that all data is written to disk.
        This method is required to be called after saving vectors to query them.
        """
        if not self.COMMIT_FLAG:
            self.logger.debug('Saving chunk immediately...')
            self.save_chunk_immediately(normalize=True)

            self.logger.debug('Saving id filter...')
            # save filter
            self.id_filter.to_file(self._filter_path)

            # save fields index
            self.logger.debug('Saving fields index...')
            self.fields_index.save(self.collections_path_parent / 'fields_index.mvdb')

            chunk_partition_size = self.storage_worker.get_shape(read_type='chunk')[0]
            if chunk_partition_size >= 100000 and self.index_mode != 'FLAT':
                self.logger.debug('Building index...')
                # build index
                self.build_index()

                # save ivf index and k-means model
                self.logger.debug('Saving ann model...')
                self.ann_model.save(self.collections_path_parent / 'ann_model.mvdb')

            self.reset_collection()

            if self.scaler_bits is not None:
                self.scaler.save(self.collections_path_parent / 'sq_model.mvdb')

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
        raise_if(ValueError, not all(isinstance(i, tuple) for i in vectors),
                 f'vectors must be tuple of (vector, id[optional], field[optional]).')

        new_ids = []

        for i in range(0, len(vectors), self.chunk_size):
            batch = vectors[i:i + self.chunk_size]

            for sample in batch:
                if len(sample) == 1:
                    vector = sample[0]
                    index = None
                    field = {}
                elif len(sample) == 2:
                    vector, index = sample
                    field = {}
                else:
                    raise_if(ValueError, len(sample) != 3,
                             f'vectors must be tuple of (vector, id[optional], field[optional]).')
                    vector, index, field = sample

                    raise_if(TypeError, not isinstance(field, dict), f'field must be dict, got {type(field)}')

                if isinstance(vector, list):
                    vector = np.array(vector)

                index = self._generate_new_id() if index is None else index

                raise_if(ValueError, index < 0, f'id must be greater than 0, got {index}')

                field = {} if field is None else field
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

    def add_item(self, vector, *, index: int = None, field: dict = None) -> int:
        """
        Add a single vector to the collection.

        Parameters:
            vector (np.ndarray): The vector to be added.
            index (int, optional, keyword-only): The ID of the vector. If None, a new ID will be generated.
            field (dict, optional, keyword-only): The field of the vector. Default is None. If None, the field will be
                set to an empty string.
        Returns:
            int: The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        raise_if(ValueError, len(vector) != self.dim,
                 f'vector dim error, expect {self.dim}, got {len(vector)}')
        if isinstance(vector, list):
            vector = np.array(vector)

        raise_if(ValueError, vector.ndim != 1, f'vector dim error, expect 1, got {vector.ndim}')
        raise_if(ValueError, field is not None and not isinstance(field, dict),
                 f'field must be dict, got {type(field)}')
        raise_if(ValueError, index is not None and not isinstance(index, int), f'id must be int, got {type(index)}')
        raise_if(ValueError, index is not None and index < 0, f'id must be greater than 0, got {index}')

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
        """Delete collection."""
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

    def build_index(self):
        """
        Build the IVF index.
        """
        self.cluster_worker.build_index(self.scaler)

    @property
    def shape(self):
        self.check_commit()
        return tuple(self.storage_worker.get_shape(read_type='all'))
