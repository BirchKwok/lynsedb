from pathlib import Path
import shutil
import time
from typing import Union

import numpy as np
from filelock import FileLock
import pandas as pd
import polars as pl
import pyarrow as pa

from spinesUtils.asserts import raise_if
import logging
Logger = logging.Logger
from tqdm import tqdm

from ..core_components.fields_cache import FieldsCache
from ..storage_layer.storage import PersistentFileStorage
from ..storage_layer.wal import WALStorage
from ..utils.utils import inject_caller_info


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
                Options are 'float16', 'float32' or 'float64'.
            warm_up (bool): Whether to warm up the collection. Default is False.
            cache_chunks (int): The buffer size for the storage worker. Default is 20.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        # set commit flag, if the flag is True, the collection will not be saved
        self.COMMIT_FLAG = True
        self.IS_DELETED = False

        self.logger = logger

        # set parent path
        self._initialize_components_path(collection_path)

        self.global_lock = FileLock(self.collections_path_parent / "global.lock", thread_local=True)

        # set dtypes
        self.dtypes = np.float32
        # set dim
        self.dim = dim
        # set chunk size
        self.chunk_size = chunk_size

        # initialize the storage worker
        self.storage_worker = PersistentFileStorage(self.collections_path_parent, self.dim,
                                                    self.chunk_size,
                                                    warm_up=warm_up, cache_chunks=cache_chunks)
        self.wal_worker = WALStorage(collection_name=self.collections_path_parent.name,
                                     chunk_size=self.chunk_size,
                                     storage_path=self.collections_path_parent,
                                     flush_interval=1)

        self._initialize_fields_index()

        # 检查是否存在未提交的数据
        if self.wal_worker.has_uncommitted_data():
            self.logger.info("Detected uncommitted data, preparing to recover...")
            # replay wal
            with self.global_lock:
                self.commit_data(recover=True)

    def _initialize_components_path(self, collections_path):
        """make directory if not exist"""
        self.collections_path_parent = (Path(collections_path).parent.absolute() /
                                        Path(collections_path).absolute().name)

        self.collections_path_parent.mkdir(parents=True, exist_ok=True)

        # field_index path
        self.field_index_path = self.collections_path_parent / 'fields_index.db'

    def _initialize_fields_index(self):
        """initialize fields index"""
        self.field_index = FieldsCache(self.field_index_path)

    def dataloader(self, filename):
        """
        Generator for loading the database and index.

        Parameters:
            filename (str): The name of the file to load.

        Yields:
            tuple: A tuple of database.

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
        # Only print progress bar if the logger level is less than 20, which means INFO or DEBUG
        # Determine logger level numerically for tqdm disabling
        _lvl = self.logger.level
        if not isinstance(_lvl, (int, float)):
            _lvl = {
                'DEBUG': 10,
                'INFO': 20,
                'WARNING': 30,
                'ERROR': 40,
                'CRITICAL': 50
            }.get(str(_lvl).upper(), 20)

        for data, fields in tqdm(
                self.wal_worker.get_file_iterator(),
                desc="Data persisting",
                total=self.wal_worker.chunk_number,
                unit='chunk',
                disable=_lvl > 20):
            self.storage_worker.write(data)
            # store fields index
            self.field_index.batch_store(fields)

            # insert data to indexer
            if hasattr(self, "indexer"):
                self.indexer.index_insert(data)

        if hasattr(self, "indexer"):
            if self.indexer.ivf is not None:
                self.indexer.update_ivf_index()
            # update filenames
            self.indexer.update_filenames()

        self.logger.info(end_msg)
        self.wal_worker.reincarnate()

    @inject_caller_info
    def commit(self):
        """
        Save the collection, ensuring that all data is written to disk.
        """
        if not self.COMMIT_FLAG:
            self.logger.info('Saving data...')
            if hasattr(self, 'buffer'):
                if len(self.buffer) > 0:
                    self.bulk_add_items(self.buffer)

            self.commit_data()

            # remove buffer
            self._remove_buffer()

            self.COMMIT_FLAG = True

    def _define_buffer(self, buffer_size: int):
        self.buffer = []
        self.buffer_size = buffer_size

    def _remove_buffer(self):
        if hasattr(self, 'buffer'):
            del self.buffer
        if hasattr(self, 'buffer_size'):
            del self.buffer_size

    def _insert_buffer(self, vector, field: dict, buffer_size: int):
        if not hasattr(self, 'buffer'):
            self._define_buffer(buffer_size)
        elif getattr(self, 'buffer_size') != buffer_size:
            self._remove_buffer()
            self._define_buffer(buffer_size)

        if len(self.buffer) < self.buffer_size:
            self.buffer.append((vector, field))
        else:
            self.bulk_add_items(self.buffer)
            self.buffer = []
            self.buffer.append((vector, field))

    def _process_vector_item(self, vector, field):
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if vector.shape[1] != self.dim:
            raise ValueError(f'vector dim error, expect {self.dim}, got {vector.shape[1]}')

        if vector.dtype != self.dtypes:
            vector = vector.astype(self.dtypes)

        return vector, field if field is not None else {}

    @inject_caller_info
    def bulk_add_items(self, vectors, **kwargs):
        """
        Bulk add vectors to the collection in batches.

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved.
                Each vector can be a tuple of (vector, field).
                And the field is optional, and type is dict. The default value is an empty dict.

        Returns:
            list: A list of indices where the vectors are stored.
        """
        raise_if(ValueError, not isinstance(vectors, (tuple, list)),
                 f'vectors must be tuple or list, got {type(vectors)}')

        data = []
        fields = []

        for i in range(0, len(vectors), self.chunk_size):
            batch = vectors[i:i + self.chunk_size]

            for sample in batch:
                sample_len = len(sample)

                if sample_len == 2:
                    vector, field = sample
                elif sample_len == 1:
                    vector = sample[0]
                    field = {}
                else:
                    raise ValueError('Each sample must be a tuple of (vector, field[optional]).')

                vector, field = self._check_input_format(vector, field)

                vector, field = self._process_vector_item(vector, field)

                data.append(vector)
                fields.append(field)

        self.wal_worker.write_log_data(data, fields)
        self.COMMIT_FLAG = False

    def _check_input_format(self, vector, field):
        if isinstance(vector, list):
            vector = np.array(vector)

        raise_if(ValueError, vector.shape not in [(1, self.dim), (self.dim,)],
                 f'vector dim error, expect (1, {self.dim}) or ({self.dim},), got {vector.shape}')
        raise_if(ValueError, field is not None and not isinstance(field, dict),
                 f'field must be dict, got {type(field)}')

        return vector, field

    @inject_caller_info
    def add_item(self, vector, field: dict = None,
                 buffer_size: Union[None, int, bool] = True, **kwargs):
        """
        Add a single vector to the collection.


        Parameters:
            vector (np.ndarray): The vector to be added.
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
        vector, field = self._check_input_format(vector, field)

        vector, field = self._process_vector_item(vector, field)

        if buffer_size is not None and buffer_size is not False:
            if buffer_size is True:
                buffer_size = self.chunk_size

            self._insert_buffer(vector, field, buffer_size)
        else:
            self.wal_worker.write_log_data(
                vector,
                [field]
            )

        self.COMMIT_FLAG = False

    def from_pandas(self, df: pd.DataFrame):
        """
        Add vectors from a pandas DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing vectors. Must have a 'vectors' column.

        Raises:
            ValueError: If 'vectors' column is not present in the DataFrame.
        """
        if 'vectors' not in df.columns:
            raise ValueError("DataFrame must contain a 'vectors' column")

        vectors = df['vectors'].values
        fields = df.drop('vectors', axis=1).to_dict('records')
        self.bulk_add_items(list(zip(vectors, fields)))

    def from_csv(self, filepath: str):
        """
        Add vectors from a csv file using Polars for better performance.

        Parameters:
            filepath (str): Path to the CSV file. Must contain a 'vectors' column.

        Raises:
            ValueError: If 'vectors' column is not present in the CSV file.
        """
        # 使用 Polars 读取 CSV，性能更好
        df = pl.read_csv(filepath)
        if 'vectors' not in df.columns:
            raise ValueError("CSV file must contain a 'vectors' column")

        # 将向量字符串转换为numpy数组
        vectors = df.select('vectors').to_numpy()
        vectors = np.array([np.fromstring(v[0].strip('[]'), sep=',', dtype=self.dtypes) for v in vectors])

        # 获取其他字段
        fields = df.drop('vectors').to_dicts()

        self.bulk_add_items(list(zip(vectors, fields)))

    def from_parquet(self, filepath: str):
        """
        Add vectors from a parquet file using Polars for better performance.

        Parameters:
            filepath (str): Path to the parquet file. Must contain a 'vectors' column.

        Raises:
            ValueError: If 'vectors' column is not present in the parquet file.
        """
        # 使用 Polars 读取 Parquet，性能更好
        df = pl.read_parquet(filepath)
        if 'vectors' not in df.columns:
            raise ValueError("Parquet file must contain a 'vectors' column")

        vectors = df.select('vectors').to_numpy().reshape(-1)
        fields = df.drop('vectors').to_dicts()

        self.bulk_add_items(list(zip(vectors, fields)))

    def from_arrow(self, table: pa.Table):
        """
        Add vectors from an Arrow table directly without pandas conversion.

        Parameters:
            table (pa.Table): Arrow table containing vectors. Must have a 'vectors' column.

        Raises:
            ValueError: If 'vectors' column is not present in the table.
        """
        if 'vectors' not in table.column_names:
            raise ValueError("Arrow table must contain a 'vectors' column")

        # 直接从Arrow获取向量数据
        vectors = table.column('vectors').to_numpy()

        # 获取其他字段
        fields_table = table.drop(['vectors'])
        fields = [dict(zip(fields_table.column_names, row))
                 for row in zip(*[fields_table[col].to_numpy() for col in fields_table.column_names])]

        self.bulk_add_items(list(zip(vectors, fields)))

    def from_polars(self, df: pl.DataFrame):
        """
        Add vectors from a Polars DataFrame directly without pandas conversion.

        Parameters:
            df (pl.DataFrame): Polars DataFrame containing vectors. Must have a 'vectors' column.

        Raises:
            ValueError: If 'vectors' column is not present in the DataFrame.
        """
        if 'vectors' not in df.columns:
            raise ValueError("Polars DataFrame must contain a 'vectors' column")

        # 直接从Polars获取向量数据
        vectors = df.select('vectors').to_numpy().reshape(-1)

        # 获取其他字段
        fields = df.drop('vectors').to_dicts()

        self.bulk_add_items(list(zip(vectors, fields)))

    def from_dict(self, data: dict):
        """
        Add vectors from a dictionary.
        """
        vectors = data['vectors']
        fields = data['fields']
        self.bulk_add_items(list(zip(vectors, fields)))

    def delete(self):
        """Delete collection."""
        with self.global_lock:
            if not self.collections_path_parent.exists():
                return None

            # stop wal
            self.wal_worker.stop()

            # close indexer
            if hasattr(self, "indexer"):
                self.indexer.close_mapped_index()

            retries = 3
            delay = 1

            for attempt in range(retries):
                try:
                    shutil.rmtree(self.collections_path_parent)
                    break
                except PermissionError as e:
                    if attempt < retries - 1:
                        time.sleep(delay)
                    else:
                        raise e
                except Exception as e:
                    self.logger.error(f"Error deleting: {e}")

            self.IS_DELETED = True

            self._initialize_fields_index()

            # clear cache
            self.storage_worker.clear_cache()

    @property
    def shape(self):
        """
        Get the shape of the collection.
        """
        return tuple(self.storage_worker.get_shape(read_type='all'))
