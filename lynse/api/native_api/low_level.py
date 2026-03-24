import threading
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from ...utils.asserts import raise_if, ParameterTypeAssert

from lynse.execution_layer.query_view import QueryView

from ...configs.parameters_validator import ParametersValidator
from ...utils.utils import unavailable_if_deleted, unavailable_if_empty, collection_repr
from ...api import logger
from ...core_components.fields_cache.filter import Filter
from ..._rust_backend import rust_available, RustEngine
from ...execution_layer.result import Result

import operator as _op
import re as _re


def _filter_to_sql(search_filter):
    """Convert a Filter object or field expression string to SQL WHERE clause for Rust ApexBase."""
    if search_filter is None:
        return None

    if isinstance(search_filter, str):
        # Field expression string like ":name: == 'John Doe' and :age: > 30"
        sql = search_filter
        # :id: maps to external_id in Rust ApexBase
        sql = sql.replace(':id:', 'external_id')
        sql = _re.sub(r':(\w+):', r'\1', sql)
        sql = sql.replace(' == ', ' = ')
        # Convert double-quoted strings to single-quoted for SQL
        sql = _re.sub(r'"([^"]*)"', r"'\1'", sql)
        sql = _re.sub(r'\bnot\s+in\s*\[([^\]]*)\]', r'NOT IN (\1)', sql, flags=_re.IGNORECASE)
        sql = _re.sub(r'\bin\s*\[([^\]]*)\]', r'IN (\1)', sql)
        sql = _re.sub(r'\band\b', 'AND', sql)
        sql = _re.sub(r'\bor\b', 'OR', sql)
        return sql

    if isinstance(search_filter, Filter):
        parts = []
        for cond in search_filter.must_fields:
            parts.append(_condition_to_sql(cond))
        if search_filter.any_fields:
            or_parts = [_condition_to_sql(c) for c in search_filter.any_fields]
            parts.append('(' + ' OR '.join(or_parts) + ')')
        for cond in search_filter.must_not_fields:
            parts.append('NOT (' + _condition_to_sql(cond) + ')')
        return ' AND '.join(parts) if parts else None

    return None


_OP_MAP = {
    _op.eq: '=', _op.ne: '!=', _op.gt: '>', _op.lt: '<',
    _op.ge: '>=', _op.le: '<=',
}


def _condition_to_sql(cond):
    """Convert a single FieldCondition to SQL clause."""
    from ...core_components.fields_cache.filter import MatchID, MatchField, MatchRange

    m = cond.matcher
    if isinstance(m, MatchID):
        ids_str = ', '.join(str(i) for i in m.indices)
        return f'external_id IN ({ids_str})'
    elif isinstance(m, MatchRange):
        key = cond.key
        if m.inclusive is True:
            return f'{key} >= {_sql_val(m.start)} AND {key} <= {_sql_val(m.end)}'
        elif m.inclusive == 'left':
            return f'{key} >= {_sql_val(m.start)} AND {key} < {_sql_val(m.end)}'
        elif m.inclusive == 'right':
            return f'{key} > {_sql_val(m.start)} AND {key} <= {_sql_val(m.end)}'
        else:
            return f'{key} > {_sql_val(m.start)} AND {key} < {_sql_val(m.end)}'
    elif isinstance(m, MatchField):
        key = cond.key
        val = m.value
        if isinstance(val, (list, tuple)):
            vals = ', '.join(_sql_val(v) for v in val)
            if m.not_in:
                return f'{key} NOT IN ({vals})'
            return f'{key} IN ({vals})'
        op = _OP_MAP.get(m.comparator, '=')
        return f'{key} {op} {_sql_val(val)}'
    return '1=1'


def _sql_val(v):
    """Format a Python value for SQL."""
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


class ExclusiveDB:
    """
    A class for managing a vector database stored in chunk files and computing vectors similarity.
    The class is exclusive and cannot be shared with other threads or processes,
    so it is not thread-safe or process-safe.

    Note:
        This class is not called at the top level, but is called through the LocalClient class.
        When directly operating on data by calling this class, users need to be clear about what they are doing.
    """
    name = "Local"

    @ParametersValidator(
        immutable_param=['dim', 'chunk_size'],
        logger=logger
    )
    @ParameterTypeAssert({
        'dim': int,
        'database_path': str,
        'chunk_size': int,
        'dtypes': (None, str),
        'cache_query': bool,
        'n_threads': (None, int),
        'warm_up': bool,
        'cache_chunks': int
    }, func_name='ExclusiveDB')
    def __init__(
            self,
            dim: int,
            database_path: Union[str, Path],
            chunk_size: int = 100_000,
            dtypes: str = 'float32',
            cache_query: bool = True,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            cache_chunks: int = 20
    ):
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str or Path): The path to the database file.
            chunk_size (int): The size of each data chunk. Default is 100,000.
                It's recommended to be between 10,000 and 500,000.
            cache_query (bool): Whether to use cache for search. Default is True.
            n_threads (int): The number of threads to use for parallel processing. Default is 10.
            warm_up (bool): Whether to warm up the database. Default is False.
            cache_chunks (int): The number of chunks to cache in memory. Default is 20.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        raise_if(ValueError, chunk_size <= 1, 'chunk_size must greater than 1')

        if chunk_size <= 1:
            raise ValueError('chunk_size must be greater than 1')

        if not 1_0000 <= chunk_size <= 100_0000:
            logger.warning('The recommended chunk size is between 10,000 and 1,000,000.')

        # In order to have enough data to index the segment
        raise_if(ValueError, chunk_size < 1000, 'chunk_size must greater than or equal to 1000.')

        self._database_path = str(database_path)
        self._database_name = Path(database_path).parent.name
        self._collection_name = Path(database_path).name
        self._dim = dim
        self._chunk_size = chunk_size
        self._IS_DELETED = False

        # ── Rust backend (thread safety handled by parking_lot::RwLock) ──
        if not rust_available():
            raise RuntimeError(
                "Rust backend (lynse_core) is required but not installed. "
                "Build with: cd rust/lynse-core && maturin develop --release"
            )

        root_path = str(Path(database_path).parent)
        self._engine = RustEngine(root_path)

        if self._engine.has_collection(self._collection_name):
            self._rust_coll = self._engine.get_collection(
                self._collection_name, dim, chunk_size
            )
        else:
            self._rust_coll = self._engine.create_collection(
                self._collection_name, dim, chunk_size
            )

        # ── Write buffer for add_item (single vector buffering) ──
        self._buffer_lock = threading.Lock()
        self._buffer_vectors: List[np.ndarray] = []
        self._buffer_fields: List[dict] = []
        self._buffer_size_limit = chunk_size

        # pre_build_index if the ExclusiveDB instance is reloaded
        if self.shape[0] > 0:
            self._pre_build_index()

    # ── Single-vector add with buffering ─────────────────────────────────────

    @unavailable_if_deleted
    def add_item(self, vector: Union[np.ndarray, list], id: int = None, *,
                 field: dict = None,
                 buffer_size: Union[int, None, bool] = True) -> int:
        """
        Add a single vector to the collection.

        It is recommended to use incremental ids for best performance.

        Parameters:
            vector (np.ndarray): The vector to be added.
            id (int, optional): The ID of the vector. Default is None (auto-assigned).
            field (dict, optional, keyword-only): The field of the vector. Default is None.
                If None, the field will be set to an empty string.
            buffer_size (int or bool or None): The buffer size for the storage worker. Default is True.

                - If None, the vector will be directly written to the disk.
                - If True, the buffer_size will be set to chunk_size,
                    and the vectors will be written to the disk when the buffer is full.
                - If False, the vector will be directly written to the disk.
                - If int, when the buffer is full, the vectors will be written to the disk.

        Returns:
            (int): The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        vector = np.asarray(vector, dtype=np.float32).ravel()
        raise_if(ValueError, vector.shape[0] != self._dim,
                 'vector dimension mismatch.')

        # Determine effective buffer limit
        if buffer_size is None or buffer_size is False:
            limit = 0
        elif buffer_size is True:
            limit = self._buffer_size_limit
        else:
            limit = int(buffer_size)

        # Current ID = total existing + buffered
        current_id = self.shape[0] + len(self._buffer_vectors)

        with self._buffer_lock:
            self._buffer_vectors.append(vector)
            self._buffer_fields.append(field if field else {})

            if limit == 0 or len(self._buffer_vectors) >= limit:
                self._flush_buffer()

        return current_id

    @unavailable_if_deleted
    def bulk_add_items(
            self, vectors: Union[List[Tuple[np.ndarray, int, dict]], List[Tuple[np.ndarray, int]]],
            **kwargs
    ) -> List[int]:
        """
        Bulk add vectors to the collection in batches.


        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved.
                Each vector can be a tuple of (vector, field).

        Returns:
            List[int]: A list of indices where the vectors are stored.
        """
        # Flush any pending buffer first
        with self._buffer_lock:
            self._flush_buffer()

        vecs = []
        fields = []
        for item in vectors:
            if not isinstance(item, (list, tuple)):
                # bare vector
                vecs.append(np.asarray(item, dtype=np.float32).ravel())
                fields.append({})
            elif len(item) >= 2 and isinstance(item[-1], dict):
                # (vector, field) or (vector, id, field)
                vecs.append(np.asarray(item[0], dtype=np.float32).ravel())
                fields.append(item[-1])
            elif len(item) >= 1:
                vecs.append(np.asarray(item[0], dtype=np.float32).ravel())
                fields.append({})

        start_id = self.shape[0]
        np_vecs = np.vstack(vecs).astype(np.float32)
        field_list = fields if any(f for f in fields) else None
        self._rust_coll.add_items(np_vecs, field_list)

        return list(range(start_id, start_id + len(vecs)))

    def _flush_buffer(self):
        """Flush the internal write buffer to the Rust backend.
        
        Note: caller must hold self._buffer_lock when calling from add_item.
        For commit() we acquire the lock ourselves.
        """
        if not self._buffer_vectors:
            return
        np_vecs = np.vstack(self._buffer_vectors).astype(np.float32)
        field_list = self._buffer_fields if any(f for f in self._buffer_fields) else None
        self._rust_coll.add_items(np_vecs, field_list)
        self._buffer_vectors.clear()
        self._buffer_fields.clear()

    def from_pandas(self, df: pd.DataFrame):
        """
        Add vectors from a pandas DataFrame.
        """
        self._flush_buffer()
        # Expect 'vector' column and remaining columns as fields
        if 'vector' not in df.columns:
            raise ValueError("DataFrame must contain a 'vector' column.")
        vectors = np.vstack(df['vector'].values).astype(np.float32)
        field_cols = [c for c in df.columns if c != 'vector']
        fields = None
        if field_cols:
            fields = df[field_cols].to_dict('records')
        self._rust_coll.add_items(vectors, fields)

    def from_arrow(self, table: pa.Table):
        """
        Add vectors from a pyarrow Table.
        """
        self.from_pandas(table.to_pandas())

    def from_polars(self, df: pl.DataFrame):
        """
        Add vectors from a polars DataFrame.
        """
        self.from_pandas(df.to_pandas())

    def from_parquet(self, path: str):
        """
        Add vectors from a parquet file.
        """
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        self.from_arrow(table)

    def from_csv(self, path: str):
        """
        Add vectors from a csv file.
        """
        df = pd.read_csv(path)
        self.from_pandas(df)

    def from_dict(self, data: dict):
        """
        Add vectors from a dictionary.
        """
        self.from_pandas(pd.DataFrame(data))

    @unavailable_if_deleted
    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        """
        with self._buffer_lock:
            self._flush_buffer()
        self._rust_coll.commit()
        self._pre_build_index()

    @unavailable_if_deleted
    def _pre_build_index(self):
        if self._rust_coll.index_mode is None and self.shape[0] > 0:
            logger.info("Pre-building the index...")
            self.build_index(index_mode="FLAT")

    @unavailable_if_deleted
    @unavailable_if_empty
    def build_index(self, index_mode, **kwargs):
        """
        Build the index for clustering.

        Parameters:
            index_mode (str): The index mode, must be one of the following:

                - 'FLAT': Flat index with inner product. (Default)
                - 'FLAT-L2': Flat index with squared L2 distance.
                - 'FLAT-COS': Flat index with cosine similarity.
                - 'FLAT-IP-SQ8': Flat index with inner product and scalar quantizer with 8 bits.
                - 'FLAT-L2-SQ8': Flat index with squared L2 distance and scalar quantizer with 8 bits.
                - 'FLAT-COS-SQ8': Flat index with cosine similarity and scalar quantizer with 8 bits.
                - 'FLAT-JACCARD-BINARY': Flat index with binary quantizer. The distance is Jaccard distance.
                - 'FLAT-HAMMING-BINARY': Flat index with binary quantizer. The distance is Hamming distance.
                - 'IVF': IVF index with inner product.
                - 'IVF-L2': IVF index with squared L2 distance.
                - 'IVF-COS': IVF index with cosine similarity.
                - 'IVF-IP-SQ8': IVF index with inner product and scalar quantizer with 8 bits.
                - 'IVF-L2-SQ8': IVF index with squared L2 distance and scalar quantizer with 8 bits.
                - 'IVF-COS-SQ8': IVF index with cosine similarity and scalar quantizer with 8 bits.
                - 'IVF-JACCARD-BINARY': IVF index with binary quantizer. The distance is Jaccard distance.
                - 'IVF-HAMMING-BINARY': IVF index with binary quantizer. The distance is Hamming distance.
            kwargs: Additional keyword arguments. The following are available:

                - 'n_clusters' (int): The number of clusters. It is only available when the index_mode including 'IVF'.

        Returns:
            None
        """
        # Normalize index mode: "FLAT" → "Flat-IP", etc.
        mode = index_mode.strip()
        alias_map = {
            'FLAT': 'Flat-IP',
            'FLAT-L2': 'Flat-L2',
            'FLAT-COS': 'Flat-Cos',
            'FLAT-IP-SQ8': 'Flat-IP-SQ8',
            'FLAT-L2-SQ8': 'Flat-L2-SQ8',
            'FLAT-COS-SQ8': 'Flat-Cos-SQ8',
            'FLAT-JACCARD-BINARY': 'Flat-Jaccard-Binary',
            'FLAT-HAMMING-BINARY': 'Flat-Hamming-Binary',
            'IVF': 'IVF-IP',
            'IVF-L2': 'IVF-L2',
            'IVF-COS': 'IVF-Cos',
            'IVF-IP-SQ8': 'IVF-IP-SQ8',
            'IVF-L2-SQ8': 'IVF-L2-SQ8',
            'IVF-COS-SQ8': 'IVF-Cos-SQ8',
            'IVF-JACCARD-BINARY': 'IVF-Jaccard-Binary',
            'IVF-HAMMING-BINARY': 'IVF-Hamming-Binary',
        }
        rust_mode = alias_map.get(mode.upper(), mode)
        self._rust_coll.build_index(rust_mode)

    @unavailable_if_deleted
    @unavailable_if_empty
    def remove_index(self):
        """
        Remove the vector index.

        Returns:
            None
        """
        self._rust_coll.remove_index()
        logger.info("Fallback to default index mode: `Flat-IP`.")
        self.build_index(index_mode='Flat-IP')  # Default index mode

    @unavailable_if_deleted
    @unavailable_if_empty
    def search(self, vector: Union[np.ndarray, list], k: int = 10, *,
               search_filter: Filter = None, **kwargs):
        """
        Search the database for the vectors most similar to the given vector.

        Parameters:
            vector (np.ndarray or list): The search vectors, it can be a single vector or a list of vectors.
                The vectors must have the same dimension as the vectors in the database,
                and the type of vector can be a list or a numpy array.
            k (int): The number of nearest vectors to return.
            search_filter (Filter or FilterExpression string, optional): The filter to apply to the search.
            kwargs: Additional keyword arguments. The following are valid:

                - rescore (bool): Whether to rescore the results of binary or scaler quantization searches.
                    Default is False. It is recommended to set it to True when the index mode is 'Binary'.
                - rescore_multiplier (int): The multiplier for the rescore operation.
                    It is only available when rescore is True.
                    If 'Binary' is in the index mode, the default is 10. Otherwise, the default is 2.
                - nprobe (int): The number of clusters to search. It is only available when the index mode is 'IVF'.
                    Default is 8.

        Returns:
            Result: 搜索结果对象，可通过 ``indices, distances, fields = res`` 解包，
            或使用 ``res.to_*`` 系列方法进行格式化。

        Raises:
            ValueError: If the database is empty.
        """
        raise_if(ValueError, not isinstance(vector, (np.ndarray, list)),
                 'vector must be np.ndarray or list.')

        raise_if(TypeError, not isinstance(k, int) and not (isinstance(k, str) and k != 'all'),
                 'k must be int or "all".')
        raise_if(ValueError, k <= 0, 'k must be greater than 0.')
        raise_if(ValueError, not isinstance(search_filter, (Filter, type(None), str)),
                 'search_filter must be Filter or None or FieldExpression string.')

        # Convert the vector to np.ndarray
        vector = np.atleast_2d(np.asarray(vector, dtype=np.float32))

        raise_if(ValueError, vector.shape[1] != self._dim,
                 'vector must be same dim with database.')

        rescore = kwargs.setdefault('rescore', False)
        rescore_multiplier = kwargs.setdefault('rescore_multiplier', 2)
        raise_if(TypeError, not isinstance(rescore, bool), 'rescore must be bool.')
        raise_if(TypeError, not isinstance(rescore_multiplier, int), 'rescore_multiplier must be int.')

        n_total = self.shape[0]
        if n_total == 0:
            raise ValueError('database is empty.')

        if k > n_total:
            k = n_total

        nprobe = kwargs.get('nprobe', 8)

        # Convert search_filter to SQL WHERE clause for Rust ApexBase
        filter_str = _filter_to_sql(search_filter)

        # Single vs batch search
        if vector.shape[0] == 1:
            rust_result = self._rust_coll.search(
                vector[0], k=k, search_filter=filter_str, nprobe=nprobe
            )
        else:
            rust_result = self._rust_coll.batch_search(
                vector, k=k, search_filter=filter_str, nprobe=nprobe
            )
            # For batch, return list of Result objects with lazy field loading
            return [
                Result(r.ids, r.distances,
                       field_loader=lambda _r=r: _r.fields)
                for r in rust_result
            ]

        # Wrap Rust result with lazy field loading (fields fetched on .to_df() etc.)
        return Result(rust_result.ids, rust_result.distances,
                      field_loader=lambda: rust_result.fields)

    @unavailable_if_deleted
    @unavailable_if_empty
    def query(self, query_filter):
        """
        Query the fields cache.

        Parameters:
            query_filter (str): Filter or dict or FieldExpression string
                The filter object or the specify data to filter.

        Returns:
            (QueryView): The records.
        """
        filter_str = _filter_to_sql(query_filter)

        ids = self._rust_coll.query_fields(filter_str)
        fields = self._rust_coll.retrieve_fields(ids)
        return QueryView((None, fields))

    @unavailable_if_deleted
    @unavailable_if_empty
    def query_vectors(self, query_filter):
        """
        Query the vector data by the filter.

        Parameters:
            query_filter (Filter or dict or FieldExpression str or None):
                The filter object or the specify data to filter.
        Returns:
            (QueryView): The vectors, IDs, and fields of the items.
        """
        return self.query(query_filter)

    @property
    def shape(self):
        """
        Return the shape of the entire database.

        Returns:
            (Tuple[int, int]): The number of vectors and the dimension of each vector in the database.
        """
        return self._rust_coll.shape

    @unavailable_if_deleted
    def insert_session(self):
        """
        Create a session to insert data, which will automatically commit the data when the session ends.

        Returns:
            DataOpsSession (DataOpsSession): The session object.
        """
        from ...execution_layer.session import DataInsertionSession

        return DataInsertionSession(self)

    @unavailable_if_deleted
    @unavailable_if_empty
    def head(self, n=5):
        """
        Return the first n vectors in the database.

        Parameters:
            n (int): The number of vectors to return. Default is 5.

        Returns:
            (QueryView): The vectors, IDs, and fields of the items.
        """
        vectors, fields = self._rust_coll.head(n)
        return QueryView((vectors, fields))

    @unavailable_if_deleted
    @unavailable_if_empty
    def tail(self, n=5):
        """
        Return the last n vectors in the database.

        Parameters:
            n (int): The number of vectors to return. Default is 5.

        Returns:
            (QueryView): The vectors, IDs, and fields of the items.
        """
        vectors, fields = self._rust_coll.tail(n)
        return QueryView((vectors, fields))

    @unavailable_if_deleted
    @unavailable_if_empty
    def list_fields(self):
        """
        Return all field names and their types.

        Returns:
            Dict: A dictionary with field names as keys and field types as values.
        """
        field_names = self._rust_coll.list_fields()
        return {name: 'unknown' for name in field_names}

    @unavailable_if_deleted
    @unavailable_if_empty
    def remove_all_field_indices(self):
        """
        Remove all the field indices.

        Returns:
            None
        """
        # Field indices are managed by ApexBase in the Rust backend;
        # this is a no-op as the Rust field store handles indexing internally.
        pass

    def delete(self):
        """
        Delete the database.

        This API is not part of the Collection in the HTTP API, but rather part of the HTTPClient in the HTTP API
        Users must call this API through the HTTPClient when they want to delete a collection on the server.

        Returns:
            None
        """
        if self._IS_DELETED:
            return

        import gc

        self._rust_coll.delete()
        self._IS_DELETED = True

        gc.collect()

    @property
    @unavailable_if_empty
    def index_mode(self):
        """
        Return the index mode of the database.

        Returns:
            (str or None): The index mode of the database.
        """
        return self._rust_coll.index_mode

    def is_deleted(self):
        """
        To check if the database is deleted.

        Returns:
            Bool: True if the database is deleted, False otherwise.
        """
        return self._IS_DELETED

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return collection_repr(self)

    def __str__(self):
        return self.__repr__()
