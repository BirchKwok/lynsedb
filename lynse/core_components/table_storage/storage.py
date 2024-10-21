import os

import numpy as np
import pandas as pd
import pyarrow as pa
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
import shutil
import threading
import duckdb
import concurrent.futures
from functools import lru_cache
import hashlib
import msgpack

from ..bitset import BitSet
from .query import Query


class SearchIterator:
    """
    An iterator for searching data in a LyStorage instance.

    Usage:
        storage = LyStorage("path/to/storage")
        iterator = storage.search()
        for row in iterator:
            print(row)
    """
    def __init__(self, storage, columns=None):
        """
        Initialize the SearchIterator.

        Parameters:
            storage (LyStorage): The LyStorage instance to search.
            columns (list or str): The columns to search. If a string, it will be converted to a list.
        """
        self._storage = storage
        if isinstance(columns, str):
            self._columns = [columns]
        else:
            self._columns = columns

        self._where_clause = None
        self._limit_value = None
        self._order_by_clause = None
        self._query_executed = False
        self._result = None

    def where(self, condition: str):
        """
        Add a WHERE clause to the search query.

        Parameters:
            condition (str): The condition to filter the data.

        Returns:
            self: The SearchIterator instance.
        """
        self._where_clause = condition
        return self

    def limit(self, n: int):
        """
        Add a LIMIT clause to the search query.

        Parameters:
            n (int): The number of rows to limit the result to.

        Returns:
            self: The SearchIterator instance.
        """
        self._limit_value = n
        return self

    def orderby(self, column: str, desc: bool = False):
        """
        Add an ORDER BY clause to the search query.

        Parameters:
            column (str): The column to order by.
            desc (bool): Whether to order in descending order.

        Returns:
            self: The SearchIterator instance.
        """
        self._order_by_clause = f"{column} {'DESC' if desc else 'ASC'}"
        return self

    def _execute_query(self):
        """
        Execute the search query.
        """
        if self._query_executed:
            return

        selected_columns = ', '.join(self._columns) if self._columns else '*'
        query = f"SELECT {selected_columns} FROM storage"
        if self._where_clause:
            query += f" WHERE {self._where_clause}"
        if self._order_by_clause:
            query += f" ORDER BY {self._order_by_clause}"
        if self._limit_value:
            query += f" LIMIT {self._limit_value}"

        self._result = self._storage.cached_execute_sql(query)
        self._query_executed = True

    def __iter__(self):
        """
        Iterate over the results of the search query.
        """
        self._execute_query()
        return iter(self._result.itertuples(index=False))

    def to_list(self):
        """
        Convert the results of the search query to a list of dictionaries.

        Returns:
            list: The list of dictionaries.
        """
        self._execute_query()
        return self._result.to_dict('records')

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the results of the search query to a pandas DataFrame.

        Returns:
            pd.DataFrame: The pandas DataFrame.
        """
        self._execute_query()
        return self._result

    def to_arrow(self) -> pa.Table:
        """
        Convert the results of the search query to a pyarrow Table.

        Returns:
            pa.Table: The pyarrow Table.
        """
        self._execute_query()
        return pa.Table.from_pandas(self._result)


class Query:
    def __init__(self, storage, query_vector: List[float]):
        """
        初始化 Query 对象。

        参数:
            storage (LyStorage): LyStorage 实例。
            query_vector (list or tuple): 查询向量。
        """
        self.storage = storage
        self.query_vector = np.array(query_vector, dtype=np.float64)
        self.filters = []
        self.limit_value = None

    def where(self, condition: str):
        """
        添加过滤条件。

        参数:
            condition (str): 过滤条件的表达式，例如 "key1 LIKE 'jp%'"

        返回:
            self
        """
        self.filters.append(condition)
        return self

    def limit(self, n: int):
        """
        设置结果限制。

        参数:
            n (int): 结果限制数量。

        返回:
            self
        """
        self.limit_value = n
        return self

    def _format_query_vector(self) -> str:
        """
        将查询向量格式化为 DuckDB 的 DOUBLE[] 类型的 ARRAY 格式字符串。

        返回:
            str: 格式化后的查询向量字符串。
        """
        return f"ARRAY[{', '.join(map(str, self.query_vector))}]::DOUBLE[128]"

    def to_pandas(self) -> pd.DataFrame:
        """
        执行查询并返回结果为 pandas DataFrame。

        返回:
            pandas.DataFrame: 查询结果。
        """
        # 创建 DuckDB 连接
        conn = duckdb.connect(database=':memory:')

        try:
            # 注册 LyStorage 的表
            arrow_table = self.storage.to_arrow()
            conn.register('storage_table', arrow_table)

            # 格式化查询向量
            query_vector_str = self._format_query_vector()

            # 构建过滤条件
            where_clause = ""
            if self.filters:
                where_clause = "WHERE " + " AND ".join(self.filters) + " "

            # 构建 SQL 查询
            # 使用 array_cosine_similarity 计算余弦相似度
            sql_query = f"""
                SELECT
                    *,
                    array_cosine_similarity(vector::DOUBLE[128], {query_vector_str}) AS Distances
                FROM
                    storage_table
                {where_clause}
                ORDER BY
                    Distances DESC
                {(f"LIMIT {self.limit_value}" if self.limit_value is not None else "")}
            """

            # 执行查询并获取结果
            result_df = conn.execute(sql_query).fetchdf()

            return result_df

        except duckdb.BinderException as e:
            raise ValueError(f"查询过程中出现 BinderException 错误: {e}")
        except duckdb.CatalogException as e:
            raise ValueError(f"查询过程中出现 CatalogException 错误: {e}")
        except Exception as e:
            raise ValueError(f"查询过程中出现错误: {e}")
        finally:
            conn.close()


class LyStorage:
    """
    A storage class for managing data in a directory.

    Usage:
        storage = LyStorage("path/to/data")
        storage.bulk_add([{"name": "John", "age": 30}])
        storage.delete_row(0)
        storage.restore_row(0)
    """
    def __init__(self, data_path: str, lazy_load: bool = True, overwrite=False, merge_threshold: int = 1000, max_rows_per_file: int = 100_000, max_rows_per_region: int = 1_000_000):
        """
        Initialize the LyStorage instance.

        Parameters:
            data_path (str): The path to the data directory.
            lazy_load (bool): Whether to lazy load the data.
            overwrite (bool): Whether to overwrite the existing data.
            merge_threshold (int): The threshold to merge the data.
            max_rows_per_file (int): The maximum number of rows per data file.
            max_rows_per_region (int): The maximum number of rows per region.
        """
        self.data_path = Path(data_path)
        self.lazy_load = lazy_load
        self.overwrite = overwrite
        self.merge_threshold = merge_threshold
        self.max_rows_per_file = max_rows_per_file
        self.max_rows_per_region = max_rows_per_region

        if overwrite:
            self.delete()

        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

        self._lifecycle_path = self.data_path / f"__lifecycle"
        if not self._lifecycle_path.exists():
            self._lifecycle_path.mkdir(parents=True, exist_ok=True)

        self.region_meta_file = self.data_path / "regions_meta.lymp"
        if not self.region_meta_file.exists() or overwrite:
            self.regions_meta = {
                'version_counter': 0,
                'regions': []
            }
            self.regions = []
            self.version_counter = 0
            self._save_regions_meta()
        else:
            self.regions_meta = msgpack.load(open(self.region_meta_file, 'rb'))
            self.regions = self.regions_meta.get('regions', [])
            self.version_counter = self.regions_meta.get('version_counter', 0)

        self.current_region = self.regions[-1] if self.regions else None
        if not self.current_region or self.current_region['row_count'] >= self.max_rows_per_region:
            self._create_new_region()

        self.table = None  # Initialize as None
        self._duckdb_conn = duckdb.connect(database=str(self.data_path / "__virtual.ddb"), read_only=False)
        self._register_table()

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()

        if not lazy_load:
            self._load_all_regions()

    def _create_new_region(self):
        """
        Create a new region.
        """
        new_region = {
            "region_id": len(self.regions) + 1,
            "start_row": self.current_region['start_row'] + self.current_region['row_count'] if self.current_region else 0,
            "row_count": 0,
            "data_files": [],
            "versions": [],
            "bitset": BitSet()
        }
        self.regions.append(new_region)
        self.current_region = new_region
        self._save_regions_meta()

    def _save_regions_meta(self):
        """
        Save the regions metadata without including the BitSet objects.
        """
        # Create a serializable copy of regions without 'bitset'
        serializable_regions = []
        for region in self.regions:
            # Copy all fields except 'bitset'
            serializable_region = {k: v for k, v in region.items() if k != 'bitset'}
            serializable_regions.append(serializable_region)

        # Update the regions_meta dictionary
        self.regions_meta['regions'] = serializable_regions
        self.regions_meta['version_counter'] = self.version_counter

        # Serialize the regions_meta to the meta file
        with open(self.region_meta_file, 'wb') as f:
            msgpack.dump(self.regions_meta, f)

    def _load_all_regions(self):
        """
        Load all regions' data and BitSets.
        """
        tables = []
        for region in self.regions:
            if region['row_count'] == 0 or not region['data_files']:
                continue

            # 确保 region['bitset'] 已经在 rollback 或其他方法中正确设置
            if region['bitset'] is None:
                # 如果仍然没有 bitset，初始化一个全 1 的 BitSet
                region['bitset'] = BitSet(size=region['row_count'], fill=1)

            for data_file in region['data_files']:
                full_path = self.data_path / data_file
                try:
                    with pa.memory_map(full_path.as_posix(), 'r') as source:
                        table = pa.ipc.open_file(source).read_all()

                        # 应用 BitSet 过滤活跃行
                        bitset = region['bitset']
                        row_indices = list(range(len(table)))
                        active_indices = [i for i in row_indices if bitset.get_bit(i)]

                        if active_indices:
                            active_table = table.take(active_indices)
                            tables.append(active_table)
                except pa.lib.ArrowInvalid as e:
                    print(f"Error: 无法读取数据文件 {data_file}。详情: {e}")

        if tables:
            try:
                self.table = pa.concat_tables(tables)
            except pa.lib.ArrowInvalid as e:
                print(f"Error: 无法合并表。详情: {e}")
                self.table = None
        else:
            self.table = None

        self._register_table()

    def _register_table(self):
        """
        Register the table to the DuckDB connection.
        """
        try:
            self._duckdb_conn.unregister("storage")
        except duckdb.InvalidInputException:
            pass

        if self.table is not None and len(self.table.schema.names) > 0:
            self._duckdb_conn.register("storage", self.table)

    def _get_data_file(self, region_id: int, timestamp: int, segment: int) -> str:
        """
        Get the data file path based on region, timestamp, and segment.

        Parameters:
            region_id (int): The ID of the region.
            timestamp (int): The timestamp of the version.
            segment (int): The segment number of the file.

        Returns:
            str: The path to the data file.
        """
        file_hash = hashlib.md5(f"{region_id}_{timestamp}_{segment}".encode()).hexdigest()
        return os.path.join(self.data_path, f"region_{region_id}_{file_hash}.ly")

    def bulk_add(self, data: Union[List[Dict[str, Any]], pd.DataFrame, pa.Table], description: Optional[str] = None):
        """
        Bulk add multiple rows of data to the storage.

        Parameters:
            data (Union[List[Dict[str, Any]], pd.DataFrame, pa.Table]): The data to add.
                Can be a list of dictionaries, pandas DataFrame, or pyarrow Table.
            description (str): The description of the data.
        """
        if data is None or (isinstance(data, (list, pd.DataFrame, pa.Table)) and len(data) == 0):
            return

        with self.lock:
            self._load_all_regions()

            if isinstance(data, list):
                try:
                    new_data = {k: [d[k] for d in data] for k in data[0].keys()}
                except KeyError as e:
                    print(f"数据添加失败，缺少列: {e}")
                    return
                df = pd.DataFrame(new_data)
                new_table = pa.Table.from_pandas(df)
            elif isinstance(data, pd.DataFrame):
                data = data.copy()
                new_table = pa.Table.from_pandas(data)
            elif isinstance(data, pa.Table):
                new_table = data
            else:
                raise ValueError("不支持的数据类型。请提供字典列表、pandas DataFrame或pyarrow Table。")

            # 更新 BitSet 和区域行数，并添加数据到区域
            new_rows = len(new_table)
            rows_to_add = new_rows
            regions_modified = set()

            while rows_to_add > 0:
                remaining_space = self.max_rows_per_region - self.current_region['row_count']
                if remaining_space <= 0:
                    self._create_new_region()
                    remaining_space = self.max_rows_per_region

                rows_in_this_iteration = min(remaining_space, rows_to_add)
                start_index = new_rows - rows_to_add
                data_slice = new_table.slice(start_index, rows_in_this_iteration)

                # Update BitSet
                self.current_region['bitset'].resize(self.current_region['row_count'] + rows_in_this_iteration, fill=1)

                # Add data to current region
                self._add_to_current_region(data_slice)

                # Remove this line to prevent double increment
                # self.current_region['row_count'] += rows_in_this_iteration

                rows_to_add -= rows_in_this_iteration
                regions_modified.add(self.current_region['region_id'])

            # 更新版本信息
            for region_id in regions_modified:
                region = next(r for r in self.regions if r['region_id'] == region_id)
                self.version_counter += 1
                self._update_version(description or "Bulk add data", region=region)

    def _add_to_current_region(self, table: pa.Table):
        region = self.current_region
        timestamp = int(time.time())
        segment = len(region['data_files'])
        data_file = self._get_data_file(region['region_id'], timestamp, segment)

        with pa.OSFile(data_file, 'wb') as sink:
            writer = pa.ipc.new_file(sink, table.schema)
            writer.write_table(table)
            writer.close()
        region['data_files'].append(os.path.basename(data_file))
        region['row_count'] += len(table)  # This is the correct place to update row_count

        self._save_regions_meta()

    def delete_row(self, row_index: int, description: Optional[str] = None):
        """
        Delete a specific row.

        Parameters:
            row_index (int): The index of the row to delete.
            description (str): Description of the deletion.
        """
        with self.lock:
            region = self._find_region_by_row(row_index)
            if not region:
                print(f"Row {row_index} 不存在。")
                return
            local_index = row_index - region['start_row']
            try:
                region['bitset'].clear_bit(local_index)
                # Update version with global version_counter
                self.version_counter += 1
                self._update_version(description or f"Delete row {row_index}", region=region)  # 传递 region 参数
                print(f"成功删除行 {row_index}。")
            except IndexError as e:
                print(f"删除行失败: {e}")

    def restore_row(self, row_index: int, description: Optional[str] = None):
        """
        Restore a specific row.

        Parameters:
            row_index (int): The index of the row to restore.
            description (str): Description of the restoration.
        """
        with self.lock:
            region = self._find_region_by_row(row_index)
            if not region:
                print(f"Row {row_index} 不存在。")
                return
            local_index = row_index - region['start_row']
            try:
                region['bitset'].set_bit(local_index)
                # Update version with global version_counter
                self.version_counter += 1
                self._update_version(description or f"Restore row {row_index}", region=region)  # 传递 region 参数
                print(f"成功恢复行 {row_index}。")
            except IndexError as e:
                print(f"恢复行失败: {e}")

    def bulk_delete_rows(self, row_indices: List[int], description: Optional[str] = None):
        """
        Delete multiple rows.

        Parameters:
            row_indices (list): The indices of the rows to delete.
            description (str): The description of the deletion.
        """
        with self.lock:
            regions_modified = set()
            for row_index in row_indices:
                region = self._find_region_by_row(row_index)
                if region:
                    local_index = row_index - region['start_row']
                    try:
                        region['bitset'].clear_bit(local_index)
                        regions_modified.add(region['region_id'])
                    except IndexError as e:
                        print(f"删除行 {row_index} 失败: {e}")
                else:
                    print(f"Row {row_index} 不存在。")
            # 更新版本信息
            for region_id in regions_modified:
                region = next(r for r in self.regions if r['region_id'] == region_id)
                self.version_counter += 1
                self._update_version(description or "Bulk delete rows", region=region)
            self._update_table()
            print(f"成功批量删除 {len(row_indices)} 行。")

    def bulk_restore_rows(self, row_indices: List[int], description: Optional[str] = None):
        """
        Restore multiple rows.

        Parameters:
            row_indices (list): The indices of the rows to restore.
            description (str): The description of the restoration.
        """
        with self.lock:
            regions_modified = set()
            for row_index in row_indices:
                region = self._find_region_by_row(row_index)
                if region:
                    local_index = row_index - region['start_row']
                    try:
                        region['bitset'].set_bit(local_index)
                        regions_modified.add(region['region_id'])
                    except IndexError as e:
                        print(f"恢复行 {row_index} 失败: {e}")
                else:
                    print(f"Row {row_index} 不存在。")
            # 更新版本信息
            for region_id in regions_modified:
                region = next(r for r in self.regions if r['region_id'] == region_id)
                self.version_counter += 1
                self._update_version(description or "Bulk restore rows", region=region)
            self._update_table()
            print(f"成功批量恢复 {len(row_indices)} 行。")

    def _find_region_by_row(self, row_index: int) -> Optional[Dict[str, Any]]:
        """
        Find the region containing the given row index.

        Parameters:
            row_index (int): The global row index.

        Returns:
            Dict[str, Any] | None: The region information or None.
        """
        for region in self.regions:
            if region['start_row'] <= row_index < region['start_row'] + region['row_count']:
                return region
        return None

    def _update_version(self, description: Optional[str] = None, region: Optional[Dict[str, Any]] = None):
        if region is None:
            region = self.current_region

        timestamp = int(time.time())
        bitset_filename = f"region_{region['region_id']}_version_{self.version_counter}.bits"
        bitset_file = (
                self._lifecycle_path.relative_to(self.data_path) / bitset_filename
        ).as_posix()
        new_version = {
            "version": self.version_counter,
            "modify_time": timestamp,
            "description": description or "Data updated",
            "region_id": region['region_id'],
            "bitset_file": bitset_file
        }
        region['versions'].append(new_version)
        bitset_path = self._lifecycle_path / bitset_filename

        region['bitset'].save_to_file(bitset_path)
        self._save_regions_meta()
        print(f"版本已更新: {new_version['version']} - {new_version['description']}")

    def rollback(self, target_version: int):
        """
        回滚到指定版本，但不删除超过目标版本的历史版本。

        参数：
            target_version (int): 要回滚到的目标版本号。
        """
        with self.lock:
            # 重新从 regions_meta 读取所有区域信息
            self.regions_meta = msgpack.load(open(self.region_meta_file, 'rb'))
            serializable_regions = self.regions_meta.get('regions', [])

            # 重建 regions 列表，并初始化 bitset
            self.regions = []
            current_start_row = 0

            for serializable_region in serializable_regions:
                region = serializable_region.copy()
                region['bitset'] = None  # 初始化为 None，稍后设置
                self.regions.append(region)

            for region in self.regions:
                # 找到该区域中所有小于等于 target_version 的版本
                valid_versions = [v for v in region['versions'] if v['version'] <= target_version]
                if valid_versions:
                    # 加载对应版本的 BitSet
                    target_version_info = valid_versions[-1]
                    bitset_file = target_version_info.get('bitset_file')
                    if bitset_file and (self.data_path / bitset_file).exists():
                        bitset_path = self.data_path / bitset_file
                        region['bitset'] = BitSet.load_from_file(bitset_path)
                    else:
                        # 如果没有 bitset 文件，初始化一个全 1 的 BitSet
                        region['bitset'] = BitSet(size=region['row_count'], fill=1)
                    # 更新该区域的版本列表，只保留小于等于 target_version 的版本
                    region['versions'] = valid_versions
                    # 更新区域的 start_row
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']
                    print(f"区域 {region['region_id']} 回滚到版本 {target_version_info['version']}。")
                else:
                    # 如果该区域没有有效版本，设置 BitSet 为全 0，表示所有行被删除
                    region['bitset'] = BitSet(size=region['row_count'], fill=0)
                    region['versions'] = []
                    # 更新区域的 start_row
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']
                    print(
                        f"区域 {region['region_id']} 没有小于等于目标版本 {target_version} 的版本，将所有行标记为已删除。")

            # 更新 version_counter
            self.version_counter = target_version

            # 不要调用 _save_regions_meta()，以免覆盖元数据文件

            # 重新加载所有区域的数据
            self._load_all_regions()
            print(f"成功回滚到版本 {target_version}。")

    def restore_to_latest(self):
        """
        恢复到最新版本。
        """
        with self.lock:
            # 重新从 regions_meta 读取所有区域信息
            self.regions_meta = msgpack.load(open(self.region_meta_file, 'rb'))
            serializable_regions = self.regions_meta.get('regions', [])

            # 重建 regions 列表，并初始化 bitset
            self.regions = []
            current_start_row = 0

            for serializable_region in serializable_regions:
                region = serializable_region.copy()
                region['bitset'] = None  # 初始化为 None，稍后设置
                self.regions.append(region)

            if not self.regions:
                print("没有版本可恢复。")
                return

            # 找到所有区域中最高的版本号
            latest_version = max(
                [v['version'] for region in self.regions for v in region['versions']], default=0
            )

            # 遍历所有区域，恢复到各自的最新版本
            for region in self.regions:
                if region['versions']:
                    latest_version_info = region['versions'][-1]
                    bitset_file = latest_version_info.get('bitset_file')
                    if bitset_file and (self.data_path / bitset_file).exists():
                        bitset_path = self.data_path / bitset_file
                        region['bitset'] = BitSet.load_from_file(bitset_path)
                    else:
                        region['bitset'] = BitSet(size=region['row_count'], fill=1)
                    # 保持原始的 row_count，不要修改
                    # 更新区域的 start_row
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']
                else:
                    region['bitset'] = BitSet(size=region['row_count'], fill=1)
                    # 更新区域的 start_row
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']

            self.version_counter = latest_version
            self._load_all_regions()
            print("成功恢复到最新版本。")

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all versions across all regions.

        Returns:
            list: The list of versions.
        """
        with self.lock:
            all_versions = []
            for region in self.regions:
                for version in region['versions']:
                    all_versions.append(version)
            # Sort versions by version number
            all_versions.sort(key=lambda x: x['version'])
            return all_versions

    def count_rows(self) -> int:
        """
        Count the number of active rows in the storage.

        Returns:
            int: The number of active rows.
        """
        with self.lock:
            active_count = 0
            for region in self.regions:
                region_active_count = region['bitset'].count()
                print(f"区域 {region['region_id']} 的活跃行数：{region_active_count}")
                active_count += region_active_count
            print(f"总活跃行数: {active_count}")
            return active_count

    def _update_table(self):
        """
        Update the DuckDB table registration.
        """
        self._register_table()

    def query(self, query_vector: list) -> Query:
        """
        创建一个 Query 对象，用于执行基于余弦相似度的查询。

        参数:
            query_vector (list or tuple): 查询向量。

        返回:
            Query: Query 对象。
        """
        return Query(self, query_vector)

    @lru_cache(maxsize=128)
    def cached_execute_sql(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.

        Parameters:
            query (str): The SQL query to execute.

        Returns:
            pandas.DataFrame: The result of the SQL query.
        """
        return self.execute_sql(query)

    def execute_sql(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.

        Parameters:
            query (str): The SQL query to execute.

        Returns:
            pandas.DataFrame: The result of the SQL query.
        """
        try:
            # Use the thread pool to execute the query in parallel
            future = self._executor.submit(self._duckdb_conn.execute, query)
            result = future.result().fetchdf()
            return result
        except duckdb.BinderException as e:
            print(f"SQL 执行失败，错误: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"SQL 执行失败，错误: {e}")
            return pd.DataFrame()

    def search(self, columns: Optional[List[str]] = None) -> SearchIterator:
        """
        Search the data from the storage.

        Parameters:
            columns (list): The columns to search.

        Returns:
            SearchIterator: The search iterator.
        """
        return SearchIterator(self, columns)

    def to_arrow(self, include_vector: bool = True, columns: Optional[List[str]] = None) -> pa.Table:
        """
        Convert the storage to a pyarrow Table.

        Parameters:
            include_vector (bool): Whether to include the vector column.
            columns (list): The columns to include.

        Returns:
            pyarrow.Table: The pyarrow Table.
        """
        with self.lock:
            if self.table is None:
                self._load_all_regions()

            table = self.table
            if columns is not None:
                if include_vector and 'vector' not in columns:
                    columns = ['vector'] + columns
                table = table.select(columns)
            else:
                table = table

            return table

    def to_pandas(self, include_vector: bool = True, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert the storage to a pandas DataFrame.

        Parameters:
            include_vector (bool): Whether to include the vector column.
            columns (list): The columns to include.

        Returns:
            pandas.DataFrame: The pandas DataFrame.
        """
        with self.lock:
            table = self.to_arrow(include_vector, columns)
            df = table.to_pandas()
            return df

    def delete(self):
        """
        Delete the storage.
        """
        if self.data_path.exists():
            shutil.rmtree(self.data_path)

    def _execute_query(self, query: str):
        """
        Execute a query by filtering data across all regions.

        Parameters:
            query (str): The condition part of the SQL WHERE clause.

        Returns:
            pd.DataFrame: The query result.
        """
        futures = []
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for region in self.regions:
                for data_file in region['data_files']:
                    full_path = self.data_path / data_file
                    sql_query = f"SELECT * FROM 'storage' WHERE {query}"
                    futures.append(executor.submit(self.cached_execute_sql, sql_query))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if not result.empty:
                    results.append(result)

        if results:
            concatenated = pd.concat(results, ignore_index=True)
            return concatenated
        return pd.DataFrame()
