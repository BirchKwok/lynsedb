import apexbase
from apexbase import ApexStorage
from typing import Dict, List, Any, Union
from pathlib import Path
import json
import pandas as pd


class FieldsStorage:
    """
    Fields storage class using ApexBase as backend storage.
    ApexBase is a high-performance HTAP embedded database with Rust core.
    """
    def __init__(self, filepath=None, cache_size: int = 1000, batch_size: int = 1000):
        """
        Initialize the FieldsStorage class.

        Parameters:
            filepath: str
                The file path to the storage.
            cache_size: int
                Maximum number of query results to cache.
            batch_size: int
                Size of batches for bulk operations.
        """
        if filepath is None:
            raise ValueError("You must provide a file path.")

        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # 配置参数
        self.batch_size = batch_size
        self._field_cache = {}  # 缓存字段信息
        self._schema_cache = {}  # 缓存表结构
        self._field_order = []  # 固定的字段顺序

        # 使用 ApexBase 作为后端存储
        # drop_if_exists=False 保留现有数据
        # durability='fast' 提高写入性能
        self.storage = ApexStorage(str(self.filepath), drop_if_exists=False, durability='fast')

        # 创建默认表
        self._initialize_database()

    def _initialize_database(self):
        """
        Initialize ApexBase database with default table.
        """
        # 创建表（如果不存在）
        tables = self.storage.list_tables()
        if "records" not in tables:
            self.storage.create_table("records")

        # 使用默认表
        self.storage.use_table("records")

        # 初始化字段顺序（如果表中有数据）
        if self.storage.row_count() > 0:
            # 获取一条记录来确定字段顺序
            first_record = self.storage.retrieve(0)
            if first_record:
                self._field_order = list(first_record.keys())

    def _quote_identifier(self, identifier: str) -> str:
        """
        正确转义标识符。

        Parameters:
            identifier: str
                需要转义的标识符

        Returns:
            str: 转义后的标识符
        """
        return f'"{identifier}"'

    def _infer_field_type(self, value: Any) -> str:
        """
        根据值推断字段类型。

        Parameters:
            value: Any
                字段值

        Returns:
            str: ApexBase 字段类型
        """
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "i64"
        elif isinstance(value, float):
            return "f64"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, (list, dict)):
            return "json"  # 复杂类型序列化为JSON字符串
        else:
            return "str"

    def _convert_to_apex_type(self, value: Any) -> Any:
        """
        将 Python 值转换为 ApexBase 兼容的类型。

        Parameters:
            value: Any
                Python 值

        Returns:
            Any: ApexBase 兼容的值
        """
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        elif isinstance(value, bool):
            return int(value)  # ApexBase 使用整数表示布尔值
        else:
            return value

    def store(self, data: dict) -> int:
        """
        Store a record in the storage.

        Parameters:
            data: dict
                The record to be stored.

        Returns:
            int: The ID of the record.
        """
        if not isinstance(data, dict):
            raise ValueError("Only dict-type data is allowed.")

        try:
            # 确保表存在
            self.storage.use_table("records")

            # 过滤掉 _id 字段（如果存在）
            record_data = {k: v for k, v in data.items() if k != '_id'}

            # 更新字段顺序
            self._update_field_order(record_data.keys())

            # 转换数据类型
            record_data = {k: self._convert_to_apex_type(v) for k, v in record_data.items()}

            # 存储记录
            record_id = self.storage.store(record_data)

            # 清除字段缓存
            self._field_cache.clear()

            return record_id

        except Exception as e:
            raise ValueError(f"Failed to store data: {str(e)}")

    def _update_field_order(self, new_keys):
        """更新字段顺序，添加新字段到末尾"""
        seen = set(self._field_order)
        for key in new_keys:
            if key not in seen:
                self._field_order.append(key)
                seen.add(key)

    def batch_store(self, data_list: List[dict]) -> List[int]:
        """
        Optimized batch store with automatic batching.

        Parameters:
            data_list: List[dict]
                List of records to be stored.

        Returns:
            List[int]: List of IDs of the stored records.
        """
        if not data_list:
            return []

        all_ids = []
        try:
            # 确保表存在
            self.storage.use_table("records")

            # 收集所有字段以更新顺序
            all_fields = set()
            for data in data_list:
                all_fields.update(data.keys())

            # 更新字段顺序
            self._update_field_order(all_fields)

            # 批量处理
            for i in range(0, len(data_list), self.batch_size):
                batch = data_list[i:i + self.batch_size]

                # 准备批量数据
                batch_data = []
                for data in batch:
                    # 过滤掉 _id 字段
                    record_data = {k: v for k, v in data.items() if k != '_id'}
                    # 转换数据类型
                    record_data = {k: self._convert_to_apex_type(v) for k, v in record_data.items()}
                    batch_data.append(record_data)

                # 批量存储
                ids = self.storage.store_batch(batch_data)
                all_ids.extend(ids)

            # 清除字段缓存
            self._field_cache.clear()

            return all_ids

        except Exception as e:
            raise ValueError(f"Batch storage failed: {str(e)}")

    def field_exists(self, field: str, use_cache: bool = True) -> bool:
        """
        Check if a field exists.

        Parameters:
            field: str
                The field to check.
            use_cache: bool
                Whether to use cache.

        Returns:
            bool: True if the field exists, False otherwise.
        """
        field = field.strip(':')

        if use_cache:
            if field in self._field_cache:
                return True

        try:
            fields = self.list_fields(use_cache=use_cache)
            exists = field in fields

            if use_cache:
                self._field_cache[field] = exists

            return exists
        except Exception:
            return False

    def list_fields(self, use_cache: bool = True) -> Dict[str, str]:
        """
        List all fields with caching.

        Returns:
            Dict[str, str]: The fields of the storage with their types.
        """
        if use_cache and self._field_cache:
            return self._field_cache

        try:
            self.storage.use_table("records")
            fields = self.storage.list_fields()

            # ApexBase 返回字段名列表，需要获取类型
            fields_with_prefix = {}
            for name in fields:
                dtype = self.storage.get_column_dtype(name) if hasattr(self.storage, 'get_column_dtype') else 'unknown'
                fields_with_prefix[f":{name}:"] = dtype

            if use_cache:
                self._field_cache = fields_with_prefix

            return fields_with_prefix
        except Exception as e:
            raise ValueError(f"Failed to list fields: {str(e)}")

    def execute_query(self, sql: str):
        """
        Execute raw SQL query.

        Parameters:
            sql: str
                SQL query string

        Returns:
            Query results as list of dicts
        """
        result = self.storage.execute(sql)
        columns = result.get('columns', [])
        rows = result.get('rows', [])

        # 转换为字典列表
        return [dict(zip(columns, row)) for row in rows]

    def query(self, where_clause: str = None, limit: int = None) -> List[int]:
        """
        Query records by WHERE clause.

        Parameters:
            where_clause: str
                WHERE clause
            limit: int
                Maximum number of records

        Returns:
            List[int]: List of record IDs
        """
        try:
            self.storage.use_table("records")
            # 使用 execute 来获取 ID
            if where_clause:
                result = self.storage.execute(f"SELECT _id FROM records WHERE {where_clause}")
            else:
                result = self.storage.execute("SELECT _id FROM records")

            rows = result.get('rows', [])
            ids = [row[0] for row in rows]

            if limit:
                ids = ids[:limit]

            return ids
        except Exception as e:
            raise ValueError(f"Query failed: {str(e)}")

    def _get_ordered_fields(self) -> List[str]:
        """
        获取固定的字段顺序。

        Returns:
            List[str]: 字段名列表，按固定顺序排列
        """
        if not self._field_order:
            # 从存储中获取字段列表
            fields = self.storage.list_fields()
            # 按照 list_fields 返回的顺序，保持一致性
            self._field_order = list(fields)
        return self._field_order

    def _order_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        按照固定顺序重新排列字典键。

        Parameters:
            d: Dict[str, Any]

        Returns:
            Dict[str, Any]: 按固定顺序排列的字典
        """
        ordered_fields = self._get_ordered_fields()
        # 先按照固定顺序，再添加新字段（保持插入顺序）
        seen = set(ordered_fields)
        result = {}
        for key in ordered_fields:
            if key in d:
                result[key] = d[key]
        # 添加不在固定列表中的字段
        for key in d.keys():
            if key not in seen:
                result[key] = d[key]
                seen.add(key)
        return result

    def retrieve(self, id_: int) -> Dict[str, Any]:
        """
        Retrieve a single record by ID.

        Parameters:
            id_: int
                The ID of the record

        Returns:
            Dict[str, Any]: The record
        """
        try:
            self.storage.use_table("records")
            result = self.storage.retrieve(id_)
            if result:
                result = self._order_dict(result)
            return result if result else None
        except Exception as e:
            raise ValueError(f"Failed to retrieve record: {str(e)}")

    def retrieve_many(self, ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple records by IDs.

        Parameters:
            ids: List[int]
                List of record IDs

        Returns:
            List[Dict[str, Any]]: List of records
        """
        if not ids:
            return []

        try:
            self.storage.use_table("records")
            results = self.storage.retrieve_many(ids)

            # 必须返回 ResultView 类型
            if not hasattr(results, 'to_pandas'):
                raise ValueError(f"retrieve_many must return ResultView, got {type(results)}")

            # 处理 ResultView 类型
            df = results.to_pandas()
            results = df.to_dict('records')
            # 按照固定顺序排列每个记录的字段
            return [self._order_dict(r) if r else None for r in results]
        except Exception as e:
            raise ValueError(f"Failed to retrieve records: {str(e)}")

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all records.

        Returns:
            List[Dict[str, Any]]: List of all records
        """
        try:
            self.storage.use_table("records")
            results = self.storage.retrieve_all()
            # 按照固定顺序排列每个记录的字段
            return [self._order_dict(r) if r else None for r in results]
            return self.storage.retrieve_all()
        except Exception as e:
            raise ValueError(f"Failed to retrieve all records: {str(e)}")

    def delete(self, id_: int):
        """
        Delete a record by ID.

        Parameters:
            id_: int
                The ID of the record to delete
        """
        try:
            self.storage.use_table("records")
            self.storage.delete(id_)
        except Exception as e:
            raise ValueError(f"Failed to delete record: {str(e)}")

    def delete_where(self, where_clause: str) -> int:
        """
        Delete records matching WHERE clause.

        Parameters:
            where_clause: str
                WHERE clause

        Returns:
            int: Number of deleted records
        """
        try:
            self.storage.use_table("records")
            return self.storage.delete_where(where_clause)
        except Exception as e:
            raise ValueError(f"Failed to delete records: {str(e)}")

    def row_count(self) -> int:
        """
        Get row count.

        Returns:
            int: Number of rows
        """
        try:
            self.storage.use_table("records")
            return self.storage.row_count()
        except Exception as e:
            raise ValueError(f"Failed to get row count: {str(e)}")

    def create_index(self, field_name: str):
        """
        Create index for a field.

        Parameters:
            field_name: str
                Field name to create index on
        """
        # ApexBase 自动处理索引，无需手动创建
        pass

    def optimize(self):
        """
        Optimize database performance.
        """
        try:
            self.storage.flush()
        except Exception as e:
            print(f"Failed to optimize database: {str(e)}")

    def close(self):
        """
        Close storage.
        """
        if hasattr(self, 'storage'):
            self.storage.close()

    def __del__(self):
        """
        Close all connections when the object is deleted.
        """
        self.close()
