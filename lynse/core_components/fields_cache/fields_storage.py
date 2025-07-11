import sqlite3
import orjson
from typing import Dict, List, Any
from pathlib import Path
from ...core_components.limited_dict import LimitedDict


class FieldsStorage:
    """
    Fields storage class using SQLite as backend storage.
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
        self._query_cache = LimitedDict(cache_size)
        self._field_cache = LimitedDict(100)  # 缓存字段信息

        # SQLite连接
        self.conn = sqlite3.connect(str(self.filepath),
                                  isolation_level=None)  # 自动提交模式，手动控制事务

        # 优化SQLite配置
        cursor = self.conn.cursor()
        # 使用WAL模式提高写入性能
        cursor.execute("PRAGMA journal_mode=WAL")
        # 设置较大的WAL文件大小限制（64MB）
        cursor.execute("PRAGMA wal_autocheckpoint=1000")
        # 降低同步级别提高性能
        cursor.execute("PRAGMA synchronous=NORMAL")
        # 设置较大的页缓存（约256MB）
        cursor.execute("PRAGMA cache_size=-262144")
        # 临时表和临时文件使用内存
        cursor.execute("PRAGMA temp_store=MEMORY")
        # 启用内存映射，提高读取性能
        cursor.execute("PRAGMA mmap_size=1099511627776")  # 1TB
        # 设置较大的页大小，提高读写效率
        cursor.execute("PRAGMA page_size=32768")  # 32KB
        # 启用严格的外键约束
        cursor.execute("PRAGMA foreign_keys=ON")
        # 读写锁定超时设置（5分钟）
        cursor.execute("PRAGMA busy_timeout=300000")

        self._initialize_database()

    def _initialize_database(self):
        """
        Initialize SQLite database with optimized settings.
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("BEGIN TRANSACTION")

            # 删除旧表（如果存在）
            cursor.execute("DROP TABLE IF EXISTS records")
            cursor.execute("DROP TABLE IF EXISTS fields_meta")

            # 创建主表，只包含ID
            cursor.execute("""
                CREATE TABLE records (
                    _id INTEGER PRIMARY KEY AUTOINCREMENT
                )
            """)

            # 创建字段元数据表
            cursor.execute("""
                CREATE TABLE fields_meta (
                    field_name TEXT PRIMARY KEY,
                    field_type TEXT NOT NULL
                )
            """)

            # 重置序列从0开始
            cursor.execute("INSERT INTO records (_id) VALUES (0)")
            cursor.execute("DELETE FROM records WHERE _id = 0")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'records'")

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e

    def _quote_identifier(self, identifier: str) -> str:
        """
        正确转义 SQLite 标识符。

        Parameters:
            identifier: str
                需要转义的标识符

        Returns:
            str: 转义后的标识符
        """
        return f'"{identifier}"'

    def _ensure_field_exists(self, field_name: str, field_type: str):
        """
        确保字段存在，如不存在则创建。

        Parameters:
            field_name: str
                字段名称
            field_type: str
                字段类型 (TEXT, INTEGER, REAL, etc.)
        """
        try:
            # 检查字段是否已存在
            result = self.conn.execute(
                "SELECT field_type FROM fields_meta WHERE field_name = ?",
                [field_name]
            ).fetchone()

            if not result:
                # 添加字段到主表，使用引号包裹字段名
                quoted_field_name = self._quote_identifier(field_name)
                self.conn.execute(f"ALTER TABLE records ADD COLUMN {quoted_field_name} {field_type}")
                # 记录字段元数据
                self.conn.execute(
                    "INSERT INTO fields_meta (field_name, field_type) VALUES (?, ?)",
                    [field_name, field_type]
                )
        except Exception as e:
            raise ValueError(f"Failed to ensure field exists: {str(e)}")

    def _infer_field_type(self, value: Any) -> str:
        """
        根据值推断字段类型。

        Parameters:
            value: Any
                字段值

        Returns:
            str: SQLite字段类型
        """
        if isinstance(value, bool):
            return "INTEGER"  # SQLite没有布尔类型，使用INTEGER
        elif isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, (list, dict)):
            return "TEXT"  # 复杂类型序列化为JSON字符串
        else:
            return "TEXT"

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
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE TRANSACTION")

            try:
                # 首先确保所有字段存在
                for field_name, value in data.items():
                    if field_name != '_id':
                        field_type = self._infer_field_type(value)
                        self._ensure_field_exists(field_name, field_type)

                # 创建记录获取ID
                if 'id' in data:
                    # 如果提供了id，直接使用它作为_id
                    cursor.execute("INSERT INTO records (_id) VALUES (?)", (data['id'],))
                    record_id = data['id']
                else:
                    cursor.execute("INSERT INTO records DEFAULT VALUES")
                    record_id = cursor.lastrowid

                # 处理每个字段
                field_updates = []
                params = []

                for field_name, value in data.items():
                    if field_name != '_id':
                        quoted_field_name = self._quote_identifier(field_name)
                        field_updates.append(f"{quoted_field_name} = ?")
                        # 如果是复杂类型，序列化为JSON
                        if isinstance(value, (list, dict)):
                            import json
                            params.append(json.dumps(value))
                        else:
                            params.append(value)

                # 如果有字段需要更新
                if field_updates:
                    update_sql = f"UPDATE records SET {', '.join(field_updates)} WHERE _id = ?"
                    params.append(record_id)
                    cursor.execute(update_sql, params)

                cursor.execute("COMMIT")
                self._invalidate_cache()
                return record_id

            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e

        except Exception as e:
            raise ValueError(f"Failed to store data: {str(e)}")

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
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE TRANSACTION")

            try:
                # 首先收集所有唯一字段并确保它们存在
                all_fields = set()
                for data in data_list:
                    for field_name, value in data.items():
                        if field_name != '_id':
                            all_fields.add((field_name, self._infer_field_type(value)))

                # 批量创建所有需要的字段
                for field_name, field_type in all_fields:
                    self._ensure_field_exists(field_name, field_type)

                # 然后批量插入数据
                for i in range(0, len(data_list), self.batch_size):
                    batch = data_list[i:i + self.batch_size]

                    for data in batch:
                        # 创建记录获取ID
                        if 'id' in data:
                            # 如果提供了id，直接使用它作为_id
                            cursor.execute("INSERT INTO records (_id) VALUES (?)", (data['id'],))
                            record_id = data['id']
                        else:
                            cursor.execute("INSERT INTO records DEFAULT VALUES")
                            record_id = cursor.lastrowid

                        all_ids.append(record_id)

                        # 处理每个字段
                        field_updates = []
                        params = []

                        for field_name, value in data.items():
                            if field_name != '_id':
                                quoted_field_name = self._quote_identifier(field_name)
                                field_updates.append(f"{quoted_field_name} = ?")
                                # 如果是复杂类型，序列化为JSON
                                if isinstance(value, (list, dict)):
                                    import json
                                    params.append(json.dumps(value))
                                else:
                                    params.append(value)

                        # 如果有字段需要更新
                        if field_updates:
                            update_sql = f"UPDATE records SET {', '.join(field_updates)} WHERE _id = ?"
                            params.append(record_id)
                            cursor.execute(update_sql, params)

                cursor.execute("COMMIT")
                self._invalidate_cache()
                return all_ids

            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e

        except Exception as e:
            raise ValueError(f"Batch storage failed: {str(e)}")

    def _parse_record(self, row: tuple) -> Dict[str, Any]:
        """
        Parse a database record into a dictionary.

        Parameters:
            row: tuple
                Database record containing all fields

        Returns:
            Dict[str, Any]: Parsed record
        """
        # 获取当前表的所有列名
        cursor = self.conn.cursor()
        columns = [description[0] for description in cursor.execute("SELECT * FROM records WHERE 1=0").description]

        data = {}
        for i, value in enumerate(row):
            if value is not None:
                column_name = columns[i]
                if column_name != '_id':  # 跳过内部ID字段
                    # 尝试解析JSON字符串
                    if isinstance(value, str):
                        try:
                            import json
                            parsed_value = json.loads(value)
                            data[column_name] = parsed_value
                        except json.JSONDecodeError:
                            data[column_name] = value
                    else:
                        data[column_name] = value

        return data

    def create_json_index(self, field_path: str):
        """
        为指定的JSON字段���径创建索引。

        Parameters:
            field_path: str
                JSON字段路径，例如 "$.name" 或 "$.address.city"
        """
        try:
            # 生成安全的索引名
            safe_name = field_path.replace('$', '').replace('.', '_').replace('[', '_').replace(']', '_')
            index_name = f"idx_json_{safe_name.strip('_')}"

            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON records(json_extract(data, ?))
            """, (field_path,))

            # 分析新创建的索引
            self.conn.execute("ANALYZE")
        except Exception as e:
            raise ValueError(f"Failed to create JSON index: {str(e)}")

    def field_exists(self, field: str, use_cache: bool = True) -> bool:
        """
        Check if a field exists with caching.

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
            cache_key = self._get_cache_key("field_exists", field=field)
            cached_result = self._field_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        try:
            result = self.conn.execute(f"""
                SELECT COUNT(*)
                FROM records
                WHERE json_extract(data, '$.{field}') IS NOT NULL
                LIMIT 1
            """).fetchone()
            exists = result[0] > 0

            if use_cache:
                self._field_cache[cache_key] = exists
            return exists
        except Exception:
            return False

    def list_fields(self, use_cache: bool = True) -> Dict[str, str]:
        """
        List all fields with caching.

        Returns:
            Dict[str, str]: The fields of the storage with their types.
        """
        if use_cache:
            cache_key = self._get_cache_key("list_fields")
            cached_result = self._field_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        try:
            results = self.conn.execute("""
                WITH RECURSIVE
                json_tree(key, type) AS (
                    SELECT
                        json_each.key,
                        CASE
                            WHEN json_each.type = 'integer' THEN 'int'
                            WHEN json_each.type = 'real' THEN 'float'
                            WHEN json_each.type = 'text' THEN 'str'
                            ELSE json_each.type
                        END
                    FROM records, json_each(records.data)
                    GROUP BY json_each.key
                )
                SELECT key, type FROM json_tree
            """).fetchall()

            fields = {f":{field_name}:": field_type
                     for field_name, field_type in results}

            if use_cache:
                self._field_cache[cache_key] = fields
            return fields
        except Exception as e:
            raise ValueError(f"Failed to list fields: {str(e)}")

    def optimize(self):
        """
        优化数据库性能
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute("BEGIN IMMEDIATE TRANSACTION")
            try:
                # 更新统计信息
                cursor.execute("ANALYZE")

                # 整理索引和表数据
                cursor.execute("REINDEX")

                # 整数据库文件
                cursor.execute("VACUUM")

                cursor.execute("COMMIT")

                # 清除缓存
                self._invalidate_cache()
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
        except Exception as e:
            print(f"Failed to optimize database: {str(e)}")

    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """生成缓存键"""
        return f"{operation}:{orjson.dumps(kwargs).decode('utf-8')}"

    def _invalidate_cache(self):
        """清除所有缓存"""
        self._query_cache.clear()
        self._field_cache.clear()

    def __del__(self):
        """
        Close all connections when the object is deleted.
        """
        if hasattr(self, 'conn'):
            self.conn.close()
