from typing import List, Union, Dict, Any
import json

class FieldsQuery:
    """
    The FieldsQuery class is used to query data in the fields_cache.
    Supports direct SQL-like query syntax for filtering records.

    Examples:
        - Basic comparison: "age > 18"
        - Range query: "10 < days <= 300"
        - Text search: "name LIKE '%John%'"
        - Multiple conditions: "age > 18 AND city = 'New York'"
        - JSON field access: "json_extract(data, '$.address.city') = 'Beijing'"
        - Numeric operations: "CAST(json_extract(data, '$.price') AS REAL) * CAST(json_extract(data, '$.quantity') AS REAL) > 1000"
    """
    def __init__(self, storage):
        """
        Initialize the FieldsQuery class.

        Parameters:
            storage: Storage
                The storage object.
        """
        self.storage = storage

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

    def query(self, query_filter: str = None, return_ids_only: bool = True, limit: int = None, offset: int = None) -> Union[List[int], List[Dict[str, Any]]]:
        """
        Query records using SQL-like filter syntax.

        Parameters:
            query_filter: str
                SQL-like filter condition. Examples:
                - "age > 18"
                - "name LIKE '%John%'"
                - "age > 18 AND city = 'New York'"
                If None, returns all records.
            return_ids_only: bool
                If True, only return IDs. If False, return complete records.
            limit: int
                Maximum number of records to return.
            offset: int
                Number of records to skip.

        Returns:
            Union[List[int], List[Dict[str, Any]]]: List of IDs or complete records.
        """
        try:
            # 构建基本查询
            if return_ids_only:
                base_query = "SELECT _id FROM records"
            else:
                # 获取所有字段名（除了_id）
                fields = self.list_fields()
                if not fields:
                    return []
                fields_str = ', '.join(f'"{field}"' for field in fields)
                base_query = f"SELECT {fields_str} FROM records"

            # 添加过滤条件
            if query_filter:
                # 替换查询中的__id__为_id
                query_filter = query_filter.replace('__id__', '_id')

                # 检查字段是否存在，如果不存在且不是复杂查询，返回空结果
                field_name = query_filter.split()[0].strip('"')  # 移除可能存在的引号
                if field_name != '_id' and not any(op in field_name for op in ['>', '<', '=', '!', 'LIKE', 'IN']):
                    exists = self.storage.conn.execute(
                        "SELECT 1 FROM fields_meta WHERE field_name = ?",
                        [field_name]
                    ).fetchone()
                    if not exists:
                        return [] if return_ids_only else {}

                # 确保字段名被正确引用
                import re
                def quote_field_names(match):
                    field = match.group(1)
                    if field != '_id' and not field.startswith('"'):
                        return self._quote_identifier(field)
                    return field

                # 使用正则表达式替换字段名
                query_filter = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?=\s*[<>=!]|\s+LIKE|\s+IN\s*\()', quote_field_names, query_filter)
                base_query += f" WHERE {query_filter}"

            # 添加分页
            if limit is not None:
                base_query += f" LIMIT {limit}"
                if offset is not None:
                    base_query += f" OFFSET {offset}"

            # 执行查询
            cursor = self.storage.conn.cursor()
            results = cursor.execute(base_query).fetchall()

            if return_ids_only:
                return [row[0] for row in results]
            else:
                # 构建结果字典，不包含内部ID
                records = []
                fields = self.list_fields()
                for row in results:
                    record = {}
                    for i, value in enumerate(row):
                        if value is not None:
                            # 尝试解析JSON字符串
                            if isinstance(value, str):
                                try:
                                    import json
                                    parsed_value = json.loads(value)
                                    record[fields[i]] = parsed_value
                                except json.JSONDecodeError:
                                    record[fields[i]] = value
                            else:
                                record[fields[i]] = value
                    records.append(record)
                return records

        except Exception as e:
            raise ValueError(f"Query failed: {str(e)}")

    def list_fields(self) -> List[str]:
        """
        获取所有可用字段列表。

        Returns:
            List[str]: 字段名列表
        """
        try:
            cursor = self.storage.conn.cursor()
            results = cursor.execute("SELECT field_name FROM fields_meta").fetchall()
            return [row[0] for row in results]
        except Exception as e:
            raise ValueError(f"Failed to list fields: {str(e)}")

    def get_field_type(self, field_name: str) -> str:
        """
        获取字段类型。

        Parameters:
            field_name: str
                字段名称

        Returns:
            str: 字段类型
        """
        try:
            cursor = self.storage.conn.cursor()
            result = cursor.execute(
                "SELECT field_type FROM fields_meta WHERE field_name = ?",
                [field_name]
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            raise ValueError(f"Failed to get field type: {str(e)}")

    def create_field_index(self, field_name: str):
        """
        为指定字段创建索引。

        Parameters:
            field_name: str
                字段名称
        """
        try:
            cursor = self.storage.conn.cursor()
            # 检查字段是否存在
            field_exists = cursor.execute(
                "SELECT 1 FROM fields_meta WHERE field_name = ?",
                [field_name]
            ).fetchone()

            if field_exists:
                # 创建索引，使用引号包裹字段名
                index_name = f"idx_{field_name}"
                quoted_field_name = self._quote_identifier(field_name)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON records({quoted_field_name})
                """)
                cursor.execute("ANALYZE")
            else:
                raise ValueError(f"Field {field_name} does not exist")
        except Exception as e:
            raise ValueError(f"Failed to create index: {str(e)}")

    def retrieve(self, id_: int) -> Dict[str, Any]:
        """
        Retrieve a single record by ID.

        Parameters:
            id_: int
                The ID of the record to retrieve.

        Returns:
            Dict[str, Any]: The record if found, None otherwise.
        """
        try:
            # 获取所有字段名（除了_id）
            fields = self.list_fields()
            if not fields:
                return None
            fields_str = ', '.join(f'"{field}"' for field in fields)

            result = self.storage.conn.execute(
                f"SELECT {fields_str} FROM records WHERE _id = ?",
                [id_]
            ).fetchone()

            if result:
                # 构建结果字典，不包含内部ID
                record = {}
                for i, value in enumerate(result):
                    if value is not None:
                        # 尝试解析JSON字符串
                        if isinstance(value, str):
                            try:
                                import json
                                parsed_value = json.loads(value)
                                record[fields[i]] = parsed_value
                            except json.JSONDecodeError:
                                record[fields[i]] = value
                        else:
                            record[fields[i]] = value
                return record
            return None
        except Exception as e:
            raise ValueError(f"Failed to retrieve record: {str(e)}")

    def retrieve_many(self, ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple records by their IDs.

        Parameters:
            ids: List[int]
                List of record IDs to retrieve.

        Returns:
            List[Dict[str, Any]]: List of retrieved records.
        """
        if not ids:
            return []

        try:
            cursor = self.storage.conn.cursor()

            # 获取所有字段名（除了_id）
            fields = self.list_fields()
            if not fields:
                return []
            fields_str = ', '.join(f'"{field}"' for field in fields)

            # 使用UNION ALL来创建ID列表
            id_queries = [f"SELECT {id_} AS _id" for id_ in ids]
            id_list_query = " UNION ALL ".join(id_queries)

            # 使用LEFT JOIN来获取记录
            results = cursor.execute(f"""
                SELECT v._id, {fields_str}
                FROM ({id_list_query}) AS v
                LEFT JOIN records r ON r._id = v._id
                ORDER BY v._id
            """).fetchall()

            # 构建ID到记录的映射
            id_to_record = {}
            for row in results:
                record_id = row[0]  # 第一列是_id
                if any(row[1:]):  # 检查是否有任何非空字段
                    record = {}
                    for i, value in enumerate(row[1:], 0):  # 跳过_id列
                        if value is not None:
                            # 尝试解析JSON字符串
                            if isinstance(value, str):
                                try:
                                    parsed_value = json.loads(value)
                                    record[fields[i]] = parsed_value
                                except json.JSONDecodeError:
                                    record[fields[i]] = value
                            else:
                                record[fields[i]] = value
                    id_to_record[record_id] = record

            # 按照输入ID的顺序返回记录
            return [id_to_record.get(id_) for id_ in ids]

        except Exception as e:
            raise ValueError(f"Failed to retrieve records: {str(e)}")

    def _create_temp_indexes(self, query_filter: str):
        """
        根据查询条件创建临时索引。

        Parameters:
            query_filter: str
                查询条件
        """
        import re

        # 提取JSON路径
        json_paths = re.findall(r'json_extract\(data,\s*\'([^\']+)\'\)', query_filter)

        if json_paths:
            cursor = self.storage.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                for path in json_paths:
                    # 为每个JSON路径创建临时索引
                    safe_name = path.replace('$', '').replace('.', '_').replace('[', '_').replace(']', '_')
                    index_name = f"temp_idx_{safe_name.strip('_')}"

                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON records(json_extract(data, ?))
                    """, (path,))

                cursor.execute("COMMIT")
            except Exception:
                cursor.execute("ROLLBACK")
