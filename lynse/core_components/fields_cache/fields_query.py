from typing import List, Union, Dict, Any
import json
import pandas as pd


class FieldsQuery:
    """
    The FieldsQuery class is used to query data in the fields_cache.
    Supports direct SQL-like query syntax for filtering records.

    Examples:
        - Basic comparison: "age > 18"
        - Range query: "10 < days <= 300"
        - Text search: "name LIKE '%John%'"
        - Multiple conditions: "age > 18 AND city = 'New York'"
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
        正确转义标识符。

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
            # 使用 ApexBase 的查询接口
            if return_ids_only:
                # 如果只需要ID，直接使用query方法
                if query_filter:
                    # ApexBase 使用 SQL WHERE 子句
                    results = self.storage.query(query_filter, limit=limit)
                else:
                    # 没有过滤条件，获取所有ID
                    results = self.storage.query(limit=limit)

                # 处理偏移量
                if offset:
                    results = results[offset:]

                return results
            else:
                # 需要返回完整记录
                if query_filter:
                    # 使用 SQL 查询
                    results = self.storage.execute_query(
                        f"SELECT * FROM records WHERE {query_filter}"
                    )
                else:
                    results = self.storage.execute_query("SELECT * FROM records")

                # 处理偏移量和限制
                if offset:
                    results = results[offset:]
                if limit:
                    results = results[:limit]

                # 解析结果
                records = []
                for row in results:
                    record = {}
                    # ApexBase 返回的结果需要解析
                    # 假设返回的是列表或类似的结构
                    if hasattr(row, '_asdict'):
                        record = row._asdict()
                    elif hasattr(row, 'keys'):
                        record = dict(row)
                    else:
                        # 如果是元组，尝试获取字段名
                        fields = self.list_fields()
                        if isinstance(row, (list, tuple)):
                            for i, field in enumerate(fields):
                                if i < len(row):
                                    value = row[i]
                                    if value is not None:
                                        # 尝试解析JSON
                                        if isinstance(value, str):
                                            try:
                                                record[field] = json.loads(value)
                                            except json.JSONDecodeError:
                                                record[field] = value
                                        else:
                                            record[field] = value
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
            fields = self.storage.list_fields()
            # 移除冒号前缀
            return [field.strip(':') for field in fields.keys()]
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
            fields = self.storage.list_fields()
            # 添加冒号前缀来匹配
            key = f":{field_name}:"
            if key in fields:
                return fields[key]
            return None
        except Exception as e:
            raise ValueError(f"Failed to get field type: {str(e)}")

    def create_field_index(self, field_name: str):
        """
        为指定字段创建索引。

        Parameters:
            field_name: str
                字段名称
        """
        # ApexBase 自动处理索引
        pass

    def retrieve(self, id_: int) -> Dict[str, Any]:
        """
        Retrieve a single record by ID.

        Parameters:
            id_: int
                The ID of the record to retrieve.

       [str, Any]: Returns:
            Dict The record if found, None otherwise.
        """
        try:
            result = self.storage.retrieve(id_)

            if result:
                # 解析 JSON 字段
                record = {}
                for key, value in result.items():
                    if value is not None:
                        # 尝试解析JSON
                        if isinstance(value, str):
                            try:
                                record[key] = json.loads(value)
                            except json.JSONDecodeError:
                                record[key] = value
                        else:
                            record[key] = value
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
            results = self.storage.retrieve_many(ids)

            # 必须返回 ResultView 类型
            if not hasattr(results, 'to_pandas'):
                raise ValueError(f"retrieve_many must return ResultView, got {type(results)}")

            # 处理 ResultView 类型
            df = results.to_pandas()
            # 转换为 dict 列表
            records = df.to_dict('records')
            return records

        except Exception as e:
            raise ValueError(f"Failed to retrieve records: {str(e)}")

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all records.

        Returns:
            List[Dict[str, Any]]: List of all records
        """
        try:
            results = self.storage.retrieve_all()

            # 解析 JSON 字段
            records = []
            for result in results:
                if result:
                    record = {}
                    for key, value in result.items():
                        if value is not None:
                            # 尝试解析JSON
                            if isinstance(value, str):
                                try:
                                    record[key] = json.loads(value)
                                except json.JSONDecodeError:
                                    record[key] = value
                            else:
                                record[key] = value
                    records.append(record)

            return records

        except Exception as e:
            raise ValueError(f"Failed to retrieve all records: {str(e)}")
