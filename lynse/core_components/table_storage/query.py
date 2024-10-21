import numpy as np
import pandas as pd
from typing import List
import duckdb


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
