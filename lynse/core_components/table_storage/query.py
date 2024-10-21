from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa


class QueryBuilder:
    def __init__(self, storage,
                 query_vector: Optional[Union[List[float], Tuple[float], np.ndarray, pa.Array, pd.Series]] = None,
                 columns: Optional[List[str]] = None):
        """
        Initialize the QueryBuilder.

        Parameters:
            storage (LyStorage): The LyStorage instance to query.
            query_vector (list of float or tuple of float or np.ndarray or pa.Array or pd.Series, optional):
                The query vector for similarity search.
            columns (list of str], optional): The columns to select.
        """
        self.storage = storage
        self.query_vector = np.array(query_vector, dtype=np.float32) \
            if isinstance(query_vector, (list, tuple, np.ndarray, pd.Series, pa.Array)) else None
        self.columns = columns
        self.conditions = []
        self.limit_value = None
        self.order_by_clause = None
        self.dim = 0

    def where(self, condition: str):
        """
        Add a WHERE clause to the query.

        Parameters:
            condition (str): The condition to filter the data.

        Returns:
            self: The QueryBuilder instance.
        """
        self.conditions.append(condition)
        return self

    def limit(self, n: int):
        """
        Add a LIMIT clause to the query.

        Parameters:
            n (int): The number of rows to limit the result to.

        Returns:
            self: The QueryBuilder instance.
        """
        self.limit_value = n
        return self

    def orderby(self, column: str, desc: bool = False):
        """
        Add an ORDER BY clause to the query.

        Parameters:
            column (str): The column to order by.
            desc (bool): Whether to order in descending order.

        Returns:
            self: The QueryBuilder instance.
        """
        order = "DESC" if desc else "ASC"
        if self.order_by_clause:
            self.order_by_clause += f", {column} {order}"
        else:
            self.order_by_clause = f"{column} {order}"
        return self

    def _format_query_vector(self) -> Tuple[str, str]:
        """
        Format the query vector and determine the type casting for DuckDB.

        Returns:
            Tuple[str, str]: A tuple containing the type casting for the vector column
                             and the formatted query vector with type casting.
        """
        if self.query_vector is not None:
            self.dim = len(self.query_vector)
            # Get the data type of the 'vector' column from the schema
            vector_field = self.storage.schema.field('vector')
            vector_type = vector_field.type  # This is an Arrow type

            # Determine the SQL type string based on the Arrow type
            if pa.types.is_list(vector_type):
                item_type = vector_type.value_type
                if pa.types.is_float32(item_type):
                    element_sql_type = 'FLOAT'
                elif pa.types.is_float64(item_type):
                    element_sql_type = 'DOUBLE'
                elif pa.types.is_int32(item_type):
                    element_sql_type = 'INT'
                elif pa.types.is_int64(item_type):
                    element_sql_type = 'BIGINT'
                else:
                    raise TypeError(f"Unsupported vector element type: {item_type}")
                sql_vector_type = f"{element_sql_type}[{self.dim}]"  # Do not specify array length
            else:
                sql_vector_type = f"FLOAT[{self.dim}]"

            vector_elements = ', '.join(map(str, self.query_vector))
            # Type casting for the vector column
            vector_format_type = f"::{sql_vector_type}"
            # Formatted query vector with type casting
            formatted_vector = f"ARRAY[{vector_elements}]::{sql_vector_type}"
            return vector_format_type, formatted_vector
        else:
            return "", ""

    def _build_query(self) -> str:
        """
        Build the SQL query string.

        Returns:
            str: The SQL query string.
        """
        selected_columns = ', '.join(self.columns) if self.columns else '*'

        where_clauses = self.conditions.copy()

        if self.query_vector is not None:
            vector_format_type, formatted_vector = self._format_query_vector()
            # Construct the similarity expression
            similarity_expr = f"array_cosine_similarity(vector{vector_format_type}, {formatted_vector}) AS __similarity"
            selected_columns += f", {similarity_expr}"
            if self.order_by_clause is None:
                self.order_by_clause = "__similarity DESC"
            else:
                self.order_by_clause += ", __similarity DESC"

        # Build the query after updating selected_columns
        query = f"SELECT {selected_columns} FROM storage"

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        if self.order_by_clause:
            query += f" ORDER BY {self.order_by_clause}"

        if self.limit_value:
            query += f" LIMIT {self.limit_value}"

        return query

    def to_pandas(self) -> pd.DataFrame:
        """
        Execute the query and return the result as a pandas DataFrame.

        Returns:
            pandas.DataFrame: The result of the query.
        """
        query = self._build_query()
        res = self.storage.execute_sql(query)
        return res

    def to_arrow(self) -> pa.Table:
        """
        Execute the query and return the result as a pyarrow Table.

        Returns:
            pyarrow.Table: The result of the query.
        """
        df = self.to_pandas()
        return pa.Table.from_pandas(df)
