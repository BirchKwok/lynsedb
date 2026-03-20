__all__ = ['FieldsCache', 'FieldsStorage',
           'FieldsQuery', 'Filter', 'MatchField', 'MatchID',
           'FieldCondition', 'MatchRange', 'ExpressionParser']


from typing import List, Tuple, Union

from lynse.execution_layer.query_view import QueryView

from .filter import MatchField, MatchID, FieldCondition, Filter, MatchRange
from .expression_parse import ExpressionParser
from .fields_storage import FieldsStorage
from .fields_query import FieldsQuery


class FieldsCache:
    def __init__(self, filepath=None, cache_size: int = 1000, batch_size: int = 1000):
        """
        Create a FieldsCache instance.

        Parameters:
            filepath: str
                The storage path for the cache file.
            cache_size: int
                Maximum number of query results to cache.
            batch_size: int
                Size of batches for bulk operations.
        """
        self.storage = FieldsStorage(filepath, cache_size=cache_size, batch_size=batch_size)
        self.query_handler = FieldsQuery(self.storage)
        self.filepath = filepath

    def store(self, data: dict):
        """
        Store a record in the cache.

        Parameters:
            data: dict
                The record to be stored.

        Returns:
            int: The internal ID of the record.
        """
        return self.storage.store(data)

    def batch_store(self, data_list: List[dict]):
        """
        Batch store multiple records in the cache.

        Parameters:
            data_list: List[Tuple[dict, int]]
                List of records to be stored. Each element is a tuple containing a data dictionary and its corresponding external ID.

        Returns:
            List[int]: List of internal IDs of the stored records.
        """
        return self.storage.batch_store(data_list)

    def query(self, query_filter, return_ids_only=True, limit=None):
        """
        Query the fields cache.

        Parameters:
            query_filter: Filter or dict or FieldExpression (string)
                Filter object or specified filter data.
            filter_ids: List[int]
                List of external IDs to filter.
            return_ids_only: bool
                If True, only return external IDs.
            limit: int
                Maximum number of records to return.

        Returns:
            List[dict]: Records. If not return_ids_only, returns records.
            List[int]: External IDs. If return_ids_only, returns external IDs.
        """
        return self.query_handler.query(query_filter, return_ids_only, limit=limit)

    def retrieve(self, id_):
        """
        Retrieve a record from the cache.

        Returns:
            dict: The record.
        """
        return self.query_handler.retrieve(id_)

    def concat(self, other: 'FieldsCache') -> 'FieldsCache':
        """
        Concatenate two caches.

        Parameters:
            other: FieldsCache
                Another cache to concatenate.

        Returns:
            FieldsCache: The concatenated cache.
        """
        if not isinstance(other, FieldsCache):
            raise ValueError("The other cache must be an instance of FieldsCache.")
        for data in other.storage.retrieve_all():
            self.store(data)
        return self

    def list_fields(self):
        """
        List the fields in the cache.

        Returns:
            List[str]: List of fields.
        """
        return list(self.storage.list_fields().keys())

    def retrieve_many(self, ids: List[int]):
        """
        Retrieve multiple records from the cache.

        Parameters:
            ids: List[int]
                List of external IDs.

        Returns:
            List[dict]: List of records.
        """
        return self.query_handler.retrieve_many(ids)
