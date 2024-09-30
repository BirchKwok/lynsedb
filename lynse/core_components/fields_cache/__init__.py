
__all__ = ['FieldsCache', 'IndexSchema', 'FieldsStorage',
           'FieldsQuery', 'Filter', 'MatchField', 'MatchID',
           'FieldCondition', 'MatchRange', 'ExpressionParser']


from typing import List, Tuple, Union

from .filter import MatchField, MatchID, FieldCondition, Filter, MatchRange
from .expression_parse import ExpressionParser
from .fields_storage import FieldsStorage
from .fields_query import FieldsQuery


class IndexSchema:
    def __init__(self):
        self.indices = {}

    @staticmethod
    def _check_field_name(field_name: str):
        if ':id:' == field_name.strip():
            raise ValueError("The field name ':id:' is reserved.")

    def _add_index(self, field_name: str, field_type):
        self._check_field_name(field_name)
        self.indices[field_name] = field_type

    def add_string_index(self, field_name: str):
        self._add_index(field_name, str)

    def add_int_index(self, field_name: str):
        self._add_index(field_name, int)

    def to_dict(self):
        indices = {k: v.__name__ for k, v in self.indices.items()}
        return indices

    def load_from_dict(self, schema_dict):
        self.indices = {k: eval(v) for k, v in schema_dict.items()}
        return self


class FieldsCache:
    def __init__(self, filepath=None):
        """
        Create a FieldsCache instance.

        Parameters:
            filepath: str
                The storage path for the cache file.
        """
        self.storage = FieldsStorage(filepath)
        self.query_handler = FieldsQuery(self.storage)
        self.filepath = filepath

    def store(self, data: dict, external_id: int):
        """
        Store a record in the cache.

        Parameters:
            data: dict
                The record to be stored.
            external_id: int
                The external ID of the record.

        Returns:
            int: The internal ID of the record.
        """
        return self.storage.store(data, external_id)

    def batch_store(self, data_list: List[Tuple[dict, int]]):
        """
        Batch store multiple records in the cache.

        Parameters:
            data_list: List[Tuple[dict, int]]
                List of records to be stored. Each element is a tuple containing a data dictionary and its corresponding external ID.

        Returns:
            List[int]: List of internal IDs of the stored records.
        """
        return self.storage.batch_store(data_list)

    def query(self, query_filter, filter_ids=None, return_ids_only=True):
        """
        Query the fields cache.

        Parameters:
            query_filter: Filter or dict or FieldExpression (string)
                Filter object or specified filter data.
            filter_ids: List[int]
                List of external IDs to filter.
            return_ids_only: bool
                If True, only return external IDs.

        Returns:
            List[dict]: Records. If not return_ids_only, returns records.
            List[int]: External IDs. If return_ids_only, returns external IDs.
        """
        return self.query_handler.query(query_filter, filter_ids, return_ids_only)

    def retrieve(self, external_id, include_external_id=False):
        """
        Retrieve a record from the cache.

        Parameters:
            external_id: int
                The external ID of the record.
            include_external_id: bool
                If True, include the external ID in the record.

        Returns:
            dict: The record.
        """
        return self.query_handler.retrieve(external_id, include_external_id=include_external_id)

    def retrieve_ids(self, external_ids: List[int], include_external_id: bool = False):
        """
        Retrieve records from the cache.

        Parameters:
            external_ids: List[int]
                List of external IDs of the records.
            include_external_id: bool
                If True, include the external ID in the records.

        Returns:
            List[dict]: List of records.
        """
        return self.query_handler.retrieve_ids(external_ids, include_external_id=include_external_id)

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
        for external_id, data in other.storage.retrieve_all():
            self.store(data, external_id)
        return self

    def delete(self):
        """
        Delete the cache.
        """
        self.storage.delete()

        if hasattr(self, 'query_handler'):
            del self.query_handler

    def build_index(self, schema: Union[IndexSchema, str]):
        """
        Build an index for the cache.

        Parameters:
            schema (IndexSchema or field name string): Index schema or field name string.
                When passing a field name string, the field name must be wrapped in ':',
                such as ':vector:', ':timestamp:'.

        Returns:
            None

        Note:
            ':id:' is a reserved field name and cannot be used.
        """
        if isinstance(schema, str):
            if schema == ':id:':
                raise ValueError("The field name ':id:' is reserved.")

            schema_type = self.storage.list_fields().get(schema, None)
            schema = schema.strip(':')

            if schema_type is None:
                raise ValueError(f"Field '{schema}' not found in the cache.")

            schema_cls = IndexSchema()

            if schema_type == 'str':
                schema_cls.add_string_index(schema)
            elif schema_type in ['int', 'float']:
                schema_cls.add_int_index(schema)
            else:
                raise ValueError(f"Unsupported field type: {schema_type}")

            schema = schema_cls

        if not schema.indices:
            raise ValueError("The index schema is empty.")

        # Build index for all fields in the schema
        for field, field_type in schema.indices.items():
            self.storage.build_index({field: field_type})

    def remove_index(self, field_name: str):
        """
        Remove an index from the cache.

        Parameters:
            field_name: str
                The field name.

        Returns:
            None
        """
        self.storage.remove_index(field_name)

    def remove_all_field_indices(self):
        """
        Remove all field indices from the cache.

        Returns:
            None
        """
        indices = self.list_indices()
        for index in indices:
            self.remove_index(index)

    def list_indices(self):
        """
        List the indices in the cache.

        Returns:
            List[str]: List of indices.
        """
        return [":" + i + ":" if i not in [':id:', ''] else i
                for i in list(self.storage.list_fields().keys())]
