import os
from typing import List

from .filter import MatchField, MatchID, FieldCondition, Filter, MatchRange
from .expression_parse import ExpressionParser


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
    def __init__(self, filepath=None, as_temp_file=False):
        """
        Create a FieldsCache instance.

        Parameters:
            filepath: str
                The path to the file where the cache will be stored.
            as_temp_file: bool
                If True, the cache will be stored in a temporary file.
        """
        from ...core_components.fields_cache.fields_query import FieldsQuery
        from ...core_components.fields_cache.fields_storage import FieldsStorage

        self.storage = FieldsStorage(filepath, as_temp_file=as_temp_file)
        self.query_handler = FieldsQuery(self.storage)
        self.filepath = filepath

    def store(self, data: dict, external_id: int):
        """
        Store a record in the cache.

        Parameters:
            data: dict
                The record to store.
            external_id: int
                The external ID of the record.

        Returns:
            int: The internal ID of the record.
        """
        return self.storage.store(data, external_id)

    def query(self, query_filter, filter_ids=None, return_ids_only=True):
        """
        Query the fields cache.

        Parameters:
            query_filter: Filter or dict or FieldExpression (string)
                The filter object or the specify data to filter.
            filter_ids: List[int]
                The list of external IDs to filter.
            return_ids_only: bool
                If True, only the external IDs will be returned.

        Returns:
            List[dict]: The records. If not return_ids_only, the records will be returned.
            List[int]: The external IDs. If return_ids_only, the external IDs will be returned.
        """
        if (not isinstance(query_filter, Filter) and not isinstance(query_filter, dict)
                and not isinstance(query_filter, str)):
            raise ValueError("The filter_instance must be an instance of Filter or a dict or a FieldExpression string.")

        return self.query_handler.query(query_filter, filter_ids, return_ids_only)

    def retrieve(self, external_id, include_external_id=False):
        """
        Retrieve a record from the cache.

        Parameters:
            external_id: int
                The external ID of the record.
            include_external_id: bool
                If True, the external ID will be included in the record.

        Returns:
            dict: The record.
        """
        return self.query_handler.retrieve(external_id, include_external_id=include_external_id)

    def retrieve_ids(self, external_ids: List[int], include_external_id: bool = False):
        """
        Retrieve records from the cache.

        Parameters:
            external_ids: List[int]
                The external IDs of the records.
            include_external_id: bool
                If True, the external IDs will be included in the records.

        Returns:
            List[dict]: The records.
        """
        return self.query_handler.retrieve_ids(external_ids, include_external_id=include_external_id)

    def concat(self, other: 'FieldsCache') -> 'FieldsCache':
        """
        Concatenate two caches.

        Parameters:
            other: FieldsCache
                The other cache to concatenate.

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

    def commit(self):
        """
        Commit the cache.

        This method is used to save the cache to disk.
        """
        self.storage.commit()

    def build_index(self, schema: IndexSchema, rebuild_if_exists=False):
        """
        Build an index for the cache.

        Parameters:
            schema (IndexSchema or Field name string): The index schema or the field name string.
                When passing the field name string, the field name must be wrapped with ':',
                like ':vector:', ':timestamp:'.
            rebuild_if_exists: bool
                If True, the index will be rebuilt if it already exists.

        Returns:
            None

        Note:
            The :id: is a reserved field name and cannot be used.
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
            elif schema_type == 'int':
                schema_cls.add_int_index(schema)
            else:
                raise ValueError(f"Unsupported field type: {schema_type}")

            schema = schema_cls

        if not schema.indices:
            raise ValueError("The index schema is empty.")
        self.storage.build_index(schema.indices, rebuild_if_exists)
        self.storage.index.save(self.storage.index_path)

    def remove_index(self, field_name: str):
        """
        Remove an index from the cache.

        Parameters:
            field_name: str
                The name of the field.

        Returns:
            None
        """
        self.storage.remove_index(field_name)
        if not self.storage.index.indices:
            if os.path.exists(self.storage.index_path):
                os.remove(self.storage.index_path)
        else:
            self.storage.index.save(self.storage.index_path)

    def remove_all_field_indices(self):
        """
        Remove all the field indices from the cache.

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
            List[str]: The list of indices.
        """
        return [":" + i + ":" if i not in [':id:', ''] else i
                for i in list(self.storage.index.indices.keys())]
