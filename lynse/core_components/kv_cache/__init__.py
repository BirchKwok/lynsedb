import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


from lynse.core_components.kv_cache.filter import MatchField, MatchID, FieldCondition, Filter, MatchRange


@dataclass
class IndexSchema:
    indices: dict = None

    def __post_init__(self):
        if self.indices is None:
            self.indices = {}

    def add_index(self, index_name: str, index_type: type):
        self.indices[index_name] = index_type


class VeloKV:
    def __init__(self, filepath=None, as_temp_file=False):
        """
        Create a VeloKV instance.

        Parameters:
            filepath: str
                The path to the file where the cache will be stored.
            as_temp_file: bool
                If True, the cache will be stored in a temporary file.
        """
        from lynse.core_components.kv_cache.kv_query import KVCacheQuery
        from lynse.core_components.kv_cache.kv_storage import KVCacheStorage

        self.storage = KVCacheStorage(filepath, as_temp_file=as_temp_file)
        self.query_handler = KVCacheQuery(self.storage)
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

    def query(self, filter_instance, filter_ids=None, return_ids_only=True):
        """
        Query the cache.

        Parameters:
            filter_instance: Filter
                The filter object.
            filter_ids: List[int]
                The list of external IDs to filter.
            return_ids_only: bool
                If True, only the external IDs will be returned.

        Returns:
            dict: The records that match the filter.
        """
        return self.query_handler.query(filter_instance, filter_ids, return_ids_only)

    def retrieve(self, external_id):
        """
        Retrieve a record from the cache.

        Parameters:
            external_id: int
                The external ID of the record.

        Returns:
            dict: The record.
        """
        return self.query_handler.retrieve(external_id)

    def retrieve_ids(self, external_ids: List[int]):
        """
        Retrieve records from the cache.

        Parameters:
            external_ids: List[int]
                The external IDs of the records.

        Returns:
            List[dict]: The records.
        """
        return self.query_handler.retrieve_ids(external_ids)

    def concat(self, other: 'VeloKV') -> 'VeloKV':
        """
        Concatenate two caches.

        Parameters:
            other: VeloKV
                The other cache to concatenate.

        Returns:
            VeloKV: The concatenated cache.
        """
        if not isinstance(other, VeloKV):
            raise ValueError("The other cache must be an instance of VeloKV.")
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
            schema: IndexSchema
                The index schema.
            rebuild_if_exists: bool
                If True, the index will be rebuilt if it already exists.

        Returns:
            None
        """
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


class IndexBuilder:
    def __init__(self, collection, schema):
        """
        Build an index schema for a collection.

        Parameters:
            collection: ExclusiveDB
            schema: IndexSchema

        """
        if collection.name == "Local":
            self.velo_kv = VeloKV(filepath=Path(collection._database_path) / 'fields_index')
        else:
            raise NotImplementedError("Only local collections are supported.")
            # self.velo_kv = VeloKV(filepath=collection.get_collection_path())
        self.schema = schema

    def build(self, rebuild_if_exists=False):
        """
        Build the index.

        Parameters:
            rebuild_if_exists: bool
                If True, the index will be rebuilt if it already exists.

        Returns:
            None
        """
        return self.velo_kv.build_index(self.schema, rebuild_if_exists=rebuild_if_exists)

    def remove(self, field_name: str):
        """
        Remove an index from the collection.

        Parameters:
            field_name: str
                The name of the field.

        Returns:
            None
        """
        return self.velo_kv.remove_index(field_name)
