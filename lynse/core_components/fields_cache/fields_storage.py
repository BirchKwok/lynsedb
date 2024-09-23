import os
import msgpack
import tempfile
import mmap
from typing import Dict
from pathlib import Path

from ...core_components.fields_cache.index import Index
from ...core_components.id_checker import IDChecker


class FieldsStorage:
    RECORD_HEADER_SIZE = 4

    def __init__(self, filepath=None, as_temp_file=False):
        if filepath is None and not as_temp_file:
            raise ValueError("You must provide a filepath or set as_temp_file to True.")
        if as_temp_file:
            self.filepath = Path(tempfile.mkstemp()[1])
        else:
            self.filepath = Path(filepath)

        self.memory_store = {}
        self.id_mapping = {}
        self.external_to_internal = {}
        self.current_internal_id = 0

        self.filter_path = self.filepath.with_suffix('.filter')
        self.index_path = self.filepath.with_suffix('.index')
        self._initialize_id_checker()

        self._load_mappings()

        self.index = Index()
        self._load_index()

        self.max_entries = 100000
        self.use_index = False

        self.all_fields = {}
        self._load_all_fields()

    def _initialize_id_checker(self):
        self.id_filter = IDChecker()

        if self.filter_path.exists():
            self.id_filter.from_file(self.filter_path)
        else:
            if self.filepath.exists() and self.filepath.stat().st_size > 0:
                for external_id, _ in self.retrieve_all():
                    self.id_filter.add(external_id)
                self.id_filter.to_file(self.filter_path)

    def _load_mappings(self):
        id_mapping_path = self.filepath.with_suffix('.idmap')
        external_to_internal_path = self.filepath.with_suffix('.ext2int')
        if id_mapping_path.exists():
            with open(id_mapping_path, 'rb') as f:
                unpacked = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
                self.id_mapping.update(unpacked)
            if self.id_mapping:
                self.current_internal_id = max(self.id_mapping.keys()) + 1
        if external_to_internal_path.exists():
            with open(external_to_internal_path, 'rb') as f:
                unpacked = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
                self.external_to_internal.update(unpacked)

    def _load_index(self):
        if self.index_path.exists():
            self.index.load(self.index_path)
            self.use_index = True

    def _load_all_fields(self):
        if self.filepath.exists() and self.filepath.stat().st_size > 0:
            for _, data in self.retrieve_all():
                for field, value in data.items():
                    if field not in self.all_fields:
                        self.all_fields[field] = type(value).__name__

    def auto_commit(self):
        if self.memory_store:
            self.commit()

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
        if not isinstance(data, dict):
            raise ValueError("Only dictionaries are allowed as data.")

        if external_id in self.id_filter:
            raise ValueError(f"external_id {external_id} already exists.")
        self.id_filter.add(external_id)

        data = {":id:": external_id, **data}
        internal_id = self.current_internal_id
        self.memory_store[internal_id] = data
        self.external_to_internal[external_id] = internal_id
        self.current_internal_id += 1

        for field, value in data.items():
            self.all_fields[field] = type(value).__name__

        if self.use_index:
            self.index.insert(data, internal_id)

        if len(self.memory_store) >= self.max_entries:
            self._flush_to_disk()

        return internal_id

    def _flush_to_disk(self):
        if not self.memory_store:
            return

        with open(self.filepath, 'ab') as f:
            for internal_id, data in self.memory_store.items():
                packed_data = msgpack.packb({internal_id: data})
                record_size = len(packed_data)
                offset = f.tell()
                f.write(record_size.to_bytes(self.RECORD_HEADER_SIZE, 'big'))
                f.write(packed_data)
                self.id_mapping[internal_id] = offset
        self.memory_store = {}

        id_mapping_path = self.filepath.with_suffix('.idmap')
        with open(id_mapping_path, 'wb') as f:
            packed_mapping = msgpack.packb(self.id_mapping)
            f.write(packed_mapping)

        external_to_internal_path = self.filepath.with_suffix('.ext2int')
        with open(external_to_internal_path, 'wb') as f:
            packed_mapping = msgpack.packb(self.external_to_internal)
            f.write(packed_mapping)

        self.id_filter.to_file(self.filter_path)

        # save field list
        fields_path = self.filepath.with_suffix('.fields')
        with open(fields_path, 'wb') as f:
            packed_fields = msgpack.packb(list(self.all_fields))
            f.write(packed_fields)

        if self.use_index:
            self.index.save(self.index_path)

    def retrieve_all(self, include_external_id=True):
        self.auto_commit()

        if not os.path.exists(self.filepath):
            return

        with open(self.filepath, 'rb') as f:
            while True:
                record_size_bytes = f.read(self.RECORD_HEADER_SIZE)
                if not record_size_bytes:
                    break
                record_size = int.from_bytes(record_size_bytes, 'big')
                chunk = f.read(record_size)
                unpacked = msgpack.unpackb(chunk, raw=False, strict_map_key=False)
                for internal_id, data in unpacked.items():
                    external_id = data.get(':id:')
                    if not include_external_id:
                        del data[':id:']
                    yield external_id, data

    def retrieve_by_external_id(self, external_ids, include_external_id=True):
        self.auto_commit()

        if isinstance(external_ids, int):
            external_ids = [external_ids]

        results = []
        with open(self.filepath, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for external_id in external_ids:
                    if external_id in self.external_to_internal:
                        internal_id = self.external_to_internal[external_id]
                        offset = self.id_mapping[internal_id]
                        mm.seek(offset)
                        record_size = int.from_bytes(mm.read(self.RECORD_HEADER_SIZE), 'big')
                        chunk = mm.read(record_size)
                        unpacked = msgpack.unpackb(chunk, raw=False, strict_map_key=False)
                        if internal_id in unpacked:
                            if not include_external_id:
                                del unpacked[internal_id][':id:']
                            results.append(unpacked[internal_id])
        return results

    def delete(self):
        if self.filepath and self.filepath.exists():
            self.filepath.unlink()
        id_mapping_path = self.filepath.with_suffix('.idmap')
        if id_mapping_path.exists():
            id_mapping_path.unlink()
        external_to_internal_path = self.filepath.with_suffix('.ext2int')
        if external_to_internal_path.exists():
            external_to_internal_path.unlink()
        if self.filter_path.exists():
            self.filter_path.unlink()
        if self.index_path.exists():
            self.index_path.unlink()

        fields_path = self.filepath.with_suffix('.fields')
        if fields_path.exists():
            fields_path.unlink()

        self.memory_store.clear()
        self.id_mapping.clear()
        self.external_to_internal.clear()
        self.filepath = None
        self.memory_store = None
        self.id_mapping = None
        self.current_internal_id = None
        self.max_entries = None
        self.use_index = False
        self.index = None
        self.all_fields = set()

    def build_index(self, schema: Dict[str, type], rebuild_if_exists=False):
        self.auto_commit()

        # pre-select data to check the schema
        _, pre_select_data = next(self.retrieve_all())

        for index_name, index_type in schema.items():
            # dtype check
            if not isinstance(pre_select_data.get(index_name), index_type):
                raise TypeError(f"Field {index_name} is not of type {index_type}")

            self.index.add_index(index_name, index_type, rebuild_if_exists)
        for external_id, data in self.retrieve_all():
            self.index.insert(data, external_id)
        self.use_index = True
        self.index.save(self.index_path)

    def remove_index(self, field_name: str):
        self.index.remove_index(field_name)
        if not self.index.indices:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)

            self.index = Index()
            self.use_index = False
        else:
            self.index.save(self.index_path)

    def commit(self):
        self._flush_to_disk()

    def list_fields(self):
        """
        Return all field names and their types.

        Returns:
            dict: A dictionary with field names as keys and field types as values.
        """
        def _warp_field_name(field):
            return f":{field}:" if field not in [':id:', ''] else field

        return {_warp_field_name(field): dtype for field, dtype in self.all_fields.copy().items()}
