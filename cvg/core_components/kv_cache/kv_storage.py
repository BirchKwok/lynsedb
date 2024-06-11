import os
import msgpack
import tempfile
import mmap
from typing import Dict
from pathlib import Path

from cvg.core_components.kv_cache.index import Index
from cvg.core_components.id_checker import IDChecker


class KVCacheStorage:
    RECORD_HEADER_SIZE = 4  # 使用4个字节来存储每个记录的大小

    def __init__(self, filepath=None, as_temp_file=False):
        if filepath is None and not as_temp_file:
            raise ValueError("You must provide a filepath or set as_temp_file to True.")
        if as_temp_file:
            self.filepath = Path(tempfile.mkstemp()[1])
        else:
            self.filepath = Path(filepath)

        self.memory_store = {}
        self.id_mapping = {}
        self.current_internal_id = 0

        self.filter_path = self.filepath.with_suffix('.filter')
        self.index_path = self.filepath.with_suffix('.index')
        self._initialize_id_checker()

        self._load_id_mapping()

        self.index = Index()
        self._load_index()

        self.max_entries = 100000
        self.use_index = False

    def _initialize_id_checker(self):
        self.id_filter = IDChecker()

        if self.filter_path.exists():
            self.id_filter.from_file(self.filter_path)
        else:
            if self.filepath.exists() and self.filepath.stat().st_size > 0:
                for external_id, _ in self.retrieve_all():
                    self.id_filter.add(external_id)
                self.id_filter.to_file(self.filter_path)

    def _load_id_mapping(self):
        id_mapping_path = self.filepath.with_suffix('.idmap')
        if id_mapping_path.exists():
            with open(id_mapping_path, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False)
                for unpacked in unpacker:
                    self.id_mapping.update(unpacked)
            if self.id_mapping:
                self.current_internal_id = max(self.id_mapping.keys()) + 1

    def _load_index(self):
        if self.index_path.exists():
            self.index.load(self.index_path)
            self.use_index = True

    def auto_commit(self):
        if self.memory_store:
            self.commit()

    def store(self, data: dict, external_id: int):
        if not isinstance(data, dict):
            raise ValueError("只支持存储字典类型的数据。")

        if external_id in self.id_filter:
            raise ValueError(f"外部ID {external_id} 已经存在于缓存存储中。")
        self.id_filter.add(external_id)

        internal_id = self.current_internal_id
        self.memory_store[internal_id] = data
        self.current_internal_id += 1

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
        self.memory_store.clear()

        id_mapping_path = self.filepath.with_suffix('.idmap')
        with open(id_mapping_path, 'wb') as f:
            packed_mapping = msgpack.packb(self.id_mapping)
            f.write(packed_mapping)

        self.id_filter.to_file(self.filter_path)

        if self.use_index:
            self.index.save(self.index_path)

    def retrieve_all(self):
        self.auto_commit()

        if os.path.exists(self.filepath):
            with open(self.filepath, 'rb') as f:
                while True:
                    record_size_bytes = f.read(self.RECORD_HEADER_SIZE)
                    if not record_size_bytes:
                        break
                    record_size = int.from_bytes(record_size_bytes, 'big')
                    chunk = f.read(record_size)
                    unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
                    unpacker.feed(chunk)
                    for unpacked in unpacker:
                        for internal_id, data in unpacked.items():
                            yield internal_id, data

    def retrieve_by_external_id(self, external_ids):
        self.auto_commit()

        if isinstance(external_ids, int):
            external_ids = [external_ids]

        results = {}
        with open(self.filepath, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for external_id in external_ids:
                    if external_id in self.id_mapping:
                        offset = self.id_mapping[external_id]
                        mm.seek(offset)
                        record_size = int.from_bytes(mm.read(self.RECORD_HEADER_SIZE), 'big')
                        chunk = mm.read(record_size)
                        unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
                        unpacker.feed(chunk)
                        for unpacked in unpacker:
                            if external_id in unpacked:
                                results[external_id] = unpacked[external_id]
                                break
        return results

    def delete(self):
        if self.filepath and self.filepath.exists():
            self.filepath.unlink()
        id_mapping_path = self.filepath.with_suffix('.idmap')
        if id_mapping_path.exists():
            id_mapping_path.unlink()
        if self.filter_path.exists():
            self.filter_path.unlink()
        if self.index_path.exists():
            self.index_path.unlink()

        self.memory_store.clear()
        self.id_mapping.clear()
        self.filepath = None
        self.memory_store = None
        self.id_mapping = None
        self.current_internal_id = None
        self.max_entries = None
        if self.use_index:
            self.index = None

    def build_index(self, schema: Dict[str, type], rebuild_if_exists=False):
        self.auto_commit()

        for index_name, index_type in schema.items():
            self.index.add_index(index_name, index_type, rebuild_if_exists)
        for internal_id, data in self.retrieve_all():
            self.index.insert(data, internal_id)
        self.use_index = True
        self.index.save(self.index_path)

    def remove_index(self, field_name: str):
        self.index.remove_index(field_name)
        if not self.index.indices:
            self.use_index = False
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
        else:
            self.index.save(self.index_path)

    def commit(self):
        self._flush_to_disk()
