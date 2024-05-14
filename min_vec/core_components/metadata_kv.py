import mmap
import os
from functools import lru_cache

import msgpack
from pyroaring import BitMap


class MetaDataKVCache:
    """
    A class to store metadata key-value pairs and provide fast retrieval for given keys, using memory-mapped files.
    This version optimizes file size and memory usage by using Bitmaps for ID management.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.data_store = {}
        self.id_map = {}
        self.data_to_internal_id = {}
        self.external_ids_dict = {}
        self.last_internal_id = 0
        self.mm = None
        self._load()  # Load existing data if available

    def _get_next_internal_id(self):
        current_id = self.last_internal_id
        self.last_internal_id += 1
        return current_id

    @staticmethod
    def _hash_data(data: dict):
        return msgpack.packb(data, use_bin_type=True)

    def store(self, data, external_id):
        if not isinstance(data, dict):
            raise ValueError("Only dictionary data types are supported for storage.")

        data_hash = self._hash_data(data)
        if data_hash in self.data_to_internal_id:
            internal_id = self.data_to_internal_id[data_hash]
        else:
            internal_id = self._get_next_internal_id()
            self.data_store[internal_id] = data
            self.data_to_internal_id[data_hash] = internal_id

        self.id_map[external_id] = internal_id  # 使用具体的外部 ID 作为键
        if internal_id in self.external_ids_dict:
            self.external_ids_dict[internal_id].add(external_id)
        else:
            self.external_ids_dict[internal_id] = BitMap([external_id])

        return external_id

    def retrieve(self, id):
        internal_id = self.id_map.get(id)
        if internal_id is not None:
            return id, self.data_store[internal_id]
        return None

    def concat(self, other):
        """
        Concatenates another MetaDataKVCache instance into this one.
        Assumes external IDs are unique across both instances or handles conflicts.
        """
        if not isinstance(other, MetaDataKVCache):
            raise ValueError("The 'other' must be an instance of MetaDataKVCache.")

        # Iterate over items in the other instance
        for external_id, other_internal_id in other.id_map.items():
            other_data = other.data_store[other_internal_id]

            # Check if the data already exists in the current cache
            data_hash = self._hash_data(other_data)
            if data_hash in self.data_to_internal_id:
                # Data already exists, just link the external ID to the existing internal ID
                internal_id = self.data_to_internal_id[data_hash]
                self.id_map[external_id] = internal_id

                # Ensure the internal ID exists in external_ids_dict before adding external ID
                if internal_id in self.external_ids_dict:
                    self.external_ids_dict[internal_id].add(external_id)
                else:
                    # If the internal ID is not found, initialize a new BitMap for it
                    self.external_ids_dict[internal_id] = BitMap([external_id])
            else:
                # New data, store it in the current instance
                new_internal_id = self._get_next_internal_id()
                self.data_store[new_internal_id] = other_data
                self.data_to_internal_id[data_hash] = new_internal_id
                self.id_map[external_id] = new_internal_id
                self.external_ids_dict[new_internal_id] = BitMap([external_id])

        # Optionally, update any internal structures if necessary
        self._update_internal_structures()

        return self

    def _update_internal_structures(self):
        """
        Update or rebuild internal structures to ensure data integrity and performance.
        This example focuses on ensuring the external_ids_dict is correct.
        """
        # Rebuild the external_ids_dict based on current id_map
        new_external_ids_dict = {}
        for external_id, internal_id in self.id_map.items():
            if internal_id in new_external_ids_dict:
                new_external_ids_dict[internal_id].add(external_id)
            else:
                new_external_ids_dict[internal_id] = BitMap([external_id])

        # Replace the old external_ids_dict with the newly constructed one
        self.external_ids_dict = new_external_ids_dict

    def save(self):
        # Serialize data with bitmaps
        packed_data = msgpack.packb([self.data_store, self.id_map, self.last_internal_id, self.data_to_internal_id])
        # Resize file if needed
        if self.mm is not None:
            self.mm.close()
        with open(self.filepath, 'wb+') as f:
            f.truncate(len(packed_data))
            f.write(packed_data)
            f.flush()
            self.mm = mmap.mmap(f.fileno(), 0)

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'rb') as f:
                self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                data = msgpack.unpackb(self.mm, raw=False, strict_map_key=False)
                self.data_store, self.id_map, self.last_internal_id, self.data_to_internal_id = data

                # 重新构建 external_ids_dict 使用 Bitmap
                self.external_ids_dict = {}  # 初始化一个空字典
                for ext_id, int_id in self.id_map.items():
                    if int_id in self.external_ids_dict:
                        self.external_ids_dict[int_id].add(ext_id)
                    else:
                        self.external_ids_dict[int_id] = BitMap([ext_id])

    def __del__(self):
        if self.mm:
            self.mm.close()

    def _single_query(self, int_id, data, filter_instance, filter_ids, return_ids_only):
        external_ids = self.external_ids_dict[int_id]

        # 检查必须满足的条件
        must_pass_fields = all(condition.evaluate(data) for condition in
                               filter_instance.must_fields) if filter_instance.must_fields else True

        if filter_instance.must_ids:
            must_pass_ids = BitMap()
            for condition in filter_instance.must_ids:
                matched_ids = BitMap([id for id in external_ids if condition.matcher.match(id)])
                if must_pass_ids:
                    must_pass_ids &= matched_ids
                else:
                    must_pass_ids = matched_ids
        else:
            must_pass_ids = external_ids

        if not must_pass_fields or len(must_pass_ids) == 0:
            return None

        # 检查不必须满足的条件
        must_not_pass_fields = any(condition.evaluate(data) for condition in
                                   filter_instance.must_not_fields) if filter_instance.must_not_fields else False

        if filter_instance.must_not_ids:
            must_not_pass_ids = BitMap()
            for condition in filter_instance.must_not_ids:
                matched_ids = BitMap([id for id in external_ids if condition.matcher.match(id)])
                must_not_pass_ids |= matched_ids
        else:
            must_not_pass_ids = BitMap()

        if must_not_pass_fields:
            return None

        final_pass = must_pass_ids - must_not_pass_ids

        # 处理 "any" 条件
        if filter_instance.any_fields or filter_instance.any_ids:
            any_pass_ids = BitMap()
            if filter_instance.any_fields:
                any_pass_fields = any(condition.evaluate(data) for condition in filter_instance.any_fields)
                if any_pass_fields:
                    any_pass_ids = external_ids

            if filter_instance.any_ids:
                for condition in filter_instance.any_ids:
                    matched_ids = BitMap([id for id in external_ids if condition.matcher.match(id)])
                    any_pass_ids |= matched_ids

            final_pass &= any_pass_ids

        filtered_ids = final_pass.to_array()

        if filter_ids is not None:
            filtered_ids = [id for id in filtered_ids if id in filter_ids]

        if return_ids_only:
            return filtered_ids
        else:
            return [(id, data) for id in filtered_ids]

    @lru_cache(maxsize=1000)
    def query(self, filter_instance, filter_ids=None, return_ids_only=True):
        matched = []

        for int_id, data in self.data_store.items():
            if int_id in self.external_ids_dict:
                res = self._single_query(int_id, data, filter_instance, filter_ids, return_ids_only)
                if res:
                    matched.extend(res)

        return matched
