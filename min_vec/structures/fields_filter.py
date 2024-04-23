"""fields_filter.py: this file contains the FieldsMapper class. It is used to map the fields of the data."""
import operator

import gzip
import random
from pathlib import Path

import msgpack
import cloudpickle


class SkipListNode:
    def __init__(self, key, level, data=None):
        self.key = key
        self.data = data
        self.forward = [None] * (level + 1)


class SkipList:
    def __init__(self, max_level, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = SkipListNode(-1, max_level)
        self.level = 0

    def random_level(self):
        """Randomly generate a level for inserting a new node."""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def insert(self, key, data=None):
        """Insert a key along with optional data into the skip list."""
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current is None or current.key != key:
            new_level = self.random_level()
            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level

            new_node = SkipListNode(key, new_level, data)
            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node

    def search(self, key):
        """Search for a key in the skip list and return the node."""
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        if current and current.key == key:
            return current.data
        return None

    def delete(self, key):
        """Delete a node by key from the skip list."""
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]
        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1


class FieldIndex:
    def __init__(self, max_level=4, p=0.5):
        self.data_store = {}
        self.id_map = {}
        self.data_to_internal_id = {}
        self.last_internal_id = 0
        self.index = SkipList(max_level, p)

    def _get_next_internal_id(self):
        current_id = self.last_internal_id
        self.last_internal_id += 1
        return current_id

    def _hash_data(self, data):
        return msgpack.packb(data, use_bin_type=True)

    def store(self, data, ids=None):
        if not isinstance(data, dict):
            raise ValueError("Only dictionary data types are supported for storage.")

        data_hash = self._hash_data(data)
        if data_hash in self.data_to_internal_id:
            internal_id = self.data_to_internal_id[data_hash]
        else:
            internal_id = self._get_next_internal_id()
            self.data_store[internal_id] = data
            self.data_to_internal_id[data_hash] = internal_id
            self.index.insert(internal_id, data)
        if ids is None:
            ids = [self._get_next_internal_id()]
        if isinstance(ids, int):
            ids = [ids]

        for id in ids:
            self.id_map[id] = internal_id
        return ids

    def retrieve(self, id):
        internal_id = self.id_map.get(id)
        if internal_id is not None:
            return (id, self.data_store[internal_id])
        return None

    def query(self, filter_instance, filter_ids=None, return_ids_only=True):
        """Query items that match the given Filter instance using the skip list."""
        matched = []
        current = self.index.header.forward[0]
        while current:
            if filter_instance.apply(self.data_store[current.key]):
                external_ids = [id for id, int_id in self.id_map.items() if int_id == current.key]
                if filter_ids is not None:
                    external_ids = [id for id in external_ids if id in filter_ids]
                matched.extend(external_ids)
            current = current.forward[0]

        if return_ids_only:
            return matched
        else:
            return [(id, self.data_store[self.id_map[id]]) for id in matched]

    def save(self, filepath):
        """Save all data to a file with gzip compression."""
        try:
            # 使用 msgpack 保存基础数据
            with gzip.open(filepath, 'wb') as f:
                f.write(msgpack.packb([self.data_store, self.id_map, self.last_internal_id, self.data_to_internal_id]))
            # 使用 cloudpickle 和 gzip 保存跳表对象
            with gzip.open(Path(filepath).parent / (Path(filepath).stem + '-obj.mvdb'), 'wb') as f:
                cloudpickle.dump(self.index, f)
        except IOError as e:
            print(f"Error saving to file {filepath}: {e}")

    def load(self, filepath):
        """Load all data from a file with gzip decompression."""
        try:
            with gzip.open(filepath, 'rb') as f:
                self.data_store, self.id_map, self.last_internal_id, self.data_to_internal_id = msgpack.unpackb(
                    f.read(), strict_map_key=False, raw=False, use_list=False)
            with gzip.open(Path(filepath).parent / (Path(filepath).stem + '-obj.mvdb'), 'rb') as f:
                self.index = cloudpickle.load(f)

            return self

        except IOError as e:
            print(f"Error loading from file {filepath}: {e}")
