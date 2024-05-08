"""metadata_kv.py: this file contains the FieldsMapper class. It is used to map the fields of the data."""
import gzip
import random
from pathlib import Path

import msgpack
import numpy as np

from min_vec.core_components.filter import Filter


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

    def serialize(self):
        """ Serialize the skip list into a list of tuples for each node. """
        nodes = []
        current = self.header.forward[0]
        while current:
            nodes.append((current.key, current.data, [f.key if f else None for f in current.forward]))
            current = current.forward[0]
        return nodes

    def deserialize(self, nodes):
        """ Deserialize from a list of node data to rebuild the skip list. """
        self.header = SkipListNode(-1, self.max_level)
        self.level = 0
        for key, data, forwards in reversed(nodes):
            new_node = SkipListNode(key, len(forwards) - 1, data)
            for i in range(len(forwards)):
                new_node.forward[i] = self.find_node(forwards[i]) if forwards[i] is not None else None
            self.insert_node_at_level(new_node, len(forwards) - 1)

    def find_node(self, key):
        """Find a node by key in the skip list."""
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        return current.forward[0] if current.forward[0] and current.forward[0].key == key else None

    def insert_node_at_level(self, node, level):
        """Insert a node at a specific level in the skip list."""
        current = self.header
        update = [None] * (level + 1)
        for i in range(level, -1, -1):
            while current.forward[i] and current.forward[i].key < node.key:
                current = current.forward[i]
            update[i] = current
        for i in range(level + 1):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node
        if level > self.level:
            self.level = level


class MetaDataKVCache:
    """
    A class to store metadata key-value pairs and provide fast retrieval for given keys.
    """
    def __init__(self, max_level=10, p=0.5):
        self.data_store = {}
        self.id_map = {}
        self.data_to_internal_id = {}
        self.last_internal_id = 0
        self.index = SkipList(max_level, p)

    def _get_next_internal_id(self):
        current_id = self.last_internal_id
        self.last_internal_id += 1
        return current_id

    def _hash_data(self, data: dict):
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

    def concat(self, another_field_index):
        for internal_id, data in another_field_index.data_store.items():
            data_hash = self._hash_data(data)
            if data_hash not in self.data_to_internal_id:
                self.data_store[internal_id] = data
                self.data_to_internal_id[data_hash] = internal_id
                self.index.insert(internal_id, data)
            for ext_id, int_id in another_field_index.id_map.items():
                if ext_id not in self.id_map:
                    self.id_map[ext_id] = int_id

    def retrieve(self, id):
        internal_id = self.id_map.get(id)
        if internal_id is not None:
            return id, self.data_store[internal_id]
        return None

    def query(self, filter_instance: Filter, filter_ids=None, return_ids_only=True):
        matched = []
        external_ids_dict = {}
        for id, int_id in self.id_map.items():
            if int_id not in external_ids_dict:
                external_ids_dict[int_id] = [id]
            else:
                external_ids_dict[int_id].append(id)

        current = self.index.header.forward[0]
        while current:
            data = self.data_store[current.key]
            external_ids = external_ids_dict[current.key]
            external_ids_len = len(external_ids)

            # Check must conditions for fields and IDs.
            must_pass_fields = all(condition.evaluate(data) for condition in
                                   filter_instance.must_fields) if filter_instance.must_fields else True
            must_pass_ids = np.all([condition.matcher.match(external_ids) for condition in filter_instance.must_ids],
                                   axis=0) if filter_instance.must_ids else np.ones(external_ids_len, dtype=bool)
            if filter_instance.must_fields:
                must_pass = np.where(must_pass_ids, must_pass_fields, must_pass_ids)
            else:
                must_pass = must_pass_ids

            # Skip to next node if must conditions are not met.
            if not must_pass.any():
                current = current.forward[0]
                continue

            # Check must not conditions for fields and IDs.
            must_not_pass_fields = any(condition.evaluate(data) for condition in
                                       filter_instance.must_not_fields) if filter_instance.must_not_fields else False
            must_not_pass_ids = np.any(
                [condition.matcher.match(external_ids) for condition in filter_instance.must_not_ids],
                axis=0) if filter_instance.must_not_ids else np.zeros(external_ids_len, dtype=bool)

            if filter_instance.must_not_fields:
                must_not_pass = np.where(must_not_pass_ids, must_not_pass_ids, must_not_pass_fields)
            else:
                must_not_pass = must_not_pass_ids

            final_pass = must_pass & ~must_not_pass

            # Only proceed with any conditions if must conditions are fully met and must_not conditions are not met.
            if must_pass.all() and not must_not_pass.all():
                if not filter_instance.any_fields and not filter_instance.any_ids:
                    pass
                else:
                    any_pass_fields = any(condition.evaluate(data) for condition in
                                          filter_instance.any_fields) if filter_instance.any_fields else False
                    any_pass_ids = np.any([condition.matcher.match(external_ids) for condition in filter_instance.any_ids],
                                          axis=0) if filter_instance.any_ids else np.zeros(external_ids_len, dtype=bool)

                    if filter_instance.any_fields:
                        if filter_instance.any_ids:
                            any_pass = any_pass_fields | any_pass_ids
                        else:
                            any_pass = any_pass_fields
                    else:
                        any_pass = any_pass_ids

                    # Combine any conditions with final pass.
                    final_pass = np.where(final_pass, final_pass & any_pass, final_pass)

            # Combine all conditions to finalize.
            filtered_ids = np.array(external_ids)[final_pass]

            if filter_ids is not None:
                filtered_ids = [id for id in filtered_ids if id in filter_ids]

            if return_ids_only:
                matched.extend(filtered_ids)
            else:
                matched.extend([(id, data) for id in filtered_ids])

            current = current.forward[0]

        return matched

    def save(self, filepath):
        try:
            with gzip.open(filepath, 'wb') as f:
                f.write(msgpack.packb([self.data_store, self.id_map, self.last_internal_id, self.data_to_internal_id]))
            with gzip.open(Path(filepath).with_suffix('.sl'), 'wb') as f:
                f.write(msgpack.packb(self.index.serialize()))
        except IOError as e:
            print(f"Error saving to file {filepath}: {e}")

    def load(self, filepath):
        try:
            with gzip.open(filepath, 'rb') as f:
                self.data_store, self.id_map, self.last_internal_id, self.data_to_internal_id = msgpack.unpackb(
                    f.read(), strict_map_key=False, raw=False, use_list=False)
            with gzip.open(Path(filepath).with_suffix('.sl'), 'rb') as f:
                nodes = msgpack.unpackb(f.read(), raw=False)
                self.index.deserialize(nodes)

            return self
        except IOError as e:
            print(f"Error loading from file {filepath}: {e}")

