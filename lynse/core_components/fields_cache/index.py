import os
import msgpack


class TrieNode:
    __slots__ = ['children', 'is_end_of_word', 'ids']

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.ids = set()


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, key, record_id):
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.ids.add(record_id)

    def search(self, key):
        if isinstance(key, (list, tuple)):
            return self._batch_search(map(str, key))
        else:
            return self._search_single(str(key))

    def _batch_search(self, keys):
        results = set()
        current_nodes = {self.root: set(keys)}

        while current_nodes:
            next_nodes = {}
            for node, prefixes in current_nodes.items():
                for prefix in prefixes:
                    if not prefix:
                        if node.is_end_of_word:
                            results.update(node.ids)
                    else:
                        char = prefix[0]
                        if char in node.children:
                            child = node.children[char]
                            next_prefix = prefix[1:]
                            if child not in next_nodes:
                                next_nodes[child] = set()
                            next_nodes[child].add(next_prefix)
            current_nodes = next_nodes

        return results

    def starts_with(self, prefix):
        def collect_all_words(node):
            stack = [node]
            ids = set()
            while stack:
                node = stack.pop()
                if node.is_end_of_word:
                    ids.update(node.ids)
                for child in node.children.values():
                    stack.append(child)
            return ids

        node = self.root
        for char in prefix:
            if char not in node.children:
                return set()
            node = node.children[char]
        return collect_all_words(node)

    def serialize(self):
        stack = [(self.root, "")]
        serialized = {}
        while stack:
            node, prefix = stack.pop()
            serialized[prefix] = {
                "children": list(node.children.keys()),
                "is_end_of_word": node.is_end_of_word,
                "ids": list(node.ids)
            }
            for char, child in node.children.items():
                stack.append((child, prefix + char))
        return serialized

    def deserialize(self, data):
        root = TrieNode()
        nodes = {"": root}
        for prefix, node_data in data.items():
            node = nodes[prefix]
            node.is_end_of_word = node_data["is_end_of_word"]
            node.ids = set(node_data["ids"])
            for char in node_data["children"]:
                child_prefix = prefix + char
                child_node = TrieNode()
                node.children[char] = child_node
                nodes[child_prefix] = child_node
        self.root = root


class BPlusTreeNode:
    __slots__ = ['is_leaf', 'keys', 'children']

    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []


class BPlusTree:
    def __init__(self, t=3):
        self.root = BPlusTreeNode(is_leaf=True)
        self.t = t

    def insert(self, key, record_id):
        root = self.root
        if len(root.keys) == (2 * self.t - 1):
            temp = BPlusTreeNode()
            self.root = temp
            temp.children.insert(0, root)
            self._split_child(temp, 0)
        self._insert_non_full(self.root, key, record_id)

    def _insert_non_full(self, node, key, record_id):
        if node.is_leaf:
            idx = len(node.keys) - 1
            while idx >= 0 and key < node.keys[idx][0]:
                idx -= 1
            node.keys.insert(idx + 1, (key, record_id))
        else:
            idx = len(node.keys) - 1
            while idx >= 0 and key < node.keys[idx][0]:
                idx -= 1
            idx += 1
            if len(node.children[idx].keys) == (2 * self.t - 1):
                self._split_child(node, idx)
                if key > node.keys[idx][0]:
                    idx += 1
            self._insert_non_full(node.children[idx], key, record_id)

    def _split_child(self, parent, idx):
        t = self.t
        child = parent.children[idx]
        new_node = BPlusTreeNode(is_leaf=child.is_leaf)
        parent.keys.insert(idx, child.keys[t - 1])
        parent.children.insert(idx + 1, new_node)

        new_node.keys = child.keys[t:(2 * t - 1)]
        child.keys = child.keys[0:(t - 1)]

        if not child.is_leaf:
            new_node.children = child.children[t:(2 * t)]
            child.children = child.children[0:t]

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        idx = 0
        while idx < len(node.keys) and key > node.keys[idx][0]:
            idx += 1
        if idx < len(node.keys) and key == node.keys[idx][0]:
            return {node.keys[idx][1]}
        if node.is_leaf:
            return set()
        return self._search(node.children[idx], key)

    def range_search(self, start_key, end_key):
        result = set()
        self._range_search(self.root, start_key, end_key, result)
        return result

    def _range_search(self, node, start_key, end_key, result):
        idx = 0
        while idx < len(node.keys) and (start_key is not None and node.keys[idx][0] < start_key):
            idx += 1

        if node.is_leaf:
            while idx < len(node.keys) and (end_key is None or node.keys[idx][0] <= end_key):
                result.add(node.keys[idx][1])
                idx += 1
        else:
            while idx < len(node.keys) and (end_key is None or node.keys[idx][0] <= end_key):
                self._range_search(node.children[idx], start_key, end_key, result)
                result.add(node.keys[idx][1])
                idx += 1
            if idx < len(node.children):
                self._range_search(node.children[idx], start_key, end_key, result)


class Index:
    def __init__(self):
        self.indices = {}
        self.index_schema = {}

    def add_index(self, field, field_type, rebuild_if_exists=False):
        if field_type not in [int, float, str]:
            raise ValueError("Unsupported field type. Supported types are int, float, and str.")
        if field in self.indices:
            if rebuild_if_exists:
                self.remove_index(field)
            else:
                return

        if field_type == str:
            self.indices[field] = Trie()
        else:
            self.indices[field] = BPlusTree()

        self.index_schema[field] = field_type

    def insert(self, record, record_id):
        for field, index in self.indices.items():
            field_value = record.get(field)
            if field_value is not None:
                index.insert(field_value, record_id)

    def search(self, field, value):
        if field not in self.indices:
            raise ValueError(f"No index for field '{field}'")
        return self.indices[field].search(value)

    def range_search(self, field, start_value, end_value):
        if field not in self.indices:
            raise ValueError(f"No index for field '{field}'")
        if isinstance(self.indices[field], Trie):
            raise ValueError("Range search is not supported for Trie index.")
        return self.indices[field].range_search(start_value, end_value)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            packed_indices = msgpack.packb((self._serialize_indices(),
                                            {k: str(v.__name__) for k, v in self.index_schema.items()}),
                                           use_bin_type=True)
            f.write(packed_indices)

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                packed_indices = f.read()
                indices, index_schema = msgpack.unpackb(packed_indices, raw=False)
                self._deserialize_indices(indices)
                self.index_schema = {k: eval(v) for k, v in index_schema.items()}
        return self

    def _serialize_indices(self):
        serialized = {}
        for field, index in self.indices.items():
            if isinstance(index, Trie):
                serialized[field] = {
                    "type": "trie",
                    "data": index.serialize()
                }
            elif isinstance(index, BPlusTree):
                serialized[field] = {
                    "type": "bplustree",
                    "data": self._serialize_bplustree(index)
                }
        return serialized

    def _deserialize_indices(self, data):
        for field, index_data in data.items():
            if index_data["type"] == "trie":
                index = Trie()
                index.deserialize(index_data["data"])
            else:
                index = BPlusTree()
                self._deserialize_bplustree(index, index_data["data"])
            self.indices[field] = index

    @staticmethod
    def _serialize_bplustree(bplustree):
        serialized_nodes = []

        def serialize_node(node):
            node_data = {
                "is_leaf": node.is_leaf,
                "keys": node.keys,
                "children": []
            }
            if not node.is_leaf:
                for child in node.children:
                    node_data["children"].append(serialize_node(child))
            return node_data

        serialized_tree = serialize_node(bplustree.root)
        serialized_nodes.append(serialized_tree)
        return serialized_nodes

    @staticmethod
    def _deserialize_bplustree(bplustree, data):
        def deserialize_node(node_data):
            node = BPlusTreeNode(is_leaf=node_data["is_leaf"])
            node.keys = node_data["keys"]
            if not node.is_leaf:
                node.children = [deserialize_node(child) for child in node_data["children"]]
            return node

        bplustree.root = deserialize_node(data[0])

    def remove_index(self, field):
        if field in self.indices:
            del self.indices[field]
        if field in self.index_schema:
            del self.index_schema[field]
