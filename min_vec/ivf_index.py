import struct


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.indices = []
        self.fields = []
        self.paths = []


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, cluster_id, index, field, path):
        index = str(index)
        field = str(field)
        path = str(path)
        node = self.root
        for char in cluster_id:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.indices.append(index)
        node.fields.append(field)
        node.paths.append(path)

    def search(self, cluster_id, indices=None, fields=None):
        indices = [str(index) for index in indices] if indices is not None else None
        fields = [str(field) for field in fields] if fields is not None else None

        node = self._search_node(cluster_id)
        if node is None or not node.is_end_of_word:
            return None

        filtered_indices, filtered_fields, filtered_paths = [], [], []

        for idx, field, path in zip(node.indices, node.fields, node.paths):
            index_condition = (indices is None) or (idx in indices)
            field_condition = (fields is None) or (field in fields)

            if index_condition and field_condition:
                filtered_indices.append(idx)
                filtered_fields.append(field)
                filtered_paths.append(path)

        return filtered_indices, filtered_fields, filtered_paths

    def _search_node(self, cluster_id):
        node = self.root
        for char in cluster_id:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def save(self, file):
        self._save_node(file, self.root)

    def _save_node(self, f, node):
        if node is None:
            f.write(struct.pack('?', False))
            return

        f.write(struct.pack('?', True))
        f.write(struct.pack('I', len(node.indices)))
        for index in node.indices:
            encoded_index = index.encode('utf-8')
            f.write(struct.pack('I', len(encoded_index)))
            f.write(encoded_index)

        f.write(struct.pack('I', len(node.fields)))
        for field in node.fields:
            encoded_field = field.encode('utf-8')
            f.write(struct.pack('I', len(encoded_field)))
            f.write(encoded_field)

        f.write(struct.pack('I', len(node.paths)))
        for path in node.paths:
            encoded_path = path.encode('utf-8')
            f.write(struct.pack('I', len(encoded_path)))
            f.write(encoded_path)

        f.write(struct.pack('I', len(node.children)))
        for char, child_node in node.children.items():
            f.write(struct.pack('c', char.encode('utf-8')))
            self._save_node(f, child_node)

    def load(self, file):
        self.root = self._load_node(file)

    def _load_node(self, f):
        exists = struct.unpack('?', f.read(1))[0]
        if not exists:
            return None

        node = TrieNode()
        node.is_end_of_word = True

        indices_length = struct.unpack('I', f.read(4))[0]
        node.indices = [f.read(struct.unpack('I', f.read(4))[0]).decode('utf-8') for _ in range(indices_length)]

        fields_length = struct.unpack('I', f.read(4))[0]
        node.fields = [f.read(struct.unpack('I', f.read(4))[0]).decode('utf-8') for _ in range(fields_length)]

        paths_length = struct.unpack('I', f.read(4))[0]
        node.paths = [f.read(struct.unpack('I', f.read(4))[0]).decode('utf-8') for _ in range(paths_length)]

        children_length = struct.unpack('I', f.read(4))[0]
        for _ in range(children_length):
            char = struct.unpack('c', f.read(1))[0].decode('utf-8')
            node.children[char] = self._load_node(f)

        return node


class CompactIVFIndex:
    def __init__(self, n_clusters):
        self.ivf_index = {i: Trie() for i in range(n_clusters)}

    def add_to_cluster(self, cluster_id, index, field, path):
        if cluster_id not in self.ivf_index:
            raise ValueError(f"Cluster ID {cluster_id} does not exist")
        self.ivf_index[cluster_id].insert(str(cluster_id), index, field, path)

    def search(self, cluster_id, indices=None, fields=None):
        if cluster_id not in self.ivf_index:
            return None

        trie = self.ivf_index[cluster_id]
        return trie.search(str(cluster_id), indices, fields)

    def save(self, filename):
        with open(filename, 'wb') as f:
            for cluster_id, trie in self.ivf_index.items():
                f.write(struct.pack('i', cluster_id))
                trie.save(f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.ivf_index = {}
            while True:
                bytes = f.read(4)
                if len(bytes) < 4:
                    break

                cluster_id = struct.unpack('i', bytes)[0]
                trie = Trie()
                trie.load(f)
                self.ivf_index[cluster_id] = trie

        return self
