import msgpack


class IndexNode:
    def __init__(self):
        self.data = {}  # 使用一个字典来存储数据，键为(primary_key, attachment_msg)元组，值为路径

    def add(self, index, field, path):
        self.data[(index, field)] = path

    def search(self, indices=None, fields=None):
        # 如果没有任何搜索条件，直接返回所有记录
        if indices is None and fields is None:
            return [(k[0], k[1], v) for k, v in self.data.items()]

        filtered_data = []

        if indices is not None and fields is None:
            indices_set = set(indices) if isinstance(indices, list) else {indices}
            for (id, fd), path in self.data.items():
                if id in indices_set:
                    filtered_data.append((id, fd, path))
            return filtered_data

        if fields is not None and indices is None:
            fields_set = set(fields) if isinstance(fields, list) else {fields}
            for (id, fd), path in self.data.items():
                if fd in fields_set:
                    filtered_data.append((id, fd, path))
            return filtered_data

        if indices is not None and fields is not None:
            indices_set = set(indices) if isinstance(indices, list) else {indices}
            fields_set = set(fields) if isinstance(fields, list) else {fields}
            for (id, fd), path in self.data.items():
                if id in indices_set and fd in fields_set:
                    filtered_data.append((id, fd, path))
            return filtered_data


class HashIndex:
    def __init__(self):
        self.index = {}

    def insert(self, cluster_id, primary_key, attachment_msg, path):
        if cluster_id not in self.index:
            self.index[cluster_id] = IndexNode()
        self.index[cluster_id].add(primary_key, attachment_msg, path)

    def search(self, cluster_id, indices=None, fields=None):
        if cluster_id not in self.index:
            return [], [], []
        res = self.index[cluster_id].search(indices, fields)
        if len(res) == 0:
            return [], [], []

        idx, f, p = zip(*res)
        return idx, f, p


class CompactIVFIndex:
    def __init__(self, n_clusters):
        self.index_ivf = {str(i): HashIndex() for i in range(n_clusters)}

    def add_to_cluster(self, cluster_id, index, field, path):
        cluster_id = str(cluster_id)
        index = str(index)
        field = str(field)
        path = str(path)
        if cluster_id not in self.index_ivf:
            raise ValueError(f"Cluster ID {cluster_id} does not exist")
        self.index_ivf[cluster_id].insert(cluster_id, index, field, path)

    def search(self, cluster_id, indices=None, fields=None):
        cluster_id = str(cluster_id)
        if indices is not None:
            indices = [str(i) for i in indices]
        if fields is not None:
            fields = [str(f) for f in fields]

        if cluster_id not in self.index_ivf:
            return [], [], []

        return self.index_ivf[cluster_id].search(cluster_id, indices=indices, fields=fields)

    def save(self, filename):
        with open(filename, 'wb') as f:
            for cluster_id, index_hash in self.index_ivf.items():
                for key, node in index_hash.index.items():
                    packed_data = msgpack.packb((cluster_id, key, node.data))
                    f.write(len(packed_data).to_bytes(4, byteorder='big'))
                    f.write(packed_data)

    def load(self, filename):
        with open(filename, 'rb') as f:
            while True:
                length_bytes = f.read(4)
                if not length_bytes:
                    break
                length = int.from_bytes(length_bytes, byteorder='big')
                data = f.read(length)
                cluster_id, key, node_data = msgpack.unpackb(data, use_list=False, strict_map_key=False, raw=False)

                if cluster_id not in self.index_ivf:
                    self.index_ivf[cluster_id] = HashIndex()
                self.index_ivf[cluster_id].index[key] = IndexNode()
                self.index_ivf[cluster_id].index[key].data = node_data

        return self
