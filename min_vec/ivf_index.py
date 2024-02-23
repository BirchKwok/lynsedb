"""ivf_index.py: This module contains the implementation of the IVF index,
which is used to speed up the search process."""


class StringPool:
    def __init__(self):
        self.pool = {}  # 从字符串到数字标识符的映射
        self.reverse_pool = {}  # 从数字标识符到字符串的映射
        self.ids = {}  # 多个id共享一个字符串
        self.next_id = 0  # 下一个可用的数字标识符

    def intern(self, string, id=None):
        from spinesUtils.asserts import raise_if_not

        raise_if_not(TypeError, isinstance(id, int) or id is None, "id must be an integer or None")

        if id is not None:
            next_id = id
        else:
            next_id = self.next_id

        if string not in self.pool:
            self.pool[string] = str(next_id)
            self.reverse_pool[str(next_id)] = string

        common_id = self.pool[string]
        self.ids[str(next_id)] = common_id

        self.next_id = next_id + 1

        return self.pool[string]

    def get_id(self, string):
        try:
            return self.pool[string]
        except KeyError:
            raise ValueError(f"'{string}' does not exist in the database.")

    def get_string(self, id):
        id = str(id)
        return self.reverse_pool[self.ids[id]]

    def save_to_disk(self, filename):
        import msgpack

        with open(filename, 'wb') as file:
            data_to_save = {
                'pool': self.pool,
                'reverse_pool': self.reverse_pool,
                'next_id': self.next_id,
                'ids': self.ids
            }
            packed_data = msgpack.packb(data_to_save)
            file.write(packed_data)

    def load_from_disk(self, filename):
        import msgpack

        with open(filename, 'rb') as file:
            data_loaded = msgpack.unpackb(file.read(), strict_map_key=False, raw=False, use_list=False)
            self.pool = data_loaded['pool']
            self.reverse_pool = data_loaded['reverse_pool']
            self.next_id = data_loaded['next_id']
            self.ids = data_loaded['ids']

        return self


class IndexNode:
    def __init__(self):
        self.data = {}  # 使用一个字典来存储数据，键为(primary_key, attachment_msg)元组，值为文件索引

    def add(self, index, field, file_index):
        self.data[(index, field)] = file_index

    def search(self, indices, fields, string_pool):
        # 将fields转换为数字标识符
        fields_set = None if fields is None else ({string_pool.get_id(f) for f in fields} if isinstance(fields, list)
                                                  else {string_pool.get_id(fields)})

        # 如果没有任何搜索条件，直接返回所有记录
        if indices is None and fields is None:
            return [(k[0], string_pool.get_string(k[1]), v) for k, v in self.data.items()]

        filtered_data = []

        if indices is not None and fields is None:
            indices_set = set(indices) if isinstance(indices, list) else {indices}
            for (id, fd), file_index in self.data.items():
                if id in indices_set:
                    filtered_data.append((id, string_pool.get_string(fd), file_index))
            return filtered_data

        if fields is not None and indices is None:
            for (id, fd), file_index in self.data.items():
                if fd in fields_set:
                    filtered_data.append((id, string_pool.get_string(fd), file_index))
            return filtered_data

        if indices is not None and fields is not None:
            indices_set = set(indices) if isinstance(indices, list) else {indices}
            for (id, fd), file_index in self.data.items():
                if id in indices_set and fd in fields_set:
                    filtered_data.append((id, string_pool.get_string(fd), file_index))
            return filtered_data


class HashIndex:
    def __init__(self):
        self.index = {}

    def insert(self, cluster_id, primary_key, attachment_msg, file_index):
        if cluster_id not in self.index:
            self.index[cluster_id] = IndexNode()
        self.index[cluster_id].add(primary_key, attachment_msg, file_index)

    def search(self, cluster_id, indices, fields, string_pool):
        if cluster_id not in self.index:
            return [], [], []
        res = self.index[cluster_id].search(indices, fields, string_pool)
        if len(res) == 0:
            return [], [], []

        idx, f, fidx = zip(*res)
        return idx, f, fidx


class IVFIndex:
    def __init__(self, n_clusters):
        self.index_ivf = {str(i): HashIndex() for i in range(n_clusters)}
        self.string_pool = StringPool()  # 使用StringPool类

    def add_to_cluster(self, cluster_id, index, field, file_index):
        cluster_id = str(cluster_id)
        index = str(index)
        field_id = self.string_pool.intern(str(field))
        if cluster_id not in self.index_ivf:
            raise ValueError(f"Cluster ID {cluster_id} does not exist")
        self.index_ivf[cluster_id].insert(cluster_id, index, field_id, file_index)

    def search(self, cluster_id, indices=None, fields=None):
        cluster_id = str(cluster_id)
        if indices is not None:
            indices = [str(i) for i in indices]
        if fields is not None:
            fields = [str(f) for f in fields]

        if cluster_id not in self.index_ivf:
            return [], [], []

        return self.index_ivf[cluster_id].search(cluster_id, indices=indices, fields=fields,
                                                 string_pool=self.string_pool)

    def save(self, filename):
        import msgpack

        filename = str(filename)
        try:
            with open(filename, 'wb') as f:
                # 保存字符串池
                self.string_pool.save_to_disk('.mvdb'.join(filename.split('.mvdb')[:-1]) + '_string_pool.mvdb')

                # 保存其他数据
                for cluster_id, hash_index in self.index_ivf.items():
                    for key, node in hash_index.index.items():
                        packed_data = msgpack.packb((cluster_id, key, node.data))
                        f.write(len(packed_data).to_bytes(4, byteorder='big'))
                        f.write(packed_data)
        except IOError as e:
            print(f"An error occurred while saving to {filename}: {e}")

    def load(self, filename):
        import msgpack

        filename = str(filename)
        try:
            with open(filename, 'rb') as f:
                # 加载字符串池
                self.string_pool.load_from_disk('.mvdb'.join(filename.split('.mvdb')[:-1]) + '_string_pool.mvdb')

                # 加载其他数据
                while True:
                    length_bytes = f.read(4)
                    if not length_bytes:
                        break
                    length = int.from_bytes(length_bytes, byteorder='big')
                    data = f.read(length)
                    cluster_id, key, node_data = msgpack.unpackb(data, use_list=False, raw=False, strict_map_key=False)

                    if cluster_id not in self.index_ivf:
                        self.index_ivf[cluster_id] = HashIndex()
                    self.index_ivf[cluster_id].index[key] = IndexNode()
                    self.index_ivf[cluster_id].index[key].data = node_data
        except IOError as e:
            print(f"An error occurred while loading from {filename}: {e}")

        return self
