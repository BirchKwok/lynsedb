import numpy as np
from pyroaring import BitMap
from spinesUtils.asserts import raise_if


class IDMapperEntry:
    def __init__(self, start_id, end_id):
        self.start_id = start_id
        self.end_id = end_id
        self.id_range = BitMap(range(start_id, end_id + 1))

    def isin(self, array: np.ndarray) -> np.ndarray:
        array_bitmap = BitMap(array)
        intersection = array_bitmap & self.id_range
        return np.isin(array, list(intersection))

    def generate_ids(self, as_range=False):
        return np.arange(self.start_id, self.end_id + 1) if not as_range \
            else range(self.start_id, self.end_id + 1)


class IDMapper:
    def __init__(self):
        self.id_map = {}

    def add_entry(self, filename, start_id, end_id):
        self.id_map[filename] = (start_id, end_id)

    def get_id_range(self, filename):
        return self.id_map.get(filename)

    def remove_entry(self, filename):
        self.id_map.pop(filename)

    def load(self, filename):
        import msgpack

        with open(filename, 'rb') as f:
            self.id_map = msgpack.unpackb(f.read(), use_list=False)

    def save(self, filename):
        import msgpack

        with open(filename, 'wb') as f:
            f.write(msgpack.packb(self.id_map))

    def __getitem__(self, item):
        raise_if(ValueError, not isinstance(item, str), "item must be a string")

        return IDMapperEntry(*self.get_id_range(item))
