"""fields_mapper.py: this file contains the FieldsMapper class. It is used to map the fields of the data."""


class FieldsMapper:
    def __init__(self):
        self.fields_int_mapper = {}  # id -> field
        self.fields_str_mapper = {}  # field -> id
        self.fields_values = set()
        self.last_id = 0  # the initial id, the first id will be 0

    def _get_next_id(self):
        current_id = self.last_id
        self.last_id += 1
        return current_id

    def _single_encode(self, single_data):
        if single_data not in self.fields_values:
            self.fields_values.add(single_data)
            self.fields_int_mapper[self.last_id] = single_data
            self.fields_str_mapper[single_data] = self.last_id
            return self._get_next_id()
        else:
            return self.fields_str_mapper[single_data]

    def encode(self, data):
        if isinstance(data, str):
            return self._single_encode(data)
        return [self._single_encode(single_data) for single_data in data]

    def _single_decode(self, id):
        try:
            return self.fields_int_mapper[id]
        except KeyError:
            raise ValueError(f"Invalid id: {id}")

    def decode(self, ids):
        if len(set(ids)) == 1:
            _t = self._single_decode(ids[0])
            return [_t for _ in ids]

        if isinstance(ids, int):
            return self._single_decode(ids)
        return [self._single_decode(id) for id in ids]

    def save(self, filepath):
        import msgpack
        try:
            with open(filepath, 'wb') as f:
                f.write(msgpack.packb([self.fields_int_mapper, self.fields_str_mapper, self.last_id]))
        except IOError as e:
            print(f"Error saving to file {filepath}: {e}")

    def load(self, filepath):
        import msgpack
        try:
            with open(filepath, 'rb') as f:
                self.fields_int_mapper, self.fields_str_mapper, self.last_id = msgpack.unpackb(f.read(), strict_map_key=False,
                                                                                               raw=False, use_list=False)
            self.fields_values = set(self.fields_str_mapper.keys())
        except IOError as e:
            print(f"Error loading from file {filepath}: {e}")
        return self
