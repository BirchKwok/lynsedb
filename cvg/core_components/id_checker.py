from pyroaring import BitMap


class IDChecker:
    def __init__(self):
        self.ids = BitMap()

    def add(self, items):
        if isinstance(items, int):
            self.ids.add(items)
        else:
            self.ids.update(items)

    def concat(self, another_id_checker):
        if not isinstance(another_id_checker, IDChecker):
            raise ValueError("Expected another_id_checker to be an instance of IDChecker")

        self.ids.update(another_id_checker.ids)

    def drop(self, items):
        if isinstance(items, int):
            if items in self.ids:
                self.ids.discard(items)
        else:
            for item in items:
                if item in self.ids:
                    self.ids.discard(item)

    def __contains__(self, item):
        return item in self.ids

    def to_file(self, filepath):
        with open(filepath, 'wb') as file:
            file.write(self.ids.serialize())

    def from_file(self, filepath):
        try:
            with open(filepath, 'rb') as file:
                self.ids = BitMap.deserialize(file.read())
        except FileNotFoundError:
            self.ids = BitMap()

    def find_max_value(self):
        return max(self.ids) if self.ids else -1
