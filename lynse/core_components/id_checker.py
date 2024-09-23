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

    def filter_ids(self, matcher):
        """
        Filter the IDs based on the matcher.

        Parameters:
            matcher: Matcher
                The matcher to use for filtering.

        Returns:
            List[int]: The filtered IDs.
        """
        if matcher.matcher.name == "MatchField":
            comparator_name = matcher.matcher.comparator.__name__
            value = matcher.matcher.value

            if comparator_name == 'eq':
                return [value] if value in self.ids else []
            elif comparator_name == 'ne':
                return list(self.ids - BitMap([value]))
            elif comparator_name == 'le':
                return [id for id in self.ids if id <= value]
            elif comparator_name == 'ge':
                return [id for id in self.ids if id >= value]
            elif comparator_name == 'lt':
                return [id for id in self.ids if id < value]
            elif comparator_name == 'gt':
                return [id for id in self.ids if id > value]
        elif matcher.matcher.name == "MatchRange":
            start, end, inclusive = matcher.matcher.start, matcher.matcher.end, matcher.matcher.inclusive
            if inclusive is True:
                return [id for id in self.ids if start <= id <= end]
            elif inclusive is False:
                return [id for id in self.ids if start < id < end]
            elif inclusive == "left":
                return [id for id in self.ids if start <= id < end]
            elif inclusive == "right":
                return [id for id in self.ids if start < id <= end]

        return [id for id in self.ids if matcher.evaluate(data=None, external_id=id)]

    def retrieve_all_ids(self, return_set=False):
        return list(self.ids) if not return_set else set(self.ids)
