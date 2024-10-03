from pyroaring import BitMap


class IDChecker:
    def __init__(self, filepath):
        """
        Initialize the IDChecker class.

        Parameters:
            filepath: str
                The file path to the storage.
        """
        self.ids = BitMap()
        self.filepath = filepath
        self.load(filepath)

    def sync_load(self):
        """
        Synchronously load the IDChecker from a file.
        """
        try:
            with open(self.filepath, 'rb') as file:
                self.ids = BitMap.deserialize(file.read())
        except FileNotFoundError:
            pass

    def is_empty(self):
        """
        Check if the IDChecker is empty.

        Returns:
            bool
                True if the IDChecker is empty, False otherwise.
        """
        return len(self.ids) == 0

    def add(self, items):
        """
        Add items to the IDChecker.

        Parameters:
            items: int or list
                The items to add to the IDChecker.
        """
        if isinstance(items, int):
            self.ids.add(items)
        else:
            self.ids.update(items)

    def drop(self, items):
        """
        Drop items from the IDChecker.

        Parameters:
            items: int or list
                The items to drop from the IDChecker.
        """
        if isinstance(items, int):
            if items in self.ids:
                self.ids.discard(items)
        else:
            for item in items:
                if item in self.ids:
                    self.ids.discard(item)

    def __contains__(self, item):
        """
        Check if an item is in the IDChecker.

        Parameters:
            item: int
                The item to check.

        Returns:
            bool
                True if the item is in the IDChecker, False otherwise.
        """
        return item in self.ids

    def save(self):
        """
        Write the IDChecker to a file.

        """
        with open(self.filepath, 'wb') as file:
            file.write(self.ids.serialize())

    def load(self, filepath):
        """
        Read the IDChecker from a file.

        Parameters:
            filepath: str
                The file path to the storage.
        """
        try:
            with open(filepath, 'rb') as file:
                self.ids = BitMap.deserialize(file.read())
        except FileNotFoundError:
            self.ids = BitMap()

    def find_max_value(self):
        """
        Find the maximum value in the IDChecker.

        Returns:
            int
                The maximum value in the IDChecker, or -1 if the IDChecker is empty.
        """
        return max(self.ids) if self.ids else -1

    def find_last_value(self):
        """
        Find the last value in the IDChecker.

        Returns:
            int
                The last value in the IDChecker, or -1 if the IDChecker is empty.
        """
        # find the last added value
        return self.ids[-1] if self.ids else -1
