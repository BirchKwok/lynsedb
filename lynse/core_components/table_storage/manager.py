from storage import LyStorage
from pathlib import Path
from typing import Dict, List, Optional, Union


class StorageManager:
    """
    A manager for managing multiple LyStorage instances.

    Usage:
        manager = StorageManager(base_path="path/to/base/directory")
        storage = manager.create_storage("my_storage")
        storage.write(data)

        storage = manager.get_storage("my_storage")
        storage.read()

        manager.delete_storage("my_storage")
    """
    def __init__(self, base_path: Union[str, Path], lazy_load: bool = True, overwrite: bool = False, merge_threshold: int = 1000):
        """
        Initializes a new instance of the StorageManager class.

        Parameters:
            base_path (str or pathlike): The base path to store the storage.
            lazy_load (bool): Whether to lazy load the storage.
            overwrite (bool): Whether to overwrite the storage if it already exists.
            merge_threshold (int): The threshold to merge the storage.
        """
        self.base_path = Path(base_path)
        self.storages: Dict[str, LyStorage] = {}
        self.lazy_load = lazy_load
        self.overwrite = overwrite
        self.merge_threshold = merge_threshold

        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)

    def create_storage(self, name: str, data_path: Optional[str] = None) -> LyStorage:
        """
        Create a new LyStorage instance and add it to the manager.

        Parameters:
            name (str): The name of the storage.
            data_path (str): The path to the data. If not provided, the base path / name will be used.

        Returns:
            The created LyStorage instance.
        """
        if name in self.storages:
            raise ValueError(f"Storage '{name}' already exists.")

        if data_path is None:
            data_path = str(self.base_path / name)

        storage = LyStorage(data_path, lazy_load=self.lazy_load, overwrite=self.overwrite, merge_threshold=self.merge_threshold)
        self.storages[name] = storage
        return storage

    def get_storage(self, name: str) -> LyStorage:
        """
        Get an existing LyStorage instance.

        Parameters:
            name (str): The name of the storage.

        Returns:
            The existing LyStorage instance.
        """
        if name not in self.storages:
            raise KeyError(f"Storage '{name}' does not exist.")
        return self.storages[name]

    def delete_storage(self, name: str):
        """
        Delete the specified LyStorage instance and its data.

        Parameters:
            name (str): The name of the storage.
        """
        if name not in self.storages:
            raise KeyError(f"Storage '{name}' does not exist.")
        storage = self.storages.pop(name)
        storage.delete()

    def list_storages(self) -> List[str]:
        """
        List all managed storage names.

        Returns:
            The list of storage names.
        """
        return list(self.storages.keys())
