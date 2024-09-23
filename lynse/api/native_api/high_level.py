import json
import shutil
import uuid
from functools import partial
from pathlib import Path
from typing import Union

from spinesUtils.asserts import ParameterTypeAssert

from ...api import logger
from ...utils.utils import unavailable_if_deleted


class _Register:
    """
    A class for registering the collections to local database path.
    """

    def __init__(self, root_path: str) -> None:
        self.root_path = Path(root_path)

    def register_collection(self, collection: str, **kwargs):
        """
        Save the collection name to the local database path.

        Parameters:
            collection (str): The name of the collection.
        """
        if not (self.root_path / 'collections.json').exists():
            with open(self.root_path / 'collections.json', 'w') as f:
                json.dump({collection: kwargs}, f)
        else:
            collections = self.get_collections_details()

            if collection not in collections:
                collections[collection] = kwargs

                with open(self.root_path / 'collections.json', 'w') as f:
                    json.dump(collections, f)

    def deregister_collection(self, collection: str):
        """
        Delete the collection name from the local database path.

        Parameters:
            collection (str): The name of the collection.
        """
        if not (self.root_path / 'collections.json').exists():
            return

        collections = self.get_collections_details()

        if collection in collections:
            del collections[collection]

            with open(self.root_path / 'collections.json', 'w') as f:
                json.dump(collections, f)

    def get_collections_details(self) -> dict:
        """
        Show the collections in the database.

        Returns:
            list: The list of collections in the database.
        """
        if not (self.root_path / 'collections.json').exists():
            return {}

        with open(self.root_path / 'collections.json', 'r') as f:
            collections = json.load(f)

        for collection in list(collections.keys()):
            if not (self.root_path / collection).exists():
                del collections[collection]
                with open(self.root_path / 'collections.json', 'w') as f:
                    json.dump(collections, f)

        return collections

    def show_collections(self) -> list:
        """
        Show the collections in the database.

        Returns:
            list: The list of collections in the database.
        """
        return list(self.get_collections_details().keys())

    def show_collections_details(self):
        """
        Show the collections in the database.

        Returns:
            list: The list of collections in the database.
        """
        details = self.get_collections_details()

        # hide the database path
        for collection in details:
            if "database_path" in details[collection]:
                del details[collection]["database_path"]

        try:
            import pandas as pd
            details = pd.DataFrame(details).T
        except ImportError:
            ...

        return details

    @ParameterTypeAssert({
        'collection': str,
        'description': (None, str, int, float, bool)
    })
    def update_description(self, collection: str, description: Union[None, str, int, float, bool]):
        """
        Update the description of the collection.

        Parameters:
            collection (str): The name of the collection.
            description (None or str or int or float or bool): The description of the collection.
        """
        collections = self.get_collections_details()

        if collection in collections:
            collections[collection]['description'] = description

            with open(self.root_path / 'collections.json', 'w') as f:
                json.dump(collections, f)

    def __contains__(self, item):
        return item in self.get_collections_details()


class LocalClient:
    """
    A singleton class for the local LynseDB client.
    Using the LocalClient class, users can create and access database instances,
    as well as operate on the collections within them (Low-Level API).
    This class is thread-safe only. Using it in multiple processes will result in a race condition.
    """
    _instance = None
    _last_root_path = None

    def __new__(cls, root_path: Union[Path, str]):
        """
        Create a new instance or return the existing instance of the class.

        Parameters:
            root_path (Path or str): The root path of the database.

        Returns:
            LocalClient (LocalClient): The instance of the class.
        """
        if cls._instance is not None and cls._last_root_path != root_path:
            cls._instance = None
            cls._last_root_path = root_path

        if cls._instance is None:
            cls._instance = super(LocalClient, cls).__new__(cls)
            cls._instance._init(root_path)

            cls._last_root_path = root_path

        Path(cls._last_root_path).mkdir(parents=True, exist_ok=True)

        if not (Path(cls._last_root_path) / '.fingerprint').exists():
            # create a database fingerprint
            with open(Path(cls._last_root_path) / '.fingerprint', 'w') as f:
                f.write(uuid.uuid4().hex)

        return cls._instance

    def _init(self, root_path: Union[Path, str]):
        """
        Initialize the vector database.
        """
        self._root_path = Path(root_path).absolute()

        self._register = _Register(root_path)
        self._collections = {}
        self.STATUS = 'INITIALIZED'

    def require_collection(
            self,
            collection: str,
            dim: int = None,
            chunk_size: int = 100_000,
            dtypes: str = 'float32',
            use_cache: bool = True,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            drop_if_exists: bool = False,
            description: str = None,
            cache_chunks: int = 20
    ):
        """Create or load a collection in the database.

        Parameters:
            collection (str): The name of the collection.
            dim (int): Dimension of the vectors. Default is None.
                When creating a new collection, the dimension of the vectors must be specified.
                When loading an existing collection, the dimension of the vectors is automatically loaded.
            chunk_size (int): The size of each data chunk. Default is 100_000.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            use_cache (bool): Whether to use cache for search. Default is True.
            n_threads (int): The number of threads to use for parallel processing. Default is 10.
            warm_up (bool): Whether to warm up the database. Default is False.
            drop_if_exists (bool): Whether to drop the collection if it already exists. Default is False.
            description (str): A description of the collection. Default is None.
                The description is limited to 500 characters.
            cache_chunks (int): The number of chunks to cache. Default is 20.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        from ...api.native_api.low_level import ExclusiveDB

        collection_path = self._root_path / collection

        if description is not None and not isinstance(description, str):
            raise ValueError('Description must be a string')
        elif description is not None and len(description) > 500:
            raise ValueError('Description must be less than 500 characters')

        if collection in self._register:
            if drop_if_exists:
                self.drop_collection(collection)
                logger.info(f"Collection '{collection}' already exists. Dropped.")
            else:
                collection_details = self._register.get_collections_details()[collection]
                dim = collection_details['dim']
                logger.info(f"Collection '{collection}' already exists. Loaded.")

        if chunk_size <= 1:
            raise ValueError('chunk_size must be greater than 1')

        self._collections[collection] = ExclusiveDB(
            dim=dim, database_path=collection_path.as_posix(), chunk_size=chunk_size, dtypes=dtypes,
            use_cache=use_cache, n_threads=n_threads,
            warm_up=warm_up, cache_chunks=cache_chunks
        )

        self._register.register_collection(
            collection, dim=dim, database_path=collection_path.as_posix(), chunk_size=chunk_size, dtypes=dtypes,
            use_cache=use_cache, n_threads=n_threads,
            warm_up=warm_up, description=description, cache_chunks=cache_chunks
        )

        self._collections[collection].update_description = partial(self.update_collection_description, collection)

        return self._collections[collection]

    @unavailable_if_deleted
    def get_collection(self, collection: str, cache_chunks: int = 20, warm_up: bool = True):
        """
        Get a collection from the database.

        Parameters:
            collection (str): The name of the collection to get.
            cache_chunks (int): The number of chunks to cache. Default is 20.
            warm_up (bool): Whether to warm up the database. Default is True.

        Returns:
            (ExclusiveDB): The collection.
        """
        from ...api.native_api.low_level import ExclusiveDB

        if collection not in self._collections:
            if collection not in self._register:
                raise ValueError(f"Collection '{collection}' does not exist.")

            params = self._register.get_collections_details()[collection]
            params.update({'cache_chunks': cache_chunks, 'warm_up': warm_up})
            if 'description' in params:
                del params['description']
            self._collections[collection] = ExclusiveDB(**params)

        return self._collections[collection]

    @unavailable_if_deleted
    def copy_collection(self, deep=False):
        """
        Copy the collection.

        Parameters:
            deep (bool): Whether to make a deep copy. Default is False.
                Which means share the data with the original collection.

        Returns:
            (ExclusiveDB): The copied collection.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError("This method is not implemented yet.")

    @unavailable_if_deleted
    def show_collections(self):
        """
        Show the collections in the database.

        Returns:
            List: The list of collections in the database.
        """
        return self._register.show_collections()

    @unavailable_if_deleted
    def drop_collection(self, collection: str):
        """
        Delete a collection from the database.

        Parameters:
            collection (str): The name of the collection to delete.

        Returns:
            None
        """

        if collection in self._collections:
            self._collections[collection].delete()
            del self._collections[collection]
            self._register.deregister_collection(collection)
        else:
            try:
                _temp_collection = self.get_collection(collection)
                self.drop_collection(collection)
            except ValueError:
                pass

        if (self._root_path / "collections.json").exists():
            with open(self._root_path / "collections.json", "r") as f:
                collections = json.load(f)
                if collection in collections:
                    del collections[collection]

            with open(self._root_path / "collections.json", "w") as f:
                json.dump(collections, f)

    def drop_database(self):
        """
        Delete the database.

        Returns:
            None
        """
        if self.STATUS == 'DELETED':
            return

        if self._root_path.exists():
            shutil.rmtree(self._root_path)

        LocalClient._instance = None
        self.STATUS = 'DELETED'

    def update_collection_description(self, collection: str, description: Union[None, str, int, float, bool]):
        """
        Update the description of the collection.

        Parameters:
            collection (str): The name of the collection.
            description (None or str or int or float or bool): The description of the collection.

        Returns:
            None
        """
        self._register.update_description(collection, description)

    def show_collections_details(self):
        """
        Show the collections in the database.

        Returns:
            (Dict or DataFrame): The details of the collections in the database.
        """
        return self._register.show_collections_details()

    def database_exists(self):
        """
        Check if the database exists.

        Returns:
            (Bool): Whether the database exists.
        """
        return self._root_path.exists()

    @property
    def root_path(self):
        """
        Get the root path of the database.

        Returns:
            (str): The root path of the database.
        """
        return self._root_path.as_posix()

    def __repr__(self):
        return (f'{self.__class__.__name__.replace("Client", "DatabaseInstance")}(name={self._root_path.name},'
                f' exists={self.database_exists()})')

    def __str__(self):
        return self.__repr__()
