import json
from functools import partial
from pathlib import Path
from typing import Union

import pandas as pd
from spinesUtils.asserts import ParameterTypeAssert

from min_vec.api.native_api.low_level import StandaloneMinVectorDB
from min_vec.api import logger
from min_vec.utils.utils import unavailable_if_deleted


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

    def show_collections_details(self) -> pd.DataFrame:
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

        details = pd.DataFrame(details).T
        details.index.name = 'collections'
        return details

    def update_description(self, collection: str, description: str):
        """
        Update the description of the collection.

        Parameters:
            collection (str): The name of the collection.
            description (str): The description of the collection.
        """
        collections = self.get_collections_details()

        if collection in collections:
            collections[collection]['description'] = description

            with open(self.root_path / 'collections.json', 'w') as f:
                json.dump(collections, f)

    def __contains__(self, item):
        return item in self.get_collections_details()


class MinVectorDBLocalClient:
    """
    A singleton class for the local MinVectorDB client.
    """
    _instance = None
    _last_root_path = None

    def __new__(cls, root_path: Union[Path, str]):
        """
        Create a new instance or return the existing instance of the class.
        """
        if cls._instance is not None and cls._last_root_path != root_path:
            cls._instance = None
            cls._last_root_path = root_path

        if cls._instance is None:
            cls._instance = super(MinVectorDBLocalClient, cls).__new__(cls)
            cls._instance._init(root_path)

            cls._last_root_path = root_path

        return cls._instance

    @ParameterTypeAssert({
        'root_path': str
    }, func_name='MinVectorDB')
    def _init(self, root_path: Union[Path, str]):
        """
        Initialize the vector database.
        """
        self._root_path = Path(root_path).absolute()
        if not self._root_path.exists():
            self._root_path.mkdir(parents=True)
        logger.info(f"Successful initialization of MinVectorDB in root_path: {self._root_path.as_posix()}")

        self._register = _Register(root_path)
        self._collections = {}
        self.STATUS = 'INITIALIZED'

    @unavailable_if_deleted
    def require_collection(
            self, collection: str, dim: int = None,
            n_clusters: int = 16, chunk_size: int = 100_000,
            distance: str = 'cosine', index_mode: str = 'IVF-FLAT', dtypes: str = 'float32',
            use_cache: bool = True, scaler_bits: Union[int, None] = 8,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            drop_if_exists: bool = False,
            description: str = None
    ):
        """Create or load a collection in the database.
            .. versionadded:: 0.3.0

        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors. Default is None.
                When creating a new collection, the dimension of the vectors must be specified.
                When loading an existing collection, the dimension of the vectors is automatically loaded.
            collection (str): The name of the collection.
            n_clusters (int): The number of clusters for the IVF-FLAT index. Default is 8.
            chunk_size (int): The size of each data chunk. Default is 100_000.
            distance (str): Method for calculating vector distance.
                Options are 'cosine' or 'L2' for Euclidean distance. Default is 'cosine'.
            index_mode (str): The storage mode of the database.
                Options are 'FLAT' or 'IVF-FLAT'. Default is 'IVF-FLAT'.
            dtypes (str): The data type of the vectors. Default is 'float32'.
                Options are 'float16', 'float32' or 'float64'.
            use_cache (bool): Whether to use cache for query. Default is True.
            scaler_bits (int): The number of bits for scalar quantization.
                Options are 8, 16, or 32. The default is None, which means no scalar quantization.
                The 8 for 8-bit, 16 for 16-bit, and 32 for 32-bit.
            n_threads (int): The number of threads to use for parallel processing. Default is 10.
            warm_up (bool): Whether to warm up the database. Default is False.
            drop_if_exists (bool): Whether to drop the collection if it already exists. Default is False.
            description (str): A description of the collection. Default is None.
                The description is limited to 500 characters.
                    .. versionadded:: 0.3.4

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        collection_path = self._root_path / collection

        if description is not None and not isinstance(description, str):
            raise ValueError('Description must be a string')
        elif description is not None and len(description) > 500:
            raise ValueError('Description must be less than 500 characters')

        logger.info(f"Creating collection {collection} with: \n "
                    f"\r//    dim={dim}, collection='{collection}', \n"
                    f"\r//    n_clusters={n_clusters}, chunk_size={chunk_size},\n"
                    f"\r//    distance='{distance}', index_mode='{index_mode}', \n"
                    f"\r//    dtypes='{dtypes}', use_cache={use_cache}, \n"
                    f"\r//    scaler_bits={scaler_bits}, n_threads={n_threads}, \n"
                    f"\r//    warm_up={warm_up}, drop_if_exists={drop_if_exists}, \n"
                    f"\r//    description={description}"
                    )

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

        self._collections[collection] = StandaloneMinVectorDB(
            dim=dim, database_path=collection_path.as_posix(), chunk_size=chunk_size, dtypes=dtypes,
            n_clusters=n_clusters, distance=distance, index_mode=index_mode, use_cache=use_cache,
            scaler_bits=scaler_bits, n_threads=n_threads, warm_up=warm_up, initialize_as_collection=True
        )
        self._register.register_collection(
            collection, dim=dim, database_path=collection_path.as_posix(), chunk_size=chunk_size, dtypes=dtypes,
            n_clusters=n_clusters, distance=distance, index_mode=index_mode, use_cache=use_cache,
            scaler_bits=scaler_bits, n_threads=n_threads, warm_up=warm_up, initialize_as_collection=True,
            description=description
        )

        self._collections[collection].update_description = partial(self.update_collection_description, collection)

        return self._collections[collection]

    @unavailable_if_deleted
    def get_collection(self, collection: str):
        """
        Get a collection from the database.
            .. versionadded:: 0.3.0

        Parameters:
            collection (str): The name of the collection to get.

        Returns:
            StandaloneMinVectorDB: The collection.
        """
        if collection not in self._collections:
            if collection not in self._register:
                raise ValueError(f"Collection '{collection}' does not exist.")

            params = self._register.get_collections_details()[collection]
            if 'description' in params:
                del params['description']
            self._collections[collection] = StandaloneMinVectorDB(**params)

        return self._collections[collection]

    @unavailable_if_deleted
    def show_collections(self):
        """
        Show the collections in the database.
            .. versionadded:: 0.3.0

        Returns:
            list: The list of collections in the database.
        """
        return self._register.show_collections()

    @unavailable_if_deleted
    def drop_collection(self, collection: str):
        """
        Delete a collection from the database.
            .. versionadded:: 0.3.0

        Parameters:
            collection (str): The name of the collection to delete.
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

    def drop_database(self):
        """
        Delete the database.
            .. versionadded:: 0.3.0
        """
        if self.STATUS == 'DELETED':
            return

        for collection in self.show_collections():
            self.drop_collection(collection)

        if self._root_path.exists():
            for file in self._root_path.iterdir():
                if file.is_file():
                    file.unlink()
                else:
                    for f in file.iterdir():
                        f.unlink()
                    file.rmdir()
            self._root_path.rmdir()

        MinVectorDBLocalClient._instance = None
        self.STATUS = 'DELETED'

    def update_collection_description(self, collection: str, description: str):
        """
        Update the description of the collection.
            .. versionadded:: 0.3.4

        Parameters:
            collection (str): The name of the collection.
            description (str): The description of the collection.
        """
        self._register.update_description(collection, description)

    def show_collections_details(self) -> pd.DataFrame:
        """
        Show the collections in the database.
            .. versionadded:: 0.3.4

        Returns:
            list: The list of collections in the database.
        """
        return self._register.show_collections_details()

    @property
    def root_path(self):
        return self._root_path.as_posix()

    def __repr__(self):
        return f"{self.STATUS} MinVectorDB(root_path='{self._root_path.as_posix()}')"

    def __str__(self):
        return f"{self.STATUS} MinVectorDB(root_path='{self._root_path.as_posix()}')"
