"""high_level.py - The MinVectorDB API."""
import json
from pathlib import Path
from typing import Union

from spinesUtils.asserts import ParameterTypeAssert

from min_vec.api.low_level import StandaloneMinVectorDB
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

        # 遍历collections，如果有不存在的collection，删除
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

    def __contains__(self, item):
        return item in self.get_collections_details()


class MinVectorDB:
    """
    A class for managing a vector database stored in .mvdb files and computing vectors similarity.
    """

    @ParameterTypeAssert({
        'root_path': str
    }, func_name='MinVectorDB')
    def __init__(self, root_path: str) -> None:
        """
        Initialize the vector database.

        Parameters:
            root_path (str): The root path of the database.
                .. versionadded:: 0.3.0
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
            drop_if_exists: bool = False
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

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        collection_path = self._root_path / collection

        logger.info(f"Creating collection {collection} with: \n "
                    f"\r//    dim={dim}, collection='{collection}', \n"
                    f"\r//    n_clusters={n_clusters}, chunk_size={chunk_size},\n"
                    f"\r//    distance='{distance}', index_mode='{index_mode}', \n"
                    f"\r//    dtypes='{dtypes}', use_cache={use_cache}, \n"
                    f"\r//    scaler_bits={scaler_bits}, n_threads={n_threads}"
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
            scaler_bits=scaler_bits, n_threads=n_threads, warm_up=warm_up, initialize_as_collection=True)

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

            self._collections[collection] = StandaloneMinVectorDB(
                **self._register.get_collections_details()[collection]
            )
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
            _temp_collection = self.get_collection(collection)
            self.drop_collection(collection)

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

        self.STATUS = 'DELETED'

    @property
    def root_path(self):
        return self._root_path.as_posix()

    def __repr__(self):
        return f"{self.STATUS} MinVectorDB(root_path='{self._root_path.as_posix()}')"

    def __str__(self):
        return f"{self.STATUS} MinVectorDB(root_path='{self._root_path.as_posix()}')"
