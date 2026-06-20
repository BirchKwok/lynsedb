from pathlib import Path
from typing import Union

from .configs.config import generate_config_file, load_config_file


FILE_PATH = Path(__file__).parent.parent

__version__ = '0.7.1'


class VectorDBClient:
    """
    LynseDB client.

    - **When `uri` is None or a local path**: Uses the Rust backend directly via PyO3
      (zero network overhead). Data is stored under the specified (or default) root path.
    - **When `uri` is a remote URL**: Connects to an existing remote Rust HTTP server.
    """

    def __init__(
            self,
            uri: Union[str, None, Path] = None,
            api_key: str = None,
            read_only: bool = False,
    ):
        """
        Initialize the LynseDB client.

        Parameters:
            uri (Pathlike or str or None): The URI of the LynseDB server. It can be either a local path or a remote URL.

               - If it is a remote URL, the client will connect to the existing server via HTTP.
               - If it is a local path, the Rust backend is used directly (no HTTP server).
                    The path refers to the root path of the LynseDB storage.
               - If set to None, the Rust backend is used with the default root path.

            api_key (str or None): Optional API key for HTTP server authentication.
               When provided, all HTTP requests include an ``Authorization: Bearer <api_key>`` header.
               Ignored in local (non-HTTP) mode.
            read_only (bool): Open local storage in read-only mode. Ignored in remote mode.

        """
        if isinstance(uri, Path):
            uri = uri.as_posix()

        self._is_remote = uri is not None and isinstance(uri, str) and (
            uri.startswith('http://') or uri.startswith('https://')
        )
        self._api_key = api_key

        if self._is_remote:
            from .utils.poster import RustRemoteSession

            remote_client = RustRemoteSession(base_url=uri, api_key=api_key)
            try:
                auth_response = remote_client.get('/list_databases')
                if auth_response.status_code == 401:
                    raise ConnectionError('Authentication failed: invalid api_key.')
                if auth_response.status_code != 200:
                    raise ConnectionError(f'Failed to connect to the server at {uri}.')
            except Exception:
                remote_client.close()
                raise

            self._uri = uri
            self._root_path = None
            self._manager = None
            self._client = remote_client
        else:
            # Local mode: direct Rust backend, no HTTP overhead
            from .configs.config import config
            from ._backend import DatabaseManager

            if uri is not None:
                root_path = str(Path(uri).resolve())
            else:
                root_path = str(config.LYNSE_DEFAULT_ROOT_PATH)

            self._root_path = root_path
            self._uri = None
            self._client = None
            self._manager = DatabaseManager(root_path, read_only=read_only)

    # ── HTTP helpers (remote mode only) ──────────────────────────────────────

    def _post(self, endpoint: str, data: dict = None):
        """Send POST request to the server (remote mode only)."""
        try:
            response = self._client.post(endpoint, json=data or {})
            if response.status_code != 200:
                rj = response.json()
                raise RuntimeError(rj.get('error', f'Server error: {response.status_code}'))
            return response.json()
        except ConnectionError:
            raise
        except OSError:
            raise ConnectionError(f'Failed to connect to the server at {self._uri}.')

    def _get(self, endpoint: str):
        """Send GET request to the server (remote mode only)."""
        try:
            response = self._client.get(endpoint)
            if response.status_code != 200:
                rj = response.json()
                raise RuntimeError(rj.get('error', f'Server error: {response.status_code}'))
            return response.json()
        except ConnectionError:
            raise
        except OSError:
            raise ConnectionError(f'Failed to connect to the server at {self._uri}.')

    # ── Public API ───────────────────────────────────────────────────────────

    def create_database(self, database_name: str, drop_if_exists: bool = False):
        """
        Create the database using a lazy mode, where entities are only created when they are actually used.

        Parameters:
            database_name (str): The name of the database to create.
            drop_if_exists (bool): Whether to drop the database if it already exists.
                If set to True, the existing database will be immediately deleted before creating a new one.

        Returns:
            LocalClient or HTTPClient: A client instance for the database.
        """
        # Limit the maximum number of databases created to 64
        if len(self.list_databases()) >= 64:
            raise ValueError('The maximum number of databases created is 64.')

        if self._is_remote:
            from .api.http_api.client_api import HTTPClient

            self._post('/create_database', {
                'database_name': database_name,
                'drop_if_exists': drop_if_exists,
            })
            return HTTPClient(uri=self._uri, database_name=database_name, api_key=self._api_key)
        else:
            from .api.local_client import LocalClient

            if drop_if_exists and self._manager.database_exists(database_name):
                self._manager.drop_database(database_name)
            self._manager.create_database(database_name)
            return LocalClient(manager=self._manager, database_name=database_name)

    def create_collection(
            self,
            database_name: str,
            collection: str,
            dim: int = None,
            n_threads: Union[int, None] = 10,
            warm_up: bool = False,
            drop_if_exists: bool = False,
            description: str = None,
            dtypes: str = "float32",
            default_index: Union[str, None] = "FLAT-IP",
            drop_database_if_exists: bool = False,
    ):
        """
        Create or open a database and collection in one call.

        Parameters:
            database_name (str): The name of the database to create or open.
            collection (str): The name of the collection to create or open.
            dim (int): Optional vector dimension. If omitted for a new
                collection, LynseDB infers it from the first inserted vectors.
            n_threads (int): The number of threads. Default is 10.
            warm_up (bool): Whether to warm up. Default is False.
            drop_if_exists (bool): Whether to drop the collection if it exists.
                Default is False.
            description (str): A description of the collection. Default is None.
            dtypes (str): Dense vector storage dtype, "float32" or "float16".
            default_index (str or None): Index mode to build automatically after
                the first write to a newly created collection. Use None to
                disable automatic index creation.
            drop_database_if_exists (bool): Whether to drop and recreate the
                database before creating the collection. Default is False.

        Returns:
            LocalCollection or Collection: The collection object.
        """
        if drop_database_if_exists or database_name not in self.list_databases():
            db = self.create_database(
                database_name, drop_if_exists=drop_database_if_exists
            )
        else:
            db = self.get_database(database_name)

        return db.require_collection(
            collection=collection,
            dim=dim,
            n_threads=n_threads,
            warm_up=warm_up,
            drop_if_exists=drop_if_exists,
            description=description,
            dtypes=dtypes,
            default_index=default_index,
        )

    def get_database(self, database_name: str):
        """
        Get an existing database.

        Parameters:
            database_name (str): The name of the database to get.

        Returns:
            LocalClient or HTTPClient: A client instance for the database.
        """
        databases = self.list_databases()
        if database_name not in databases:
            raise ValueError(f'{database_name} does not exist.')

        if self._is_remote:
            from .api.http_api.client_api import HTTPClient
            return HTTPClient(uri=self._uri, database_name=database_name, api_key=self._api_key)
        else:
            from .api.local_client import LocalClient
            return LocalClient(manager=self._manager, database_name=database_name)

    def list_databases(self):
        """
        List all databases.

        Returns:
            List: A list of all databases.
        """
        if self._is_remote:
            rj = self._get('/list_databases')
            return rj['params']['databases']
        else:
            return self._manager.list_databases()

    def drop_database(self, database_name: str):
        """
        Delete a database.

        Parameters:
            database_name (str): The name of the database to delete.

        Returns:
            None
        """
        databases = self.list_databases()
        if database_name not in databases:
            return

        if self._is_remote:
            self._post('/delete_database', {'database_name': database_name})
        else:
            self._manager.drop_database(database_name)

    def snapshot_database(self, database_name: str, snapshot_path: Union[str, Path]):
        """
        Create a filesystem snapshot for a database.

        Parameters:
            database_name (str): The name of the database to snapshot.
            snapshot_path (Pathlike or str): The snapshot target path.
        """
        if self._is_remote:
            self._post('/snapshot_database', {
                'database_name': database_name,
                'snapshot_path': str(snapshot_path),
            })
        else:
            self._manager.snapshot_database(database_name, str(snapshot_path))

    def restore_database(
            self,
            database_name: str,
            snapshot_path: Union[str, Path],
            overwrite: bool = False,
    ):
        """
        Restore a database from a filesystem snapshot.

        Parameters:
            database_name (str): The database name to restore into.
            snapshot_path (Pathlike or str): The snapshot source path.
            overwrite (bool): Whether to replace an existing database.
        """
        if self._is_remote:
            self._post('/restore_database', {
                'database_name': database_name,
                'snapshot_path': str(snapshot_path),
                'overwrite': overwrite,
            })
        else:
            self._manager.restore_database(database_name, str(snapshot_path), overwrite)

    def close(self):
        """Release the local storage handle (writer lock) for this root path."""
        if self._is_remote:
            if self._client is not None:
                self._client.close()
                self._client = None
        elif self._manager is not None:
            self._manager.close()
            self._manager = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if self._is_remote:
            return f'{self.__class__.__name__}(uri={self._uri})'
        else:
            return f'{self.__class__.__name__}(root_path={self._root_path})'

    def __str__(self):
        return self.__repr__()
