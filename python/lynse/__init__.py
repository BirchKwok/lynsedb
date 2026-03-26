from pathlib import Path
from typing import Union

from .configs.config import generate_config_file, load_config_file


FILE_PATH = Path(__file__).parent.parent

__version__ = '0.3.0'


generate_config_file(), load_config_file()


class VectorDBClient:
    """
    LynseDB client.

    - **When `uri` is None or a local path**: Uses the Rust backend directly via PyO3
      (zero network overhead). Data is stored under the specified (or default) root path.
    - **When `uri` is a remote URL**: Connects to an existing remote Rust HTTP server.
    """

    def __init__(self, uri: Union[str, None, Path] = None):
        """
        Initialize the LynseDB client.

        Parameters:
            uri (Pathlike or str or None): The URI of the LynseDB server. It can be either a local path or a remote URL.

               - If it is a remote URL, the client will connect to the existing server via HTTP.
               - If it is a local path, the Rust backend is used directly (no HTTP server).
                    The path refers to the root path of the LynseDB storage.
               - If set to None, the Rust backend is used with the default root path.

        """
        if isinstance(uri, Path):
            uri = uri.as_posix()

        self._is_remote = uri is not None and isinstance(uri, str) and (
            uri.startswith('http://') or uri.startswith('https://')
        )

        if self._is_remote:
            import httpx
            # Connect to existing remote server
            try:
                response = httpx.get(uri)
                if response.status_code != 200:
                    raise ConnectionError(f'Failed to connect to the server at {uri}.')
                rj = response.json()
                if rj.get('status') != 'success':
                    raise ConnectionError(f'Failed to connect to the server at {uri}.')
            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {uri}.')

            self._uri = uri
            self._root_path = None
            self._manager = None

            # Persistent connection pool for remote HTTP requests
            transport = httpx.HTTPTransport(retries=3)
            self._client = httpx.Client(transport=transport, timeout=300, base_url=self._uri)
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
            self._manager = DatabaseManager(root_path)

    # ── HTTP helpers (remote mode only) ──────────────────────────────────────

    def _post(self, endpoint: str, data: dict = None):
        """Send POST request to the server (remote mode only)."""
        import httpx
        try:
            response = self._client.post(endpoint, json=data or {})
            if response.status_code != 200:
                rj = response.json()
                raise RuntimeError(rj.get('error', f'Server error: {response.status_code}'))
            return response.json()
        except httpx.RequestError:
            raise ConnectionError(f'Failed to connect to the server at {self._uri}.')

    def _get(self, endpoint: str):
        """Send GET request to the server (remote mode only)."""
        import httpx
        try:
            response = self._client.get(endpoint)
            if response.status_code != 200:
                rj = response.json()
                raise RuntimeError(rj.get('error', f'Server error: {response.status_code}'))
            return response.json()
        except httpx.RequestError:
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
            return HTTPClient(uri=self._uri, database_name=database_name)
        else:
            from .api.local_client import LocalClient

            if drop_if_exists and self._manager.database_exists(database_name):
                self._manager.drop_database(database_name)
            self._manager.create_database(database_name)
            return LocalClient(manager=self._manager, database_name=database_name)

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
            return HTTPClient(uri=self._uri, database_name=database_name)
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

    def __repr__(self):
        if self._is_remote:
            return f'{self.__class__.__name__}(uri={self._uri})'
        else:
            return f'{self.__class__.__name__}(root_path={self._root_path})'

    def __str__(self):
        return self.__repr__()
