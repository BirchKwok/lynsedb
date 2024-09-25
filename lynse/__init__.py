
__version__ = '0.1.5'

from pathlib import Path
from typing import Union

from .api.http_api.http_api.app import launch_in_jupyter
from .core_components import fields_cache as field_models
from .configs.config import generate_config_file, load_config_file

generate_config_file(), load_config_file()


class _InstanceDistributor:
    from pathlib import Path
    from typing import Union

    def __new__(cls, root_path: Union[str, Path, None], database_name: str):
        from pathlib import Path
        from .api.native_api.high_level import LocalClient
        from .api.http_api.client_api import HTTPClient
        from .configs.config import config

        if isinstance(root_path, Path):
            root_path = root_path.as_posix()

        if root_path is not None and (root_path.startswith('http://') or root_path.startswith('https://')):
            instance = HTTPClient(uri=root_path, database_name=database_name)
        else:
            if root_path is None:
                native_api_root_path = config.LYNSE_DEFAULT_ROOT_PATH
            else:
                # new features, root_path can be specified directly
                native_api_root_path = Path(root_path)

            instance = LocalClient(native_api_root_path / database_name)

        return instance


class VectorDBClient:
    """
    This class determines whether it is local or remote based on the URI, thereby implementing the distribution of local and remote client communication.
    The data storage location is also determined accordingly.

    - **When `uri` is None**: The local client is used by default, and the data is stored under the default root path.
    - **When `uri` is a local path**: The local client is used, and the data is stored under the specified path.
    - **When `uri` is a remote URL**: The remote client is used, and the data is stored on the remote server.
    """
    def __init__(self, uri: Union[str, None, Path] = None):
        """
        Initialize the LynseDB client.

        Parameters:
            uri (Pathlike or str or None): The URI of the LynseDB server. It can be either a local path or a remote URL.

               - If it is a remote URL, the client will use the HTTP API.
               - If it is a local path, the client will use the native API.
                    The path refers to the root path of the LynseDB storage.
               - If set to None, the client will use the native API,
                    and the database will be stored in the default root path,
                    when you need to change the default root path,
                    you can set the environment variable LYNSE_DEFAULT_ROOT_PATH or change the config file.

        """
        self._is_remote = uri is not None and (uri.startswith('http://') or uri.startswith('https://'))

        if self._is_remote:
            from spinesUtils.asserts import raise_if
            import httpx

            raise_if(ValueError, not isinstance(uri, (str, Path)),
                     'uri must be a string, Pathlike or None')
            if isinstance(uri, Path):
                uri = uri.as_posix()

            if uri.startswith('http://') or uri.startswith('https://'):
                try:
                    response = httpx.get(uri)
                    if response.status_code != 200:
                        raise ConnectionError(f'Failed to connect to the server at {uri}.')

                    try:
                        rj = response.json()
                        if rj != {'status': 'success', 'message': 'LynseDB HTTP API'}:
                            raise ConnectionError(f'Failed to connect to the server at {uri}.')
                    except Exception as e:
                        print(e)
                        raise ConnectionError(f'Failed to connect to the server at {uri}.')

                except httpx.RequestError:
                    raise ConnectionError(f'Failed to connect to the server at {uri}.')
        else:
            uri = Path(uri) if uri is not None else None

        self._uri = uri

    def create_database(self, database_name: str, drop_if_exists: bool = False):
        """
        Create the database using a lazy mode, where entities are only created when they are actually used.

        Parameters:
            database_name (str): The name of the database to create.
            drop_if_exists (bool): Whether to drop the database if it already exists.
                If set to True, the existing database will be immediately deleted before creating a new one.

        Returns:
            None
        """
        from .api.http_api.client_api import raise_error_response
        from .configs.config import config
        import httpx

        if not self._is_remote:
            from .api.native_api.database_manager import DatabaseManager

            db_manager = DatabaseManager(root_path=self._uri or config.LYNSE_DEFAULT_ROOT_PATH)
            db_manager.register(db_name=database_name)

            if drop_if_exists:
                _InstanceDistributor(root_path=self._uri or config.LYNSE_DEFAULT_ROOT_PATH,
                                     database_name=database_name).drop_database()
        else:
            try:
                rj = httpx.post(f'{self._uri}/create_database', json={'database_name': database_name,
                                                                      'drop_if_exists': drop_if_exists})
                if rj.status_code != 200:
                    raise_error_response(rj)
            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {self._uri}.')

        return _InstanceDistributor(root_path=self._uri or config.LYNSE_DEFAULT_ROOT_PATH, database_name=database_name)

    def get_database(self, database_name: str):
        """
        Get an existing database.

        Parameters:
            database_name (str): The name of the database to get.

        Returns:
            LynseDB: (LocalClient, HTTPClient):
                The appropriate LynseDB client instance based on the root path.
                If the root path is a local path, return a LocalClient instance,
                otherwise return a HTTPClient instance.
        """
        from spinesUtils.asserts import raise_if

        from .api.http_api.client_api import raise_error_response
        from .configs.config import config
        import httpx

        if not self._is_remote:
            from .api.native_api.database_manager import DatabaseManager

            db_manager = DatabaseManager(root_path=self._uri or config.LYNSE_DEFAULT_ROOT_PATH)
            databases = db_manager.list_database()
            raise_if(ValueError, database_name not in databases, f'{database_name} does not exist.')
        else:
            try:
                rj = httpx.get(f'{self._uri}/list_databases')
                if rj.status_code != 200:
                    raise_error_response(rj)

                databases = rj.json()['params']['databases']
                raise_if(ValueError, database_name not in databases, f'{database_name} does not exist.')
            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {self._uri}.')

        return _InstanceDistributor(root_path=self._uri or config.LYNSE_DEFAULT_ROOT_PATH, database_name=database_name)

    def list_databases(self):
        """
        List all databases.

        Returns:
            List: A list of all databases.
        """
        from .api.http_api.client_api import raise_error_response
        import httpx

        if not self._is_remote:
            from .api.native_api.database_manager import DatabaseManager
            from .configs.config import config

            db_manager = DatabaseManager(root_path=self._uri or config.LYNSE_DEFAULT_ROOT_PATH)
            return db_manager.list_database()
        else:
            try:
                rj = httpx.get(f'{self._uri}/list_databases')
                if rj.status_code != 200:
                    raise_error_response(rj)

                return rj.json()['params']['databases']
            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {self._uri}.')

    def drop_database(self, database_name: str):
        """
        Delete a database.

        Parameters:
            database_name (str): The name of the database to delete.

        Returns:
            None
        """
        from .api.http_api.client_api import raise_error_response
        import httpx

        if not self._is_remote:
            from .api.native_api.database_manager import DatabaseManager
            from .configs.config import config

            db_manager = DatabaseManager(root_path=self._uri or config.LYNSE_DEFAULT_ROOT_PATH)
            databases = db_manager.list_database()
            if database_name not in databases:
                return
            db_manager.delete(database_name)
        else:
            try:
                rj = httpx.post(f'{self._uri}/delete_database', json={'database_name': database_name})
                if rj.status_code != 200:
                    raise_error_response(rj)
            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {self._uri}.')

    def __repr__(self):
        from .configs.config import config
        return f'{self.__class__.__name__}(uri={self._uri or "DefaultRootPath"})'

    def __str__(self):
        return self.__repr__()


def _load_and_register_module(module):
    """Dynamically import and register a module under the current module namespace."""
    import importlib
    import sys

    rename_modules = {'fields_cache': 'field_models'}

    try:
        if isinstance(module, str):
            module_name = module.rsplit('.', 1)[-1]
            module = importlib.import_module(module)
        else:
            module_name = module.__name__.rsplit('.', 1)[-1]

        module_name = rename_modules.get(module_name, module_name)

    except ImportError as e:
        raise ImportError(f"Could not import module {module}") from e

    current_module_name = __name__
    sys.modules[f'{current_module_name}.{module_name}'] = module


_load_and_register_module(field_models)
