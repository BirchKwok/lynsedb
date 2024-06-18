
__version__ = '0.0.1'

from lynse.api.http_api.http_api import launch_in_jupyter
from lynse.core_components import kv_cache as field_models


class _InstanceDistributor:
    from pathlib import Path
    from typing import Union

    def __new__(cls, root_path: Union[str, Path, None], database_name: str):
        from pathlib import Path
        from lynse.api.native_api.high_level import LocalClient
        from lynse.api.http_api.client_api import HTTPClient
        from lynse.configs.config import config

        if isinstance(root_path, Path):
            root_path = root_path.as_posix()

        if root_path is not None and (root_path.startswith('http://') or root_path.startswith('https://')):
            instance = HTTPClient(url=root_path, database_name=database_name)
        else:
            native_api_root_path = config.LYNSE_DEFAULT_ROOT_PATH

            instance = LocalClient((native_api_root_path / database_name).as_posix())

        return instance


class VectorDBClient:
    from typing import Union

    def __init__(self, url: Union[str, None] = None):
        """
        Initialize the LynseDB client.

        Parameters:
            url (str): The URL of the LynseDB server. If None, the client will use the native API.
        """
        if url is not None:
            from spinesUtils.asserts import raise_if
            import httpx

            raise_if(ValueError, not isinstance(url, str), 'url must be a string')
            raise_if(ValueError, not (url.startswith('http://') or url.startswith('https://')),
                     'url must start with http:// or https://, '
                     'if you are using native API, please pass None as url')

            try:
                response = httpx.get(url)
                if response.status_code != 200:
                    raise ConnectionError(f'Failed to connect to the server at {url}.')

                try:
                    rj = response.json()
                    if rj != {'status': 'success', 'message': 'LynseDB HTTP API'}:
                        raise ConnectionError(f'Failed to connect to the server at {url}.')
                except Exception as e:
                    print(e)
                    raise ConnectionError(f'Failed to connect to the server at {url}.')

            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {url}.')

        self._url = url

    def create_database(self, database_name: str, drop_if_exists: bool = False):
        """
        Create a new database.
    
        Parameters:
            database_name (str): The name of the database to create.
            drop_if_exists (bool): Whether to drop the database if it already exists.

        Returns:
            None
        """
        from lynse.api.http_api.client_api import raise_error_response
        from lynse.configs.config import config
        import httpx

        if self._url is None:
            from lynse.api.native_api.database_manager import DatabaseManager

            db_manager = DatabaseManager(root_path=config.LYNSE_DEFAULT_ROOT_PATH)
            db_manager.register(db_name=database_name)

            if drop_if_exists:
                _InstanceDistributor(root_path=None, database_name=database_name).drop_database()
        else:
            try:
                rj = httpx.post(f'{self._url}/create_database', json={'database_name': database_name,
                                                                      'drop_if_exists': drop_if_exists})
                if rj.status_code != 200:
                    raise_error_response(rj)
            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {self._url}.')

        return _InstanceDistributor(root_path=self._url, database_name=database_name)

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
        from lynse.api.http_api.client_api import raise_error_response
        from spinesUtils.asserts import raise_if
        from lynse.configs.config import config
        import httpx

        if self._url is None:
            from lynse.api.native_api.database_manager import DatabaseManager

            db_manager = DatabaseManager(root_path=config.LYNSE_DEFAULT_ROOT_PATH)
            databases = db_manager.list_database()
            raise_if(ValueError, database_name not in databases, f'{database_name} does not exist.')
        else:
            try:
                rj = httpx.get(f'{self._url}/list_databases')
                if rj.status_code != 200:
                    raise_error_response(rj)

                databases = rj.json()['params']['databases']
                raise_if(ValueError, database_name not in databases, f'{database_name} does not exist.')
            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {self._url}.')

        return _InstanceDistributor(root_path=self._url, database_name=database_name)

    def list_databases(self):
        """
        List all databases.

        Returns:
            List: A list of all databases.
        """
        from lynse.api.http_api.client_api import raise_error_response
        import httpx

        if self._url is None:
            from lynse.api.native_api.database_manager import DatabaseManager
            from lynse.configs.config import config

            db_manager = DatabaseManager(root_path=config.LYNSE_DEFAULT_ROOT_PATH)
            return db_manager.list_database()
        else:
            try:
                rj = httpx.get(f'{self._url}/list_databases')
                if rj.status_code != 200:
                    raise_error_response(rj)

                return rj.json()['params']['databases']
            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {self._url}.')

    def drop_database(self, database_name: str):
        """
        Delete a database.

        Parameters:
            database_name (str): The name of the database to delete.

        Returns:
            None
        """
        from lynse.api.http_api.client_api import raise_error_response
        import httpx

        if self._url is None:
            from lynse.api.native_api.database_manager import DatabaseManager
            from lynse.configs.config import config

            db_manager = DatabaseManager(root_path=config.LYNSE_DEFAULT_ROOT_PATH)
            databases = db_manager.list_database()
            if database_name not in databases:
                return
            db_manager.delete(database_name)
        else:
            try:
                rj = httpx.post(f'{self._url}/delete_database', json={'database_name': database_name})
                if rj.status_code != 200:
                    raise_error_response(rj)
            except httpx.RequestError:
                raise ConnectionError(f'Failed to connect to the server at {self._url}.')


def load_and_register_module(module):
    """Dynamically import and register a module under the current module namespace."""
    import importlib
    import sys

    rename_modules = {'kv_cache': 'field_models'}

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


load_and_register_module(field_models)
