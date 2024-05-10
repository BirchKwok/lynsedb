
__version__ = '0.3.5'


class MinVectorDB:
    """
    Create a MinVectorDB instance.

    Parameters:
        root_path (str or Path): The root path of the MinVectorDB instance.

    Returns:
        MinVectorDB: (MinVectorDBLocalClient, MinVectorDBHTTPClient):
            The appropriate MinVectorDB client instance based on the root path.
            If the root path is a local path, return a MinVectorDBLocalClient instance,
            otherwise return a MinVectorDBHTTPClient instance.
    """
    from pathlib import Path
    from typing import Union

    def __new__(cls, root_path: Union[str, Path]):
        from pathlib import Path
        from min_vec.api.native_api.high_level import MinVectorDBLocalClient
        from min_vec.api.http_api.client_api import MinVectorDBHTTPClient

        if isinstance(root_path, Path):
            root_path = root_path.as_posix()

        if root_path.startswith('http://') or root_path.startswith('https://'):
            instance = MinVectorDBHTTPClient(url=root_path)
        else:
            instance = MinVectorDBLocalClient(root_path)

        return instance
