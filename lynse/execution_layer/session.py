"""session.py: this file is used to manage the database insertion operations."""
from typing import Union, List, Tuple

import numpy as np


class DataOpsSession:
    def __init__(self, db):
        """
        A class to manage the database insertion operations.

        Parameters:
            db: The database to be managed.
        """
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If an exception occurred, rollback the transaction
        if exc_type:
            raise exc_type(exc_val).with_traceback(exc_tb)

        handler = getattr(self.db, '_matrix_serializer', self.db)
        if not handler.COMMIT_FLAG:
            self.db.commit()

        return False

    def add_item(self, vector: Union[np.ndarray, list], id: int, *,
                 field: dict = None, normalize=False, **kwargs) -> int:
        """
        Add a single vector to the database.
            .. versionadded:: 0.3.6

        Parameters:
            vector (np.ndarray or list): The vector to be added.
            id (int): The ID of the vector.
            field (dict, optional, keyword-only): The field of the vector. Default is None. If None, the field will be
                set to an empty string.
            normalize (bool): Whether to normalize the vector. Default is False.
            kwargs: Additional keyword arguments.
                If you are using the http API, you can pass the parameter `delay_num` as keyword.
                It means the number of items to delay push. Default is 1000.

        Returns:
            int: The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        return self.db.add_item(vector, id=id, field=field, normalize=normalize, **kwargs)

    def bulk_add_items(
            self, vectors: Union[List[Tuple[np.ndarray, int, dict]], List[Tuple[np.ndarray, int]]],
            *, normalize=False, **kwargs
    ):
        """
        Bulk add vectors to the database in batches.
            .. versionadded:: 0.3.6

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved. Each vector can be a tuple of (
            vector, id, field).
            normalize (bool): Whether to normalize the vectors. Default is False.
            kwargs: Additional keyword arguments.
                If you are using the http API, you can pass the parameter `enable_progress_bar`
                and the parameter `batch_size` as keyword.
                The `enable_progress_bar` is a boolean value to enable the progress.
                The `batch_size` is the number of items to delay push. Default is 1000.

        Returns:
            list: A list of indices where the vectors are stored.
        """
        return self.db.bulk_add_items(vectors, normalize=normalize, **kwargs)
