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

    def _commit(self):
        handler = getattr(self.db, '_matrix_serializer', self.db)
        if not handler.COMMIT_FLAG:
            self.db.commit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If an exception occurred, rollback the transaction
        if exc_type:
            self._commit()
            raise exc_type(exc_val).with_traceback(exc_tb)

        self._commit()

        return False

    def add_item(self, vector: Union[np.ndarray, list], id: int, *,
                 field: dict = None, buffer_size: int = True) -> int:
        """
        Add a single vector to the collection.

        It is recommended to use incremental ids for best performance.

        Parameters:
            vector (np.ndarray): The vector to be added.
            id (int): The ID of the vector.
            field (dict, optional, keyword-only): The field of the vector. Default is None.
                If None, the field will be set to an empty string.
            buffer_size (int or bool or None): The buffer size for the storage worker. Default is True.

                - If None, the vector will be directly written to the disk.
                - If True, the buffer_size will be set to chunk_size,
                    and the vectors will be written to the disk when the buffer is full.
                - If False, the vector will be directly written to the disk.
                - If int, when the buffer is full, the vectors will be written to the disk.

        Returns:
            int: The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        return self.db.add_item(vector, id=id, field=field, buffer_size=buffer_size)

    def bulk_add_items(
            self, vectors: Union[List[Tuple[np.ndarray, int, dict]], List[Tuple[np.ndarray, int]]],
            **kwargs
    ):
        """
        Bulk add vectors to the collection in batches.

        It is recommended to use incremental ids for best performance.

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved.
                Each vector can be a tuple of (vector, id, field).

        Returns:
            list: A list of indices where the vectors are stored.
        """
        return self.db.bulk_add_items(vectors, **kwargs)
