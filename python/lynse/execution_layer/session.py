"""session.py: this file is used to manage the database insertion operations."""
import queue
from typing import Union, List, Tuple

import numpy as np
from ..utils.utils import thread_local


class DataInsertionSession:
    def __init__(self, db):
        """
        A class to manage the database insertion operations.

        Parameters:
            db: The database to be managed.
        """
        self.db = db

    def __enter__(self):
        thread_local.caller_name = "DataInsertionSession"
        return self.db

    def _commit(self):
        self.db.commit()

    def _discard_pending(self):
        lock = getattr(self.db, "_lock", None)
        if lock is None or not hasattr(self.db, "_mesosphere_list"):
            return
        with lock:
            self.db._mesosphere_list = queue.Queue()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type:
                self._discard_pending()
            else:
                self._commit()
        finally:
            if hasattr(thread_local, "caller_name"):
                del thread_local.caller_name
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
            buffer_size (int or bool): The buffer size for the storage worker. Default is True.

                - If True, the client default batch size (10000 for local collections) will be used.
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
            batch_size: int = 1000,
            enable_progress_bar: bool = True,
    ):
        """
        Bulk add vectors to the collection in batches.

        It is recommended to use incremental ids for best performance.

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved.
                Each vector can be a tuple of (vector, id, field).
            batch_size (int): The batch size. Default is 1000.
            enable_progress_bar (bool): Whether to enable the progress bar.

        Returns:
            list: A list of indices where the vectors are stored.
        """
        return self.db.bulk_add_items(
            vectors,
            batch_size=batch_size,
            enable_progress_bar=enable_progress_bar,
        )

