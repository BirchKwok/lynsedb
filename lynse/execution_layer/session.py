"""session.py: this file is used to manage the database insertion operations."""
from typing import Union, List, Tuple

import numpy as np

from ..utils import copy_doc


class DataOpsSession:
    def __init__(self, db):
        """
        A class to manage the database insertion operations.

        Parameters:
            db: The database to be managed.
        """
        self.db = db
        copy_doc(self.add_item, db.add_item)
        copy_doc(self.bulk_add_items, db.bulk_add_items)

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
                 field: dict = None, buffer_size: int = 10000) -> int:
        return self.db.add_item(vector, id=id, field=field, buffer_size=buffer_size)

    def bulk_add_items(
            self, vectors: Union[List[Tuple[np.ndarray, int, dict]], List[Tuple[np.ndarray, int]]],
            **kwargs
    ):
        return self.db.bulk_add_items(vectors, **kwargs)
