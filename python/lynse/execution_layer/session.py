"""session.py: this file is used to manage database insertion sessions."""
import queue
from typing import Any

from ..api._records import normalize_external_ids
from ..utils.utils import thread_local


class DataInsertionSession:
    def __init__(self, db):
        """
        A class to manage the database insertion operations.

        Parameters:
            db: The database to be managed.
        """
        self.db = db
        self._pending_adds = []

    def __enter__(self):
        thread_local.caller_name = "DataInsertionSession"
        return self

    def __getattr__(self, name):
        return getattr(self.db, name)

    def _commit(self):
        for ids, vectors, documents, fields, batch_size, wire_dtype in self._pending_adds:
            self.db.add(
                ids,
                vectors=vectors,
                documents=documents,
                fields=fields,
                batch_size=batch_size,
                wire_dtype=wire_dtype,
            )
        self._pending_adds.clear()
        self.db.commit()

    def _discard_pending(self):
        self._pending_adds.clear()
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

    def add(self, ids: Any, *, vectors=None, documents=None, fields=None, batch_size: int = 1000,
            wire_dtype: str = "float32"):
        """Buffer records and write them when the session exits successfully."""
        external_ids, single_id = normalize_external_ids(ids)
        self._pending_adds.append((
            ids,
            vectors,
            documents,
            fields,
            batch_size,
            wire_dtype,
        ))
        return external_ids[0] if single_id else external_ids
