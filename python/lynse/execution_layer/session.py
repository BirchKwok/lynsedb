"""session.py: this file is used to manage database insertion sessions."""
import queue
from typing import Any

import numpy as np

from ..api._records import (
    normalize_documents,
    normalize_external_ids,
    normalize_fields,
    normalize_vectors,
)
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
        self._pending_batches = []
        self._compact_threshold = 50_000

    def __enter__(self):
        thread_local.caller_name = "DataInsertionSession"
        return self

    def __getattr__(self, name):
        return getattr(self.db, name)

    def _commit(self):
        self._compact_pending_adds()
        for ids, vectors, documents, fields, batch_size, wire_dtype in self._pending_batches:
            self.db.add(
                ids,
                vectors=vectors,
                documents=documents,
                fields=fields,
                batch_size=batch_size,
                wire_dtype=wire_dtype,
            )
        self._pending_adds.clear()
        self._pending_batches.clear()
        self.db.commit()

    def _iter_batches(self):
        current = None

        def flush_current():
            nonlocal current
            if current is None or not current["ids"]:
                current = None
                return None

            ids = current["ids"]
            fields = current["fields"]
            batch_size = current["batch_size"]
            wire_dtype = current["wire_dtype"]
            if current["mode"] == "vectors":
                vectors = np.ascontiguousarray(current["vectors"], dtype=np.float32)
                batch = (ids, vectors, None, fields, batch_size, wire_dtype)
            else:
                batch = (ids, None, current["documents"], fields, batch_size, wire_dtype)
            current = None
            return batch

        for external_ids, vectors, documents, fields, batch_size, wire_dtype in self._pending_adds:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
            n_records = len(external_ids)
            if fields is None:
                field_list = [{} for _ in range(n_records)]
            elif n_records == 1 and isinstance(fields, dict):
                field_list = [fields]
            else:
                field_list = normalize_fields(fields, n_records)

            if vectors is None:
                if documents is None:
                    raise ValueError("add() requires vectors or documents")
                docs, _ = normalize_documents(documents, n_records)
                mode = "documents"
                rows = docs
            elif n_records == 1 and documents is None:
                mode = "vectors"
                rows = [vectors]
            else:
                mode = "vectors"
                rows = normalize_vectors(vectors, n_records)

            if current is None:
                current = {
                    "mode": mode,
                    "batch_size": batch_size,
                    "wire_dtype": wire_dtype,
                    "ids": [],
                    "vectors": [],
                    "documents": [],
                    "fields": [],
                }

            if (
                current["mode"] != mode
                or current["batch_size"] != batch_size
                or current["wire_dtype"] != wire_dtype
            ):
                batch = flush_current()
                if batch is not None:
                    yield batch
                current = {
                    "mode": mode,
                    "batch_size": batch_size,
                    "wire_dtype": wire_dtype,
                    "ids": [],
                    "vectors": [],
                    "documents": [],
                    "fields": [],
                }

            if mode == "vectors":
                for start in range(0, n_records, batch_size):
                    end = min(start + batch_size, n_records)
                    if current["ids"] and len(current["ids"]) + (end - start) > batch_size:
                        batch = flush_current()
                        if batch is not None:
                            yield batch
                        current = {
                            "mode": mode,
                            "batch_size": batch_size,
                            "wire_dtype": wire_dtype,
                            "ids": [],
                            "vectors": [],
                            "documents": [],
                            "fields": [],
                        }
                    current["ids"].extend(external_ids[start:end])
                    current["vectors"].extend(rows[start:end])
                    current["fields"].extend(field_list[start:end])
            else:
                for start in range(0, n_records, batch_size):
                    end = min(start + batch_size, n_records)
                    if current["ids"] and len(current["ids"]) + (end - start) > batch_size:
                        batch = flush_current()
                        if batch is not None:
                            yield batch
                        current = {
                            "mode": mode,
                            "batch_size": batch_size,
                            "wire_dtype": wire_dtype,
                            "ids": [],
                            "vectors": [],
                            "documents": [],
                            "fields": [],
                        }
                    current["ids"].extend(external_ids[start:end])
                    current["documents"].extend(rows[start:end])
                    current["fields"].extend(field_list[start:end])

        batch = flush_current()
        if batch is not None:
            yield batch

    def _compact_pending_adds(self):
        if not self._pending_adds:
            return
        self._pending_batches.extend(self._iter_batches())
        self._pending_adds.clear()

    def _discard_pending(self):
        self._pending_adds.clear()
        self._pending_batches.clear()
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

    def add(self, ids: Any, *, vectors=None, documents=None, fields=None, batch_size: int = 50000,
            wire_dtype: str = "float32"):
        """Buffer records and write them when the session exits successfully."""
        external_ids, single_id = normalize_external_ids(ids)
        self._pending_adds.append((
            external_ids,
            vectors,
            documents,
            fields,
            batch_size,
            wire_dtype,
        ))
        if len(self._pending_adds) >= self._compact_threshold:
            self._compact_pending_adds()
        return external_ids[0] if single_id else external_ids
