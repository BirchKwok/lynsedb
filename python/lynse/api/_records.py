"""Shared record helpers for the public collection API."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


DOCUMENT_FIELD = "document"

def _is_scalar_id(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (str, int, np.integer))


def normalize_external_ids(ids: Any) -> tuple[list[str | int], bool]:
    """Normalize a public ID or ID sequence into Python str/int values."""
    if _is_scalar_id(ids):
        return [_python_id(ids)], True
    if isinstance(ids, np.ndarray):
        values = ids.tolist()
    elif isinstance(ids, Iterable) and not isinstance(ids, (str, bytes, dict)):
        values = list(ids)
    else:
        raise TypeError("ids must be a string/int ID or a sequence of string/int IDs")

    if not values:
        raise ValueError("ids cannot be empty")
    normalized = [_python_id(value) for value in values]
    return normalized, False


def _python_id(value: Any) -> str | int:
    if isinstance(value, bool):
        raise TypeError("bool is not a valid LynseDB ID")
    if isinstance(value, np.integer):
        value = int(value)
    if not isinstance(value, (str, int)):
        raise TypeError("IDs must be strings or integers")
    if isinstance(value, int) and value < 0:
        raise ValueError("integer IDs must be non-negative")
    if isinstance(value, str) and value == "":
        raise ValueError("string IDs cannot be empty")
    return value


def validate_unique_external_ids(ids: list[str | int]) -> None:
    seen: set[tuple[str, str | int]] = set()
    for value in ids:
        key = ("int", value) if isinstance(value, int) else ("str", value)
        if key in seen:
            raise ValueError(f"duplicate id {value!r} in the same add call")
        seen.add(key)


def normalize_documents(documents: Any, n: int | None = None) -> tuple[list[str] | None, bool]:
    if documents is None:
        return None, False
    if isinstance(documents, str):
        values = [documents]
        single = True
    elif isinstance(documents, Iterable) and not isinstance(documents, (bytes, dict)):
        values = list(documents)
        single = False
    else:
        raise TypeError("documents must be a string or a sequence of strings")
    if n is not None and len(values) != n:
        raise ValueError(f"documents length ({len(values)}) must match ids length ({n})")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("all documents must be strings")
    return values, single


def normalize_fields(fields: Any, n: int) -> list[dict[str, Any]]:
    if fields is None:
        return [{} for _ in range(n)]
    if isinstance(fields, dict):
        if n != 1:
            raise ValueError("fields must be a list of dicts when adding multiple records")
        return [dict(fields)]
    if isinstance(fields, Iterable) and not isinstance(fields, (str, bytes)):
        values = list(fields)
        if len(values) != n:
            raise ValueError(f"fields length ({len(values)}) must match ids length ({n})")
        normalized: list[dict[str, Any]] = []
        for field in values:
            if field is None:
                normalized.append({})
            elif isinstance(field, dict):
                normalized.append(dict(field))
            else:
                raise TypeError("fields entries must be dict or None")
        return normalized
    raise TypeError("fields must be a dict, a sequence of dicts, or None")


def normalize_vectors(vectors: Any, n: int) -> np.ndarray:
    if vectors is None:
        raise ValueError("vectors cannot be None")
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        if n != 1:
            raise ValueError("a single 1D vector can only be used with one ID")
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError("vectors must be a 1D vector or a 2D matrix")
    if arr.shape[0] != n:
        raise ValueError(f"vectors row count ({arr.shape[0]}) must match ids length ({n})")
    return np.ascontiguousarray(arr, dtype=np.float32)


def attach_documents(
    fields: list[dict[str, Any]],
    documents: list[str] | None,
) -> list[dict[str, Any]]:
    if documents is None:
        return fields
    output: list[dict[str, Any]] = []
    for idx, field in enumerate(fields):
        item = dict(field)
        item[DOCUMENT_FIELD] = documents[idx]
        output.append(item)
    return output


def id_array(values: list[str | int]) -> np.ndarray:
    if all(isinstance(value, int) for value in values):
        return np.array(values, dtype=np.int64)
    return np.array(values, dtype=object)
