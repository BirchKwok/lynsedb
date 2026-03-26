"""
ResultView — unified result container for all LynseDB query operations.

Wraps search results (ids, distances), data results (vectors, ids), and field
metadata into a single object with rich previewing and zero-copy conversions.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _parse_index_mode(index_mode: Optional[str]) -> Tuple[str, str]:
    """Extract (index_type, distance_metric) from an index mode string.

    Examples:
        'FLAT'            -> ('Flat', 'IP')
        'FLAT-L2'         -> ('Flat', 'L2')
        'FLAT-COS-SQ8'    -> ('Flat', 'Cosine')
        'IVF-L2'          -> ('IVF', 'L2')
        'IVF-HAMMING-BINARY' -> ('IVF', 'Hamming')
    """
    if not index_mode:
        return ("Flat", "IP")
    parts = index_mode.upper().split("-")
    idx_type = parts[0]  # FLAT / IVF / HNSW / DISKANN
    idx_type_map = {"FLAT": "Flat", "IVF": "IVF", "HNSW": "HNSW", "DISKANN": "DiskANN"}
    idx_type = idx_type_map.get(idx_type, idx_type)

    # Match Rust's metric_from_mode_str: uses contains() on the full string
    full = "-".join(parts[1:])  # e.g. "L2SQ-SQ8" or "COS" or "IP"
    if "JACCARD" in full:
        metric = "Jaccard"
    elif "HAMMING" in full:
        metric = "Hamming"
    elif "L2" in full:
        metric = "L2"
    elif "COS" in full:
        metric = "Cosine"
    else:
        metric = "IP"
    return idx_type, metric


class ResultView:
    """Unified result container for LynseDB query / search / head / tail operations.

    Holds any combination of:
      - ``ids``        — ``np.ndarray[int64]``
      - ``distances``  — ``np.ndarray[float32]``
      - ``vectors``    — ``np.ndarray[float32]`` with shape ``(n, dim)``
      - ``fields``     — ``list[dict]``

    All stored arrays are kept as-is (zero-copy).  The ``to_*`` conversion
    methods avoid unnecessary copies whenever the target library supports it.

    Supports ``len()``, indexing (``result[0]``), equality, iteration (tuple
    unpacking), and a rich ``__repr__`` preview.
    """

    __slots__ = (
        "_ids", "_distances", "_vectors", "_fields",
        "_k", "_distance", "_index", "_result_type",
        "_components",
    )

    def __init__(
        self,
        *,
        ids: Optional[np.ndarray] = None,
        distances: Optional[np.ndarray] = None,
        vectors: Optional[np.ndarray] = None,
        fields: Optional[List[Dict[str, Any]]] = None,
        k: Optional[int] = None,
        distance: Optional[str] = None,
        index: Optional[str] = None,
        result_type: str = "search",
    ):
        # Store arrays directly — zero-copy from caller
        self._ids = ids
        self._distances = distances
        self._vectors = vectors
        self._fields = fields if fields is not None else []
        self._k = k
        self._distance = distance
        self._index = index
        self._result_type = result_type  # "search" | "data" | "query"

        # Pre-build the ordered component list for __getitem__ / __iter__
        self._components = self._build_components()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_components(self) -> list:
        """Build ordered list of components for indexing / iteration.

        Always includes all components so that tuple unpacking works:
          - search:  ``ids, dists, fields = result``
          - data:    ``vecs, ids, fields = result``
          - query:   ``ids, fields = result``  (or just ``(ids,)`` if no fields)
        """
        if self._result_type == "search":
            # search: always (ids, distances, fields) for backward compat
            return [self._ids, self._distances, self._fields]
        elif self._result_type == "data":
            # head / tail / query_vectors / read_by_id: always (vectors, ids, fields)
            return [self._vectors, self._ids, self._fields]
        else:
            # query: (ids, fields) or (ids,)
            if self._fields:
                return [self._ids, self._fields]
            return [self._ids]

    # ------------------------------------------------------------------
    # Properties — zero-copy accessors
    # ------------------------------------------------------------------

    @property
    def ids(self) -> Optional[np.ndarray]:
        return self._ids

    @property
    def distances(self) -> Optional[np.ndarray]:
        return self._distances

    @property
    def vectors(self) -> Optional[np.ndarray]:
        return self._vectors

    @property
    def fields(self) -> List[Dict[str, Any]]:
        return self._fields

    @property
    def k(self) -> Optional[int]:
        return self._k

    @property
    def distance_metric(self) -> Optional[str]:
        return self._distance

    @property
    def index_type(self) -> Optional[str]:
        return self._index

    @property
    def result_type(self) -> str:
        return self._result_type

    # ------------------------------------------------------------------
    # Dunder protocols
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._ids is not None:
            return len(self._ids)
        if self._distances is not None:
            return len(self._distances)
        if self._vectors is not None:
            return self._vectors.shape[0]
        if self._fields:
            return len(self._fields)
        return 0

    def __getitem__(self, index):
        """Support ``result[i]`` for component access (backward-compatible).

        - Integer index returns the i-th component (e.g. 0→ids, 1→distances).
        - Slice returns a new ResultView with a row-slice of the data.
        """
        if isinstance(index, int):
            if index < 0:
                index += len(self._components)
            if 0 <= index < len(self._components):
                return self._components[index]
            raise IndexError(f"Index {index} out of range (0-{len(self._components) - 1})")

        if isinstance(index, slice):
            return self._slice(index)

        raise TypeError(f"Indices must be integers or slices, not {type(index).__name__}")

    def _slice(self, s: slice) -> "ResultView":
        """Return a new ResultView with a row-slice of data."""
        new_ids = self._ids[s] if self._ids is not None else None
        new_dists = self._distances[s] if self._distances is not None else None
        new_vecs = self._vectors[s] if self._vectors is not None else None
        new_fields = self._fields[s] if self._fields else []
        new_k = len(new_ids) if new_ids is not None else (
            len(new_dists) if new_dists is not None else None
        )
        return ResultView(
            ids=new_ids, distances=new_dists, vectors=new_vecs, fields=new_fields,
            k=new_k, distance=self._distance, index=self._index,
            result_type=self._result_type,
        )

    def __iter__(self):
        """Support tuple unpacking: ``ids, dists, fields = result``."""
        yield from self._components

    def __eq__(self, other) -> bool:
        if not isinstance(other, ResultView):
            return NotImplemented
        if self._result_type != other._result_type:
            return False
        if len(self) != len(other):
            return False
        # Compare arrays
        if (self._ids is None) != (other._ids is None):
            return False
        if self._ids is not None and not np.array_equal(self._ids, other._ids):
            return False
        if (self._distances is None) != (other._distances is None):
            return False
        if self._distances is not None and not np.array_equal(self._distances, other._distances):
            return False
        if (self._vectors is None) != (other._vectors is None):
            return False
        if self._vectors is not None and not np.array_equal(self._vectors, other._vectors):
            return False
        if self._fields != other._fields:
            return False
        return True

    def __bool__(self) -> bool:
        return len(self) > 0

    def __repr__(self) -> str:
        n = len(self)
        parts = []

        if self._result_type == "search":
            parts.append(_compact_array_repr(self._ids))
            parts.append(_compact_array_repr(self._distances))
        elif self._result_type == "data":
            parts.append(_compact_array_repr(self._vectors))
            parts.append(_compact_array_repr(self._ids))
        else:
            if self._ids is not None:
                parts.append(_compact_array_repr(self._ids))
            elif self._fields:
                parts.append(f"[{len(self._fields)} records]")

        meta = []
        if self._k is not None:
            meta.append(f"k={self._k}")
        if self._distance:
            meta.append(f'distance="{self._distance}"')
        if self._index:
            meta.append(f'index="{self._index}"')
        if self._result_type == "data":
            meta.append(f"n={n}")

        meta_str = ", ".join(meta)
        if meta_str:
            meta_str = ", " + meta_str

        # Compact single-line form when body is short and has no newlines
        inline_body = ", ".join(parts)
        total_len = len("ResultView([") + len(inline_body) + len("]") + len(meta_str) + len(")")
        if total_len < 120 and "\n" not in inline_body:
            return f"ResultView([{inline_body}]{meta_str})"

        # Multi-line form
        indented = "\n".join(f"    {p}," for p in parts)
        return f"ResultView(\n    [\n{indented}\n    ]{meta_str}\n)"

    # ------------------------------------------------------------------
    # Conversion methods — zero-copy where possible
    # ------------------------------------------------------------------

    def to_tuple(self) -> tuple:
        """Return components as a plain tuple (zero-copy, arrays shared)."""
        return tuple(self._components)

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Return a dict of numpy arrays (zero-copy, arrays shared).

        Keys present depend on result type: 'ids', 'distances', 'vectors'.
        """
        out = {}
        if self._ids is not None:
            out["ids"] = self._ids
        if self._distances is not None:
            out["distances"] = self._distances
        if self._vectors is not None:
            out["vectors"] = self._vectors
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a columnar dict.

        For search results: ``{'ids': [...], 'distances': [...], ...field columns...}``
        For data results: ``{'vectors': [...], 'ids': [...], ...field columns...}``
        """
        d: Dict[str, Any] = {}

        if self._result_type == "search":
            if self._ids is not None:
                d["ids"] = self._ids.tolist()
            if self._distances is not None:
                d["distances"] = self._distances.tolist()
        elif self._result_type == "data":
            if self._vectors is not None:
                d["vectors"] = self._vectors.tolist()
            if self._ids is not None:
                d["ids"] = self._ids.tolist()
        else:
            if self._ids is not None:
                d["ids"] = self._ids.tolist() if isinstance(self._ids, np.ndarray) else list(self._ids)

        if self._fields:
            all_keys: set = set()
            for f in self._fields:
                if f:
                    all_keys.update(f.keys())
            for key in sorted(all_keys):
                d[key] = [f.get(key) if f else None for f in self._fields]

        return d

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to a list of row-dicts.

        Each dict contains the row data (id, distance/vector, field values).
        """
        result = []
        n = len(self)
        for i in range(n):
            entry: Dict[str, Any] = {}
            if self._ids is not None:
                entry["id"] = int(self._ids[i])
            if self._distances is not None:
                entry["distance"] = float(self._distances[i])
            if self._vectors is not None:
                entry["vector"] = self._vectors[i].tolist()
            if i < len(self._fields) and self._fields[i]:
                entry.update(self._fields[i])
            result.append(entry)
        return result

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_list(), **kwargs)

    def to_pandas(self):
        """Convert to a ``pandas.DataFrame`` (zero-copy for numeric columns).

        Requires ``pandas`` to be installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_pandas(). "
                "Install it with: pip install pandas"
            )
        d: Dict[str, Any] = {}
        if self._result_type == "search":
            if self._ids is not None:
                d["id"] = self._ids        # zero-copy: pandas wraps ndarray
            if self._distances is not None:
                d["distance"] = self._distances
        elif self._result_type == "data":
            if self._ids is not None:
                d["id"] = self._ids
            if self._vectors is not None:
                d["vector"] = list(self._vectors)  # list of 1-D arrays
        else:
            if self._ids is not None:
                d["id"] = self._ids if isinstance(self._ids, np.ndarray) else list(self._ids)

        if self._fields:
            all_keys: set = set()
            for f in self._fields:
                if f:
                    all_keys.update(f.keys())
            for key in sorted(all_keys):
                d[key] = [f.get(key) if f else None for f in self._fields]

        return pd.DataFrame(d)

    def to_polars(self):
        """Convert to a ``polars.DataFrame`` (zero-copy for numeric columns).

        Requires ``polars`` to be installed.
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is required for to_polars(). "
                "Install it with: pip install polars"
            )
        d: Dict[str, Any] = {}
        if self._result_type == "search":
            if self._ids is not None:
                d["id"] = pl.Series("id", self._ids, dtype=pl.Int64)
            if self._distances is not None:
                d["distance"] = pl.Series("distance", self._distances, dtype=pl.Float32)
        elif self._result_type == "data":
            if self._ids is not None:
                d["id"] = pl.Series("id", self._ids, dtype=pl.Int64)
            if self._vectors is not None:
                d["vector"] = pl.Series("vector", self._vectors.tolist())
        else:
            if self._ids is not None:
                arr = self._ids if isinstance(self._ids, np.ndarray) else np.array(self._ids, dtype=np.int64)
                d["id"] = pl.Series("id", arr, dtype=pl.Int64)

        if self._fields:
            all_keys: set = set()
            for f in self._fields:
                if f:
                    all_keys.update(f.keys())
            for key in sorted(all_keys):
                d[key] = [f.get(key) if f else None for f in self._fields]

        return pl.DataFrame(d)

    def to_arrow(self):
        """Convert to a ``pyarrow.Table`` (zero-copy for numeric columns).

        Requires ``pyarrow`` to be installed.
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                "pyarrow is required for to_arrow(). "
                "Install it with: pip install pyarrow"
            )
        arrays = {}
        if self._result_type == "search":
            if self._ids is not None:
                arrays["id"] = pa.array(self._ids, type=pa.int64())
            if self._distances is not None:
                arrays["distance"] = pa.array(self._distances, type=pa.float32())
        elif self._result_type == "data":
            if self._ids is not None:
                arrays["id"] = pa.array(self._ids, type=pa.int64())
            if self._vectors is not None:
                dim = self._vectors.shape[1] if self._vectors.ndim == 2 else 0
                flat = self._vectors.ravel()
                inner = pa.array(flat, type=pa.float32())
                offsets = pa.array(
                    list(range(0, len(flat) + 1, dim)), type=pa.int32()
                )
                list_arr = pa.ListArray.from_arrays(offsets, inner)
                arrays["vector"] = list_arr
        else:
            if self._ids is not None:
                arr = self._ids if isinstance(self._ids, np.ndarray) else np.array(self._ids, dtype=np.int64)
                arrays["id"] = pa.array(arr, type=pa.int64())

        if self._fields:
            all_keys: set = set()
            for f in self._fields:
                if f:
                    all_keys.update(f.keys())
            for key in sorted(all_keys):
                vals = [f.get(key) if f else None for f in self._fields]
                arrays[key] = pa.array(vals)

        return pa.table(arrays)


# ------------------------------------------------------------------
# Helper: compact array repr (truncated, single-line for short arrays)
# ------------------------------------------------------------------

def _compact_array_repr(arr) -> str:
    """Produce a compact repr of a numpy array, similar to numpy's default."""
    if arr is None:
        return "None"
    return repr(arr)
