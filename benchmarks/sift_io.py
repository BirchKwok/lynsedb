"""Load ANN_SIFT1M-style .fvecs / .ivecs files for local gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_fvecs(path: Path | str, *, max_rows: int | None = None) -> np.ndarray:
    """Read little-endian fvecs into float32 array of shape (n, dim)."""
    path = Path(path)
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty fvecs file: {path}")
    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"invalid fvecs dim={dim} in {path}")
    row_stride = dim + 1
    if raw.size % row_stride != 0:
        raise ValueError(
            f"fvecs size {raw.size} not divisible by dim+1={row_stride} ({path})"
        )
    n = raw.size // row_stride
    if max_rows is not None:
        n = min(n, max_rows)
    mat = raw[: n * row_stride].reshape(n, row_stride)
    if not np.all(mat[:, 0] == dim):
        raise ValueError(f"inconsistent dims inside {path}")
    return mat[:, 1:].view(np.float32).copy()


def read_ivecs(path: Path | str, *, max_rows: int | None = None) -> np.ndarray:
    """Read little-endian ivecs into int32 array of shape (n, k)."""
    path = Path(path)
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty ivecs file: {path}")
    k = int(raw[0])
    if k <= 0:
        raise ValueError(f"invalid ivecs width={k} in {path}")
    row_stride = k + 1
    if raw.size % row_stride != 0:
        raise ValueError(
            f"ivecs size {raw.size} not divisible by k+1={row_stride} ({path})"
        )
    n = raw.size // row_stride
    if max_rows is not None:
        n = min(n, max_rows)
    mat = raw[: n * row_stride].reshape(n, row_stride)
    if not np.all(mat[:, 0] == k):
        raise ValueError(f"inconsistent widths inside {path}")
    return mat[:, 1:].copy()


def load_sift_dataset(
    sift_dir: Path | str,
    *,
    rows: int | None = None,
    query_limit: int | None = None,
    base_name: str = "sift_base.fvecs",
    query_name: str = "sift_query.fvecs",
    gt_name: str = "sift_groundtruth.ivecs",
    prefer_learn_if_rows_le: int = 100_000,
) -> dict:
    """Load SIFT base/query/(optional) groundtruth.

    When ``rows <= 100_000`` and ``sift_learn.fvecs`` exists, use learn as base
    (ANN_SIFT1M learn set is 100k) so mid-size benches stay fast. Official
    groundtruth applies only to the full 1M ``sift_base.fvecs``.
    """
    sift_dir = Path(sift_dir)
    learn_path = sift_dir / "sift_learn.fvecs"
    base_path = sift_dir / base_name
    if (
        rows is not None
        and rows <= prefer_learn_if_rows_le
        and learn_path.exists()
        and rows <= 100_000
    ):
        base_path = learn_path

    base = read_fvecs(base_path, max_rows=rows)
    queries = read_fvecs(sift_dir / query_name, max_rows=query_limit)
    gt_path = sift_dir / gt_name
    groundtruth = None
    # Official GT is only valid for the full 1M base file.
    if gt_path.exists() and base_path.name == "sift_base.fvecs" and base.shape[0] == 1_000_000:
        groundtruth = read_ivecs(gt_path, max_rows=queries.shape[0])

    return {
        "base": base,
        "queries": queries,
        "groundtruth": groundtruth,
        "base_path": str(base_path),
        "dim": int(base.shape[1]),
        "n_base": int(base.shape[0]),
        "n_queries": int(queries.shape[0]),
    }
