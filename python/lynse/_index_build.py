"""Shared helpers for ``build_index(**kwargs)`` across local / HTTP clients.

Keyword arguments
-----------------
Unknown names raise ``ValueError``. Keys that do not apply to the selected
index family are ignored (so one shared kwargs dict can be reused across modes).

**IVF / SPANN**

- ``n_clusters`` / ``n_centroids`` (int, default ``256``):
  Number of coarse centroids / partitions.
- ``nprobe`` (int, default ``32``):
  Default number of partitions probed at search time (overridable per query).
- ``replica_count`` (int, default ``1``, SPANN only):
  Boundary replica count for SPANN posting lists.

**HNSW**

- ``m`` (int, default ``16``):
  Max neighbors per layer (graph degree).
- ``ef_construction`` (int, default ``128``):
  Candidate list size while building; higher → better recall, slower build.
- ``ef_search`` (int, default ``50``):
  Default search beam width (overridable per query via ``nprobe`` / ef).
- ``max_level`` (int, optional):
  Cap on the maximum HNSW layer; omit to use the index's internal default.

**DiskANN**

- ``r`` (int, default ``16``):
  Target out-degree (R) of the Vamana graph.
- ``l`` (int, default ``64``):
  Search/build beam width (L); build may further cap ``L_build``.
- ``alpha`` (float, default ``1.2``, must be ``>= 1.0``):
  Robust-prune expansion factor.
- ``max_degree`` (int, default equals ``r``):
  Hard cap on adjacency list length after prune.

**Flat / PQ / RaBitQ / PolarVec**

- No build kwargs; any known family keys are ignored.
"""

from __future__ import annotations

from typing import Any, Mapping

# Keys accepted by the Rust ``IndexBuildOptions`` parser.
KNOWN_BUILD_KEYS = frozenset(
    {
        "n_clusters",
        "n_centroids",
        "m",
        "ef_construction",
        "ef_search",
        "max_level",
        "r",
        "l",
        "alpha",
        "max_degree",
        "nprobe",
        "replica_count",
    }
)

# Per-family keys that are actually applied (others are ignored for shared kwargs).
FAMILY_BUILD_KEYS = {
    "FLAT": frozenset(),
    "HNSW": frozenset({"m", "ef_construction", "ef_search", "max_level"}),
    "DISKANN": frozenset({"r", "l", "alpha", "max_degree"}),
    "IVF": frozenset({"n_clusters", "n_centroids", "nprobe"}),
    "SPANN": frozenset({"n_clusters", "n_centroids", "nprobe", "replica_count"}),
}


def index_family(index_mode: str) -> str:
    upper = index_mode.upper()
    for family in ("DISKANN", "HNSW", "SPANN", "IVF", "FLAT"):
        if upper.startswith(family):
            return family
    return "FLAT"


def normalize_build_kwargs(index_mode: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate kwargs and drop ``None`` values.

    Unknown keys raise ``ValueError``. Keys that belong to another index family
    are kept in the dict so the Rust layer can filter them (or callers can
    pre-filter with :func:`applicable_build_kwargs`).
    """
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key not in KNOWN_BUILD_KEYS:
            raise ValueError(
                f"unknown index build parameter {key!r}; "
                f"supported keys: {', '.join(sorted(KNOWN_BUILD_KEYS))}"
            )
        out[key] = value
    # Touch family so typos in mode still surface early for empty kwargs paths.
    _ = index_family(index_mode)
    return out


def applicable_build_kwargs(index_mode: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return only kwargs that apply to ``index_mode``'s family."""
    allowed = FAMILY_BUILD_KEYS[index_family(index_mode)]
    normalized = normalize_build_kwargs(index_mode, kwargs)
    return {k: v for k, v in normalized.items() if k in allowed}
