"""Ensure the local perf-gate mode list covers documented index aliases."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

from gate_index_modes import (  # noqa: E402
    ALL_INDEX_MODES,
    PAPER_FIX_INDEX_MODES,
    SIFT_PAPER_FIX_INDEX_MODES,
    SPANN_INDEX_MODES,
    documented_build_index_aliases,
)


def test_all_index_modes_cover_documented_aliases():
    documented = documented_build_index_aliases()
    assert documented, "expected build_index aliases in indexing.md"
    missing = sorted(set(documented) - set(ALL_INDEX_MODES))
    assert not missing, f"perf gate missing documented aliases: {missing}"


def test_all_index_modes_include_spann_family():
    missing = sorted(set(SPANN_INDEX_MODES) - set(ALL_INDEX_MODES))
    assert not missing, f"perf gate missing SPANN aliases: {missing}"


def test_all_index_modes_unique():
    assert len(ALL_INDEX_MODES) == len(set(ALL_INDEX_MODES))


def test_paper_fix_modes_are_subset_of_all_and_unique():
    assert len(PAPER_FIX_INDEX_MODES) == len(set(PAPER_FIX_INDEX_MODES))
    missing = sorted(set(PAPER_FIX_INDEX_MODES) - set(ALL_INDEX_MODES))
    assert not missing, f"paper-fix modes not in ALL_INDEX_MODES: {missing}"


def test_sift_paper_fix_modes_are_subset_of_all_and_unique():
    assert len(SIFT_PAPER_FIX_INDEX_MODES) == len(set(SIFT_PAPER_FIX_INDEX_MODES))
    missing = sorted(set(SIFT_PAPER_FIX_INDEX_MODES) - set(ALL_INDEX_MODES))
    assert not missing, f"sift paper-fix modes not in ALL_INDEX_MODES: {missing}"
    assert not any("HAMMING" in m or "JACCARD" in m for m in SIFT_PAPER_FIX_INDEX_MODES)
