"""Canonical index-mode list for the local 1M performance gate.

Kept importable without a built LynseDB extension so unit tests can validate
coverage against docs/tutorials/indexing.md.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEXING_DOC = ROOT / "docs" / "tutorials" / "indexing.md"

# Dense families from indexing.md + SPANN (supported in resolve_index_type).
DENSE_INDEX_MODES = [
    "FLAT-IP",
    "FLAT-L2",
    "FLAT-COS",
    "FLAT-COSINE",
    "FLAT-IP-SQ8",
    "FLAT-L2-SQ8",
    "FLAT-COS-SQ8",
    "FLAT-COSINE-SQ8",
    "HNSW-IP",
    "HNSW-L2",
    "HNSW-COS",
    "HNSW-COSINE",
    "HNSW-IP-SQ8",
    "HNSW-L2-SQ8",
    "HNSW-COS-SQ8",
    "HNSW-COSINE-SQ8",
    "DISKANN-IP",
    "DISKANN-L2",
    "DISKANN-COS",
    "DISKANN-COSINE",
    "DISKANN-IP-PQ",
    "DISKANN-L2-PQ",
    "DISKANN-COS-PQ",
    "DISKANN-IP-SQ8",
    "DISKANN-L2-SQ8",
    "DISKANN-COS-SQ8",
    "DISKANN-COSINE-SQ8",
    "IVF-IP",
    "IVF-L2",
    "IVF-COS",
    "IVF-COSINE",
    "IVF-IP-SQ8",
    "IVF-L2-SQ8",
    "IVF-COS-SQ8",
    "IVF-COSINE-SQ8",
    "SPANN-IP",
    "SPANN-L2",
    "SPANN-COS",
    "SPANN-COSINE",
    "SPANN-IP-SQ8",
    "SPANN-L2-SQ8",
    "SPANN-COS-SQ8",
    "SPANN-COSINE-SQ8",
]

DOMAIN_INDEX_MODES = [
    "FLAT-L1",
    "FLAT-HAVERSINE",
    "FLAT-CORRELATION",
    "FLAT-HELLINGER",
    "FLAT-WASSERSTEIN",
    "FLAT-JENSEN-SHANNON",
    "FLAT-CHEBYSHEV",
    "FLAT-CANBERRA",
    "FLAT-BRAY-CURTIS",
    "HNSW-L1",
    "HNSW-HAVERSINE",
    "HNSW-CORRELATION",
    "HNSW-HELLINGER",
    "HNSW-WASSERSTEIN",
    "HNSW-JENSEN-SHANNON",
    "HNSW-CHEBYSHEV",
]

FLAT_QUANT_INDEX_MODES = [
    "FLAT-IP-PQ",
    "FLAT-L2-PQ",
    "FLAT-COS-PQ",
    "FLAT-COSINE-PQ",
    "FLAT-IP-PQ8",
    "FLAT-IP-PQ16",
    "FLAT-L2-PQ8",
    "FLAT-COS-PQ8",
    "FLAT-IP-RABITQ",
    "FLAT-L2-RABITQ",
    "FLAT-COS-RABITQ",
    "FLAT-COSINE-RABITQ",
    "FLAT-IP-POLARVEC",
    "FLAT-L2-POLARVEC",
    "FLAT-COS-POLARVEC",
    "FLAT-COSINE-POLARVEC",
    "FLAT-IP-POLARVEC3",
    "FLAT-IP-POLARVEC4",
    "FLAT-IP-POLARVEC8",
]

BINARY_INDEX_MODES = [
    "FLAT-HAMMING-BINARY",
    "FLAT-HAMMING",
    "FLAT-JACCARD-BINARY",
    "FLAT-JACCARD",
    "FLAT-TANIMOTO-BINARY",
    "FLAT-TANIMOTO",
    "FLAT-DICE-BINARY",
    "FLAT-DICE",
    "IVF-HAMMING-BINARY",
    "IVF-HAMMING",
    "IVF-JACCARD-BINARY",
    "IVF-JACCARD",
]

ALL_INDEX_MODES = (
    DENSE_INDEX_MODES
    + DOMAIN_INDEX_MODES
    + FLAT_QUANT_INDEX_MODES
    + BINARY_INDEX_MODES
)

SPANN_INDEX_MODES = [
    "SPANN-IP",
    "SPANN-L2",
    "SPANN-COS",
    "SPANN-COSINE",
    "SPANN-IP-SQ8",
    "SPANN-L2-SQ8",
    "SPANN-COS-SQ8",
    "SPANN-COSINE-SQ8",
]

# Focused subset for paper-fidelity / recall fixes regression tracking.
# Covers HNSW heuristic+ef wiring, DiskANN Vamana build, IVF/SPANN nprobe
# defaults, SQ8 re-rank paths, and Binary threshold — plus FLAT controls.
PAPER_FIX_INDEX_MODES = [
    "FLAT-IP",
    "FLAT-IP-SQ8",
    "FLAT-HAMMING",
    "HNSW-IP",
    "HNSW-L2",
    "HNSW-IP-SQ8",
    "DISKANN-IP",
    "DISKANN-L2",
    "DISKANN-IP-PQ",
    "DISKANN-IP-SQ8",
    "IVF-IP",
    "IVF-IP-SQ8",
    "IVF-HAMMING",
    "SPANN-IP",
    "SPANN-IP-SQ8",
]

# Dense-only subset for ANN_SIFT1M / sift_learn benches (no binary collections).
SIFT_PAPER_FIX_INDEX_MODES = [
    "FLAT-L2",
    "FLAT-IP",
    "FLAT-L2-SQ8",
    "FLAT-IP-SQ8",
    "HNSW-L2",
    "HNSW-IP",
    "HNSW-L2-SQ8",
    "HNSW-IP-SQ8",
    "DISKANN-L2",
    "DISKANN-IP",
    "DISKANN-L2-PQ",
    "DISKANN-IP-PQ",
    "IVF-L2",
    "IVF-IP",
    "IVF-L2-SQ8",
    "IVF-IP-SQ8",
    "SPANN-L2",
    "SPANN-IP",
]


def normalize_doc_alias(alias: str) -> str:
    """Normalize documented aliases to the uppercase form used in the gate."""
    return alias.strip().upper().replace("DISKANN-", "DISKANN-")


def documented_build_index_aliases(doc_text: str | None = None) -> list[str]:
    text = doc_text if doc_text is not None else INDEXING_DOC.read_text(encoding="utf-8")
    return [normalize_doc_alias(m) for m in re.findall(r'build_index\("([^"]+)"', text)]


def metric_for_mode(mode: str) -> str:
    upper = mode.upper()
    if "JACCARD" in upper or "TANIMOTO" in upper:
        return "JACCARD"
    if "DICE" in upper or "SORENSEN" in upper:
        return "DICE"
    if "HAMMING" in upper:
        return "HAMMING"
    if "HAVERSINE" in upper:
        return "HAVERSINE"
    if "CORRELATION" in upper:
        return "CORRELATION"
    if "HELLINGER" in upper:
        return "HELLINGER"
    if "WASSERSTEIN" in upper:
        return "WASSERSTEIN"
    if "JENSEN-SHANNON" in upper or "JENSEN_SHANNON" in upper:
        return "JENSEN-SHANNON"
    if "CHEBYSHEV" in upper:
        return "CHEBYSHEV"
    if "CANBERRA" in upper:
        return "CANBERRA"
    if "BRAY-CURTIS" in upper or "BRAY_CURTIS" in upper:
        return "BRAY-CURTIS"
    if "L1" in upper or "MANHATTAN" in upper:
        return "L1"
    if "L2" in upper:
        return "L2"
    if "COS" in upper:
        return "COS"
    return "IP"


def collection_kind_for_mode(mode: str) -> str:
    metric = metric_for_mode(mode)
    if metric in {"HAMMING", "JACCARD", "DICE"}:
        return "binary"
    if metric == "HAVERSINE":
        return "haversine"
    if metric in {"HELLINGER", "WASSERSTEIN", "JENSEN-SHANNON"}:
        return "distribution"
    return "dense"


def exact_mode_for_metric(metric: str) -> str:
    return {
        "IP": "FLAT-IP",
        "L2": "FLAT-L2",
        "COS": "FLAT-COS",
        "L1": "FLAT-L1",
        "HAVERSINE": "FLAT-HAVERSINE",
        "CORRELATION": "FLAT-CORRELATION",
        "HELLINGER": "FLAT-HELLINGER",
        "WASSERSTEIN": "FLAT-WASSERSTEIN",
        "JENSEN-SHANNON": "FLAT-JENSEN-SHANNON",
        "CHEBYSHEV": "FLAT-CHEBYSHEV",
        "CANBERRA": "FLAT-CANBERRA",
        "BRAY-CURTIS": "FLAT-BRAY-CURTIS",
        "JACCARD": "FLAT-JACCARD-BINARY",
        "HAMMING": "FLAT-HAMMING-BINARY",
        "DICE": "FLAT-DICE-BINARY",
    }[metric]


def n_clusters_for(n: int) -> int:
    return max(64, min(1024, int(math.sqrt(max(1, n)))))


def recall_floor_for_mode(mode: str) -> float:
    upper = mode.upper()
    is_flat = upper.startswith("FLAT-")
    is_quant = any(token in upper for token in ("-SQ8", "-PQ", "-RABITQ", "-POLARVEC"))
    if is_flat and not is_quant:
        return 0.999
    if "-SQ8" in upper:
        return 0.95
    if any(token in upper for token in ("-PQ", "-RABITQ", "-POLARVEC")):
        return 0.70
    if upper.startswith(("HNSW-", "DISKANN-")):
        return 0.90
    if upper.startswith(("IVF-", "SPANN-")):
        return 0.85
    return 0.85
