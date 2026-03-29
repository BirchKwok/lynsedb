"""
Comprehensive LynseDB index benchmark  —  ALL index types across ALL metrics.

Datasets:
  Float  (1M × 128) : SIFT-1M from /Users/guobingming/Downloads/sift
  Binary (100K × 128): synthetic random 0/1 float32 vectors (generated in-process)

Ground truth per metric family:
  L2      — official SIFT .ivecs ground truth
  IP      — FLAT  brute-force (computed once, cached)
  COS     — FLAT-COS brute-force (computed once, cached)
  JACCARD — FLAT-JACCARD-BINARY on binary collection (computed once, cached)
  HAMMING — FLAT-HAMMING-BINARY on binary collection (computed once, cached)
"""

import os
import time

import numpy as np
import lynse

# ─── Configuration ────────────────────────────────────────────────────────────

SIFT_DIR      = "/Users/guobingming/Downloads/sift"
DB_PATH       = "/tmp/lynse_comp_bench"
DB_NAME       = "comp_bench"
COLL_NAME     = "sift1m"
BIN_COLL_NAME = "bin100k"

DIM         = 128
N_BASE      = 1_000_000
N_BIN_BASE  = 100_000
K           = 10
N_QUERIES   = 100   # 100 queries balances accuracy with runtime across 40+ indices

# (index_mode, short_description, metric_group, nprobe)
#   metric_group: 'l2' | 'ip' | 'cos' | 'jaccard' | 'hamming'
#   nprobe: ef_search for HNSW; nprobe for IVF/DiskANN; ignored for Flat family
INDEX_MODES = [
    # ── L2 ──────────────────────────────────────────────────────────────────
    ("FLAT-L2",              "Exact L2",          "l2",      64),
    ("FLAT-L2-SQ8",          "SQ8",               "l2",      64),
    ("FLAT-L2-PQ",           "PQ (auto sub)",     "l2",      64),
    ("FLAT-L2-PQ8",          "PQ-8",              "l2",      64),
    ("FLAT-L2-PQ16",         "PQ-16",             "l2",      64),
    ("FLAT-L2-RABITQ",       "RaBitQ",            "l2",      64),
    ("FLAT-L2-POLARVEC3",    "PolarVec-3b",       "l2",      64),
    ("FLAT-L2-POLARVEC",     "PolarVec-4b",       "l2",      64),
    ("FLAT-L2-POLARVEC8",    "PolarVec-8b",       "l2",      64),
    ("HNSW-L2",              "HNSW",              "l2",      64),
    ("HNSW-L2-SQ8",          "HNSW+SQ8",          "l2",      64),
    ("IVF-L2",               "IVF",               "l2",      20),
    ("IVF-L2-SQ8",           "IVF+SQ8",           "l2",      20),
    ("DiskANN-L2",           "DiskANN",           "l2",      64),
    ("DiskANN-L2-SQ8",       "DiskANN+SQ8",       "l2",      64),
    # ── IP ──────────────────────────────────────────────────────────────────
    ("FLAT",                 "Exact IP",          "ip",      64),
    ("FLAT-IP-SQ8",          "SQ8",               "ip",      64),
    ("FLAT-IP-PQ",           "PQ (auto sub)",     "ip",      64),
    ("FLAT-IP-PQ8",          "PQ-8",              "ip",      64),
    ("FLAT-IP-PQ16",         "PQ-16",             "ip",      64),
    ("FLAT-IP-RABITQ",       "RaBitQ",            "ip",      64),
    ("FLAT-IP-POLARVEC3",    "PolarVec-3b",       "ip",      64),
    ("FLAT-IP-POLARVEC",     "PolarVec-4b",       "ip",      64),
    ("FLAT-IP-POLARVEC8",    "PolarVec-8b",       "ip",      64),
    ("HNSW",                 "HNSW",              "ip",      64),
    ("HNSW-IP-SQ8",          "HNSW+SQ8",          "ip",      64),
    ("IVF",                  "IVF",               "ip",      20),
    ("IVF-IP-SQ8",           "IVF+SQ8",           "ip",      20),
    ("DiskANN",              "DiskANN",           "ip",      64),
    ("DiskANN-IP-SQ8",       "DiskANN+SQ8",       "ip",      64),
    # ── COS ─────────────────────────────────────────────────────────────────
    ("FLAT-COS",             "Exact COS",         "cos",     64),
    ("FLAT-COS-SQ8",         "SQ8",               "cos",     64),
    ("FLAT-COS-PQ",          "PQ (auto sub)",     "cos",     64),
    ("FLAT-COS-RABITQ",      "RaBitQ",            "cos",     64),
    ("FLAT-COS-POLARVEC",    "PolarVec-4b",       "cos",     64),
    ("HNSW-Cos",             "HNSW",              "cos",     64),
    ("HNSW-Cos-SQ8",         "HNSW+SQ8",          "cos",     64),
    ("IVF-COS",              "IVF",               "cos",     20),
    ("IVF-COS-SQ8",          "IVF+SQ8",           "cos",     20),
    ("DiskANN-Cos",          "DiskANN",           "cos",     64),
    ("DiskANN-Cos-SQ8",      "DiskANN+SQ8",       "cos",     64),
    # ── JACCARD (binary collection) ──────────────────────────────────────────
    ("FLAT-JACCARD-BINARY",  "Exact Jaccard",     "jaccard", 64),
    ("IVF-JACCARD-BINARY",   "IVF Jaccard",       "jaccard", 20),
    # ── HAMMING (binary collection) ──────────────────────────────────────────
    ("FLAT-HAMMING-BINARY",  "Exact Hamming",     "hamming", 64),
    ("IVF-HAMMING-BINARY",   "IVF Hamming",       "hamming", 20),
]

BINARY_GROUPS = {"jaccard", "hamming"}

# ─── File readers ─────────────────────────────────────────────────────────────

def read_fvecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0]); stride = 1 + d; n = len(raw) // stride
    return np.ascontiguousarray(raw.view(np.float32).reshape(n, stride)[:, 1:], dtype=np.float32)

def read_ivecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0]); stride = 1 + d; n = len(raw) // stride
    return np.ascontiguousarray(raw.reshape(n, stride)[:, 1:], dtype=np.int32)

# ─── Recall helper ────────────────────────────────────────────────────────────

def compute_recall(result_ids_list, gt_ids_list, k: int) -> float:
    """Average Recall@k. gt_ids_list is list/array of ground-truth ID arrays."""
    total = 0.0
    for ret_ids, gt_row in zip(result_ids_list, gt_ids_list):
        gt_set = set(int(x) for x in gt_row[:k])
        total += sum(1 for rid in ret_ids[:k] if int(rid) in gt_set) / k
    return total / len(result_ids_list)

# ─── Search helper ────────────────────────────────────────────────────────────

def run_search(rust_coll, queries: np.ndarray, k: int, nprobe: int):
    """Sequential single-query search. Returns (list_of_id_arrays, median_ms, qps)."""
    latencies, all_ids = [], []
    for q in queries:
        qv = np.ascontiguousarray(q, dtype=np.float32)
        t0 = time.perf_counter()
        res = rust_coll.search(qv, k=k, where=None, nprobe=nprobe)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_ids.append(res.ids)
    med = float(np.median(latencies))
    return all_ids, med, (1000.0 / med if med > 0 else 0.0)

def compute_brute_force_gt(coll, rust_coll, queries, exact_index_mode, k, nprobe=64):
    """Build exact index on coll, run queries, return list of ID arrays."""
    coll.build_index(exact_index_mode)
    all_ids, _, _ = run_search(rust_coll, queries, k=k * 10, nprobe=nprobe)
    return all_ids  # each entry has up to k*10 IDs

# ─── Setup client / DB ────────────────────────────────────────────────────────

print("=" * 80)
print("LynseDB Comprehensive Index Benchmark  —  ALL index types  —  Recall@10")
print("=" * 80)

client = lynse.VectorDBClient(DB_PATH)
try:    db = client.get_database(DB_NAME)
except ValueError:
    db = client.create_database(DB_NAME)

# ─── Float collection (SIFT-1M) ───────────────────────────────────────────────

collection = db.require_collection(COLL_NAME, dim=DIM)
n_existing = collection.shape[0]

if n_existing < N_BASE:
    print(f"\nLoading SIFT base vectors from: {SIFT_DIR}")
    base_vecs = read_fvecs(os.path.join(SIFT_DIR, "sift_base.fvecs"))
    assert len(base_vecs) >= N_BASE, f"Expected ≥{N_BASE} vectors, got {len(base_vecs)}"
    if n_existing > 0:
        collection = db.require_collection(COLL_NAME, dim=DIM, drop_if_exists=True)
    BATCH = 100_000
    print(f"  Inserting {N_BASE:,} vectors (batch={BATCH:,}) ...")
    rc = collection._rust_coll
    t0 = time.perf_counter()
    for start in range(0, N_BASE, BATCH):
        end = min(start + BATCH, N_BASE)
        rc.add_items(base_vecs[start:end], None)
        rc.commit()
        print(f"\r  {end:>9,}/{N_BASE:,}  {(time.perf_counter()-t0):.0f}s",
              end="", flush=True)
    print(f"\n  Done ({time.perf_counter()-t0:.1f}s)")
    del base_vecs
else:
    print(f"\nFloat collection: {n_existing:,} vectors already present.")

rust_coll = collection._rust_coll

# ─── Binary collection (synthetic 100K) ──────────────────────────────────────

bin_coll = db.require_collection(BIN_COLL_NAME, dim=DIM)
n_bin_existing = bin_coll.shape[0]

if n_bin_existing < N_BIN_BASE:
    print(f"\nGenerating {N_BIN_BASE:,} synthetic binary vectors (0/1 float32) ...")
    rng = np.random.default_rng(seed=12345)
    bin_vecs = rng.integers(0, 2, size=(N_BIN_BASE, DIM)).astype(np.float32)
    if n_bin_existing > 0:
        bin_coll = db.require_collection(BIN_COLL_NAME, dim=DIM, drop_if_exists=True)
    rc_bin = bin_coll._rust_coll
    rc_bin.add_items(bin_vecs, None)
    rc_bin.commit()
    print(f"  Inserted {N_BIN_BASE:,} binary vectors.")
    del bin_vecs
else:
    print(f"Binary collection: {n_bin_existing:,} vectors already present.")

rust_bin_coll = bin_coll._rust_coll

# ─── Load SIFT queries & L2 ground truth ─────────────────────────────────────

print(f"\nLoading SIFT queries and L2 ground truth ...")
query_vecs = read_fvecs(os.path.join(SIFT_DIR, "sift_query.fvecs"))[:N_QUERIES]
gt_l2_raw  = read_ivecs(os.path.join(SIFT_DIR, "sift_groundtruth.ivecs"))[:N_QUERIES]
gt_l2      = [list(row) for row in gt_l2_raw]
print(f"  {N_QUERIES} queries loaded, L2 GT shape = {gt_l2_raw.shape}")

# ─── Synthetic binary queries ─────────────────────────────────────────────────

rng = np.random.default_rng(seed=99999)
bin_query_vecs = rng.integers(0, 2, size=(N_QUERIES, DIM)).astype(np.float32)

# ─── Pre-compute GT for IP, COS, JACCARD, HAMMING ────────────────────────────

print(f"\nPre-computing ground truth for IP / COS / JACCARD / HAMMING metrics ...")

print("  [IP]  running FLAT brute-force ...")
gt_ip = compute_brute_force_gt(collection, rust_coll, query_vecs, "FLAT", K)
print(f"        done. {len(gt_ip)} query GT entries.")

print("  [COS] running FLAT-COS brute-force ...")
gt_cos = compute_brute_force_gt(collection, rust_coll, query_vecs, "FLAT-COS", K)
print(f"        done.")

print("  [JACCARD] running FLAT-JACCARD-BINARY brute-force ...")
gt_jaccard = compute_brute_force_gt(bin_coll, rust_bin_coll, bin_query_vecs,
                                    "FLAT-JACCARD-BINARY", K)
print(f"        done.")

print("  [HAMMING] running FLAT-HAMMING-BINARY brute-force ...")
gt_hamming = compute_brute_force_gt(bin_coll, rust_bin_coll, bin_query_vecs,
                                    "FLAT-HAMMING-BINARY", K)
print(f"        done.")

GT_MAP = {
    "l2":      (query_vecs,     rust_coll,     gt_l2),
    "ip":      (query_vecs,     rust_coll,     gt_ip),
    "cos":     (query_vecs,     rust_coll,     gt_cos),
    "jaccard": (bin_query_vecs, rust_bin_coll, gt_jaccard),
    "hamming": (bin_query_vecs, rust_bin_coll, gt_hamming),
}
COLL_MAP = {
    "l2":      collection,
    "ip":      collection,
    "cos":     collection,
    "jaccard": bin_coll,
    "hamming": bin_coll,
}

# ─── Benchmark loop ───────────────────────────────────────────────────────────

results = []  # (mode, desc, group, build_s, median_ms, recall)
current_group = None

total = len(INDEX_MODES)
print(f"\nRunning {total} index modes ({N_QUERIES} queries each, k={K})\n")
print("-" * 80)

GROUP_LABELS = {
    "l2":      "L2",
    "ip":      "IP (Inner Product)",
    "cos":     "Cosine",
    "jaccard": "Jaccard (Binary, 100K vectors)",
    "hamming": "Hamming (Binary, 100K vectors)",
}

for idx, (mode, desc, group, nprobe) in enumerate(INDEX_MODES, 1):
    if group != current_group:
        current_group = group
        print(f"\n── {GROUP_LABELS[group]} {'─' * (60 - len(GROUP_LABELS[group]))}")

    queries, rc, gt = GT_MAP[group]
    coll = COLL_MAP[group]

    print(f"  [{idx:02d}/{total}] {mode:<28} [{desc}]")

    # ── Build ──
    t0 = time.perf_counter()
    try:
        coll.build_index(mode)
    except Exception as e:
        print(f"    ✗ build_index error: {e}")
        results.append((mode, desc, group, float('nan'), float('nan'), float('nan')))
        continue
    build_s = time.perf_counter() - t0
    print(f"    build: {build_s:.2f}s", end="  ")

    # ── Warmup ──
    wv = np.ascontiguousarray(queries[0], dtype=np.float32)
    for _ in range(3):
        rc.search(wv, k=K, where=None, nprobe=nprobe)

    # ── Timed search ──
    try:
        all_ids, median_ms, qps = run_search(rc, queries, K, nprobe)
    except Exception as e:
        print(f"\n    ✗ search error: {e}")
        results.append((mode, desc, group, build_s, float('nan'), float('nan')))
        continue
    print(f"search: {median_ms:.3f} ms  ({qps:,.0f} QPS)", end="  ")

    # ── Recall ──
    recall = compute_recall(all_ids, gt, K)
    mark = "✓✓" if recall >= 0.99 else ("✓" if recall >= 0.95 else
           (" " if recall >= 0.90 else "!"))
    print(f"recall@{K}: {recall:.4f} {mark}")

    results.append((mode, desc, group, build_s, median_ms, recall))

# ─── Summary tables (one per metric group) ────────────────────────────────────

W = 82
FLOAT_N = f"{N_BASE:,}"
BIN_N   = f"{N_BIN_BASE:,}"

print()
print("=" * W)
print(f"  COMPREHENSIVE BENCHMARK SUMMARY  —  k={K}  —  {N_QUERIES} queries")
print("=" * W)

for group, label in GROUP_LABELS.items():
    group_results = [(m, d, g, b, ms, r) for m, d, g, b, ms, r in results if g == group]
    if not group_results:
        continue
    n_vecs = BIN_N if group in BINARY_GROUPS else FLOAT_N
    print(f"\n  ── {label}  ({n_vecs} × {DIM} vectors) {'─' * max(0, 50 - len(label) - len(n_vecs))}")
    print(f"  {'Index Mode':<30} {'Description':<18} {'Build':>7} {'ms/q':>8} {'Recall':>8}  {'':2}")
    print(f"  {'-'*30} {'-'*18} {'-'*7} {'-'*8} {'-'*8}")
    for mode, desc, grp, bt, ms, rec in group_results:
        if any(v != v for v in [bt, ms, rec]):   # isnan check without import math
            print(f"  {mode:<30} {desc:<18} {'ERR':>7} {'ERR':>8} {'ERR':>8}  ✗")
            continue
        if rec >= 0.99:   mark = "✓✓"
        elif rec >= 0.95: mark = "✓ "
        elif rec >= 0.90: mark = "  "
        else:             mark = "! "
        print(f"  {mode:<30} {desc:<18} {bt:>7.2f} {ms:>8.3f} {rec:>8.4f}  {mark}")

print()
print("=" * W)
print("  ✓✓ recall≥0.99  ✓ recall≥0.95  (space) recall≥0.90  ! recall<0.90  ✗ error")
print()

# ─── Compression reference ────────────────────────────────────────────────────

f32bpv = DIM * 4
print("  Compression reference (per float32 vector):")
for mode, bpv, ratio in [
    ("FLAT / FLAT-L2",       f32bpv,              "1.0×"),
    ("*-SQ8",                DIM,                 "~4×"),
    ("*-PQ8",                8,                   "~64×"),
    ("*-PQ16",               16,                  "~32×"),
    ("*-RABITQ",             DIM // 8,            "~32×"),
    ("*-POLARVEC3",          (DIM * 3 + 7) // 8,  "~10.7×"),
    ("*-POLARVEC / POLARVEC4", DIM // 2,          "~8×"),
    ("*-POLARVEC8",          DIM,                 "~4×"),
]:
    mb_1m = bpv * N_BASE / 1e6
    print(f"    {mode:<28}  {bpv:>4} B/vec   {mb_1m:>7.1f} MB/1M  ({ratio})")
