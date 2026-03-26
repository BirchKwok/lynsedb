"""Verify SQ8 two-pass recall vs brute-force f32."""
from lynse._core import FlatIndex
import numpy as np
import time

d = 128
n = 100_000  # 100k for fast test

# Generate data
np.random.seed(42)
data = np.random.random((n, d)).astype(np.float32)
query = np.random.random(d).astype(np.float32)

# Write to temp file
import tempfile, os
tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "vectors.bin")
data.tofile(path)

fi = FlatIndex(path, d)
print(f"Vectors: {len(fi)}")

# Ground truth: numpy brute force
for metric in ["ip", "l2", "cosine"]:
    # FlatIndex search (uses SQ8 for n >= 50k)
    ids, dists = fi.search(query, k=10, metric=metric)
    
    # Numpy ground truth
    if metric == "ip":
        all_dists = data @ query
        gt_ids = np.argsort(-all_dists)[:10]
    elif metric == "l2":
        all_dists = np.sum((data - query[None, :]) ** 2, axis=1)
        gt_ids = np.argsort(all_dists)[:10]
    elif metric == "cosine":
        norms = np.linalg.norm(data, axis=1) * np.linalg.norm(query)
        all_dists = 1.0 - (data @ query) / np.maximum(norms, 1e-30)
        gt_ids = np.argsort(all_dists)[:10]
    
    gt_set = set(gt_ids.tolist())
    result_set = set(ids.tolist())
    recall = len(gt_set & result_set) / len(gt_set)
    print(f"  {metric:>6}: recall@10 = {recall:.1%}  ids={ids.tolist()[:5]}...  gt={gt_ids.tolist()[:5]}...")

# Also test 1M scale recall
print("\n--- 1M scale recall test ---")
n2 = 1_000_000
data2 = np.random.random((n2, d)).astype(np.float32)
path2 = os.path.join(tmpdir, "vectors2.bin")
data2.tofile(path2)
fi2 = FlatIndex(path2, d)

# Warmup
fi2.search(query, k=10, metric="ip")

for metric in ["ip", "l2"]:
    ids, dists = fi2.search(query, k=10, metric=metric)
    
    if metric == "ip":
        all_dists = data2 @ query
        gt_ids = np.argsort(-all_dists)[:10]
    else:
        all_dists = np.sum((data2 - query[None, :]) ** 2, axis=1)
        gt_ids = np.argsort(all_dists)[:10]
    
    gt_set = set(gt_ids.tolist())
    result_set = set(ids.tolist())
    recall = len(gt_set & result_set) / len(gt_set)
    
    # Time it
    times = []
    for _ in range(10):
        s = time.perf_counter()
        fi2.search(query, k=10, metric=metric)
        e = time.perf_counter()
        times.append((e - s) * 1000)
    times.sort()
    print(f"  {metric:>6}: recall@10 = {recall:.1%}  median={times[5]:.2f}ms  ids={ids.tolist()[:5]}  gt={gt_ids.tolist()[:5]}")

# Cleanup
import shutil
shutil.rmtree(tmpdir)
