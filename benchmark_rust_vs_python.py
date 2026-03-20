"""Benchmark: Rust backend vs Python backend for LynseDB core operations.

Comprehensive comparison covering:
  1. Single-pair distance computation
  2. Top-K brute force (exact) search
  3. Storage write / read / shape
  4. Flat (brute-force) index search
  5. HNSW approximate index build + search
"""
import time, numpy as np, tempfile, shutil, pathlib

import lynse_core
import simsimd
from usearch.compiled import exact_search
from usearch.index import Index, MetricKind


def bench(label, fn, repeat=5, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"  {label:48s} {avg:9.3f} ms  (±{std:.3f})")
    return avg


def timed(label, fn):
    """Run once, print elapsed time, return result."""
    t0 = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  {label:48s} {elapsed:9.3f} ms")
    return elapsed, result


W = 72
print("=" * W)
print("Benchmark: Rust (lynse_core) vs Python (simsimd/usearch)")
print("=" * W)

# ── 1. Single-pair distance ──────────────────────────────────────────
print("\n## 1. Single-pair distance (dim=768, 200 iters)")
dim = 768; np.random.seed(42)
a = np.random.randn(dim).astype(np.float32)
b = np.random.randn(dim).astype(np.float32)

bench("Rust  IP",  lambda: lynse_core.py_compute_distance(a, b, "ip"), 200)
bench("Py    simsimd.inner", lambda: float(simsimd.inner(a, b)), 200)
bench("Rust  L2",  lambda: lynse_core.py_compute_distance(a, b, "l2"), 200)
bench("Py    simsimd.sqeuclidean", lambda: float(simsimd.sqeuclidean(a, b)), 200)
bench("Rust  Cos", lambda: lynse_core.py_compute_distance(a, b, "cosine"), 200)
bench("Py    simsimd.cosine", lambda: float(simsimd.cosine(a, b)), 200)

# ── 2. Top-K brute force search ──────────────────────────────────────
for n in [10_000, 100_000]:
    dim = 128
    print(f"\n## 2. Top-10 brute force (exact): {n:,} × dim={dim}")
    np.random.seed(42)
    q = np.random.randn(dim).astype(np.float32)
    db = np.ascontiguousarray(np.random.randn(n, dim).astype(np.float32))
    q2d = q.reshape(1, -1)

    bench("Rust  top_k IP",  lambda: lynse_core.py_top_k_search(q, db, "ip", 10))
    bench("Py    usearch exact IP", lambda: exact_search(db, q2d, 10, metric_kind=MetricKind.IP))
    bench("Rust  top_k L2",  lambda: lynse_core.py_top_k_search(q, db, "l2", 10))
    bench("Py    usearch exact L2", lambda: exact_search(db, q2d, 10, metric_kind=MetricKind.L2sq))

# ── 3. Storage write + read + shape ─────────────────────────────────
print(f"\n## 3. Storage: write / read / shape — 50,000 × dim=128")
dim = 128; n = 50_000
vectors = np.random.randn(n, dim).astype(np.float32)

# Rust
tmpdir_r = tempfile.mkdtemp()
eng = lynse_core.DatabaseEngine(tmpdir_r)
coll = eng.create_collection("bench", dim)
bench("Rust  write 50k",  lambda: coll.add_items(vectors), repeat=3)
bench("Rust  shape",      lambda: coll.shape(), repeat=10)
shutil.rmtree(tmpdir_r)

# Python (numpy as storage baseline)
tmpdir_p = tempfile.mkdtemp()
pfile = pathlib.Path(tmpdir_p) / "chunk_0.npy"
bench("Py    np.save 50k", lambda: np.save(pfile, vectors), repeat=3)
bench("Py    np.load 50k", lambda: np.load(pfile), repeat=10)
# np.load to get shape is trivial — read header only
bench("Py    shape (npy header)", lambda: np.load(pfile, mmap_mode='r').shape, repeat=10)
shutil.rmtree(tmpdir_p)

# ── 4. Flat (brute-force) index search (50k) ─────────────────────────
print(f"\n## 4. Flat (brute-force) search: 50k × dim=128, top-10")

# Rust: Flat-L2 via lynse_core
tmpdir_r = tempfile.mkdtemp()
eng = lynse_core.DatabaseEngine(tmpdir_r)
coll = eng.create_collection("flat_bench", dim)
coll.add_items(vectors)
timed("Rust  Flat-L2 build", lambda: coll.build_index("Flat-L2"))
bench("Rust  Flat-L2 search",  lambda: coll.search(vectors[0], k=10), repeat=20)
shutil.rmtree(tmpdir_r)

# Python: usearch exact_search (equivalent to Flat brute-force)
q2d = vectors[0].reshape(1, -1)
bench("Py    usearch exact L2 search", lambda: exact_search(vectors, q2d, 10, metric_kind=MetricKind.L2sq), repeat=20)

# ── 5. HNSW index build + search (5k) ────────────────────────────────
print(f"\n## 5. HNSW index: 5k × dim=128, top-10")
small = vectors[:5000]

# Rust: HNSW-L2
tmpdir_h = tempfile.mkdtemp()
eng2 = lynse_core.DatabaseEngine(tmpdir_h)
coll2 = eng2.create_collection("hnsw_bench", dim)
coll2.add_items(small)
timed("Rust  HNSW-L2 build (5k)", lambda: coll2.build_index("HNSW-L2"))
bench("Rust  HNSW-L2 search (5k)",  lambda: coll2.search(small[0], k=10), repeat=20)
shutil.rmtree(tmpdir_h)

# Python: usearch HNSW Index (multi-threaded, default)
py_idx = Index(ndim=dim, metric=MetricKind.L2sq, dtype="f32")
keys = np.arange(len(small), dtype=np.uint64)
timed("Py    usearch HNSW build (5k, multi-thread)", lambda: py_idx.add(keys, small))
bench("Py    usearch HNSW search (5k)", lambda: py_idx.search(small[0], 10), repeat=20)
# Python: usearch single-threaded
py_idx_st = Index(ndim=dim, metric=MetricKind.L2sq, dtype="f32")
timed("Py    usearch HNSW build (5k, 1 thread)", lambda: py_idx_st.add(keys, small, threads=1))

# ── 6. HNSW index build + search (50k) ───────────────────────────────
print(f"\n## 6. HNSW index: 50k × dim=128, top-10")

# Rust: HNSW-L2
tmpdir_h2 = tempfile.mkdtemp()
eng3 = lynse_core.DatabaseEngine(tmpdir_h2)
coll3 = eng3.create_collection("hnsw_bench_50k", dim)
coll3.add_items(vectors)
timed("Rust  HNSW-L2 build (50k)", lambda: coll3.build_index("HNSW-L2"))
bench("Rust  HNSW-L2 search (50k)", lambda: coll3.search(vectors[0], k=10), repeat=20)
shutil.rmtree(tmpdir_h2)

# Python: usearch HNSW Index (multi-threaded, default)
py_idx2 = Index(ndim=dim, metric=MetricKind.L2sq, dtype="f32")
keys2 = np.arange(len(vectors), dtype=np.uint64)
timed("Py    usearch HNSW build (50k, multi-thread)", lambda: py_idx2.add(keys2, vectors))
bench("Py    usearch HNSW search (50k)", lambda: py_idx2.search(vectors[0], 10), repeat=20)
# Python: usearch single-threaded
py_idx2_st = Index(ndim=dim, metric=MetricKind.L2sq, dtype="f32")
timed("Py    usearch HNSW build (50k, 1 thread)", lambda: py_idx2_st.add(keys2, vectors, threads=1))


print("\n" + "=" * W)
print("Benchmark complete.")
print("=" * W)
