"""Final benchmark: LynseDB optimized query performance at 1M scale.

Baselines (user-reported, 1M rows):
  - LynseDB (before): 20.2ms ± 0.237ms  (collection.query('"order"=1'))
  - Direct ApexBase:   1.05ms ± 0.011ms  (client.execute("SELECT * ... WHERE `order`=1"))
"""
import time
import numpy as np
import lynse
import shutil

N = 1_000_000
DIM = 128

def bench(collection, expr, label, n_iters=50):
    _ = collection.query(expr)  # warmup
    _ = collection.query(expr)  # warmup
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = collection.query(expr)
        times.append((time.perf_counter() - t0) * 1000)
    median = np.median(times)
    mean = np.mean(times)
    std = np.std(times)
    print(f"  {label}: {median:.3f}ms ± {std:.3f}ms  (matches={len(result.ids)})")
    return median

print(f"=== LynseDB Optimized Query Benchmark (n={N:,}) ===")
print("Optimizations applied:")
print("  1. query_with_fields() — single ApexBase query for IDs+fields")
print("  2. buffer_size=2^31 — all inserts buffered until commit (1 batch)")
print()

shutil.rmtree('test_perf/', ignore_errors=True)
client = lynse.VectorDBClient('test_perf/')
my_db = client.create_database("perf_db", drop_if_exists=True)
collection = my_db.require_collection("perf_vectors", dim=DIM, drop_if_exists=True)

t0 = time.perf_counter()
with collection.insert_session() as session:
    for i in range(N):
        vec = np.random.random(DIM).astype(np.float32)
        session.add_item(vec, id=i, field={"test": f"test_{i // 1000}", "order": i})
t_insert = time.perf_counter() - t0
print(f"Insert {N:,} vectors: {t_insert:.1f}s (single batch)\n")

# ─── Benchmarks ───────────────────────────────────────────────────────────────
m1 = bench(collection, '"order"=1', 'query("order"=1) — int, 1 match')
m2 = bench(collection, '"test"=\'test_0\'', 'query("test"=\'test_0\') — str, ~1000 matches')
m3 = bench(collection, '"order"=1', 'query("order"=1) — repeat')

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  BEFORE optimization:  20.2ms  (user-reported)")
print(f"  Direct ApexBase:       1.05ms (user-reported)")
print(f"  AFTER optimization:   {m1:.3f}ms")
print(f"{'='*60}")
speedup_vs_before = 20.2 / m1
if m1 < 1.05:
    print(f"  ✅ {speedup_vs_before:.0f}x faster than before")
    print(f"  ✅ FASTER than direct ApexBase Python API ({1.05/m1:.1f}x)")
else:
    print(f"  {speedup_vs_before:.1f}x faster than before")
    ratio = m1 / 1.05
    print(f"  {ratio:.1f}x vs direct ApexBase")

# Cleanup
shutil.rmtree('test_perf/', ignore_errors=True)
