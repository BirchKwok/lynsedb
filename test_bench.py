"""Benchmark: verify SELECT column vs SELECT * in apexbase 1.14.0."""
import lynse
import numpy as np
import time

client = lynse.VectorDBClient('test/')
my_db = client.create_database("my_vec_db", drop_if_exists=True)
collection = my_db.require_collection("my_vectors", dim=128, drop_if_exists=True)

d = 128
n = 1_000_000

query = np.random.random(d).astype(np.float32)

print(f"Inserting {n} vectors...")
t0 = time.perf_counter()
with collection.insert_session() as session:
    for i in range(n):
        vec = query if i == 0 else np.random.random(d).astype(np.float32)
        session.add_item(vec, id=i, field={"test": f"test_{i // 1000}", "order": i})
print(f"Insert: {time.perf_counter()-t0:.1f}s, shape={collection.shape}")

coll = collection._rust_coll
qvec = np.ascontiguousarray(query, dtype=np.float32).ravel()

# Warmup
for _ in range(3):
    coll.query_fields('"order"=1')
    coll.query_with_fields('"order"=1')
    coll._inner.search(qvec, 10, '"order"=1', 10)
    coll._inner.search(qvec, 10, None, 10)

N = 30

def bench(fn):
    times = []
    for _ in range(N):
        s = time.perf_counter()
        fn()
        times.append((time.perf_counter() - s) * 1000)
    return np.median(times)

print()
print("=== apexbase 1.14.0: SELECT column vs SELECT * ===")
t_col = bench(lambda: coll.query_fields('"order"=1'))
t_star = bench(lambda: coll.query_with_fields('"order"=1'))
print(f"query_fields  (SELECT external_id):  {t_col:.2f}ms  [used by search(where=...)]")
print(f"query_with_fields (SELECT *):        {t_star:.2f}ms  [used by collection.query()]")
if t_col < t_star:
    print(f"=> SELECT column is now FASTER by {t_star/t_col:.1f}x — revert field_store.query() SQL")
elif t_col > t_star * 1.2:
    print(f"=> SELECT column is still SLOWER by {t_col/t_star:.1f}x — keep SELECT *")
else:
    print(f"=> Performance is roughly equal — keep SELECT *")

print()
print("=== End-to-end search timing ===")
t_filtered = bench(lambda: coll._inner.search(qvec, 10, '"order"=1', 10))
t_plain = bench(lambda: coll._inner.search(qvec, 10, None, 10))
print(f"Rust search(where='order'=1):  {t_filtered:.2f}ms")
print(f"Rust search(no filter):        {t_plain:.2f}ms")
