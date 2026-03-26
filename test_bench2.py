"""Benchmark: optimized FlatMmap search on 1M vectors."""
import lynse
from lynse._core import FlatIndex
import numpy as np
import time
import os

# --- Reuse existing data or create new ---
client = lynse.VectorDBClient('test/')
my_db = client.create_database("my_vec_db", drop_if_exists=True)
collection = my_db.require_collection("my_vectors", dim=128, drop_if_exists=True,
                                      cache_chunks=0, cache_query=False, chunk_size=10_0000)

d = 128
n = 100_0000
query = np.random.random(d).astype(np.float32)

print("Inserting 1M vectors...")
t0 = time.perf_counter()
with collection.insert_session() as session:
    for i in range(n):
        vec = query if i == 0 else np.random.random(d).astype(np.float32)
        session.add_item(vec, id=i, field={"test": f"test_{i // 1000}", "order": i})
t1 = time.perf_counter()
print(f"Insert: {t1-t0:.1f}s, shape={collection.shape}")

# --- FlatIndex direct benchmark ---
coll_path = os.path.join('test/', 'my_vec_db', 'my_vectors')
vectors_bin = os.path.join(coll_path, 'vectors.bin')
print(f"\nvectors.bin: {os.path.getsize(vectors_bin)/1024/1024:.1f} MB")

fi = FlatIndex(vectors_bin, d)
print(f"FlatIndex vectors: {len(fi)}")

for metric_name in ["ip", "l2", "cosine"]:
    # Warmup (3 calls)
    for _ in range(3):
        fi.search(query, k=10, metric=metric_name)

    times = []
    for _ in range(20):
        s = time.perf_counter()
        ids, dists = fi.search(query, k=10, metric=metric_name)
        e = time.perf_counter()
        times.append((e - s) * 1000)

    times.sort()
    median = times[10]
    p10 = times[2]
    p90 = times[17]
    print(f"  FlatIndex {metric_name:>6}: median={median:.2f}ms  p10={p10:.2f}ms  p90={p90:.2f}ms")

# --- Collection.search benchmark ---
print("\nCollection.search:")
for _ in range(3):
    collection.search(query, k=10)

times = []
for _ in range(20):
    s = time.perf_counter()
    result = collection.search(query, k=10)
    e = time.perf_counter()
    times.append((e - s) * 1000)

times.sort()
median = times[10]
p10 = times[2]
p90 = times[17]
print(f"  Collection IP: median={median:.2f}ms  p10={p10:.2f}ms  p90={p90:.2f}ms")
