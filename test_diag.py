"""Diagnostic: isolate where the 28ms is spent."""
import lynse
from lynse._core import FlatIndex
import numpy as np
import time
import os

# --- Step 1: Create collection and insert data ---
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

# --- Step 2: Find vectors.bin ---
coll_path = os.path.join('test/', 'my_vec_db', 'my_vectors')
vectors_bin = os.path.join(coll_path, 'vectors.bin')
print(f"\nvectors.bin exists: {os.path.exists(vectors_bin)}")
if os.path.exists(vectors_bin):
    sz = os.path.getsize(vectors_bin)
    print(f"vectors.bin size: {sz} bytes ({sz/1024/1024:.1f} MB)")
    print(f"Expected: {n * d * 4} bytes ({n * d * 4 / 1024/1024:.1f} MB)")

# --- Step 3: Test FlatIndex (FlatMmap) directly ---
print("\n--- FlatIndex (FlatMmap) direct test ---")
fi = FlatIndex(vectors_bin, d)
print(f"FlatIndex len: {len(fi)}")

# Warmup
fi.search(query, k=10, metric="ip")

times = []
for _ in range(10):
    s = time.perf_counter()
    ids, dists = fi.search(query, k=10, metric="ip")
    e = time.perf_counter()
    times.append((e - s) * 1000)
print(f"FlatIndex search (ms): {[f'{t:.2f}' for t in times]}")
print(f"FlatIndex median: {sorted(times)[5]:.2f}ms")

# --- Step 4: Test Collection.search ---
print("\n--- Collection.search test ---")
# Warmup
collection.search(query, k=10)

times2 = []
for _ in range(10):
    s = time.perf_counter()
    result = collection.search(query, k=10)
    e = time.perf_counter()
    times2.append((e - s) * 1000)
print(f"Collection search (ms): {[f'{t:.2f}' for t in times2]}")
print(f"Collection median: {sorted(times2)[5]:.2f}ms")

# --- Step 5: Check index_mode ---
print(f"\nindex_mode: {collection._rust_coll._inner.get_index_mode()}")
