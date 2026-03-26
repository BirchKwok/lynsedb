"""Benchmark: insert 1M vectors + search performance."""
import lynse
import numpy as np
import time

client = lynse.VectorDBClient('test/')
my_db = client.create_database("my_vec_db", drop_if_exists=True)

collection = my_db.require_collection("my_vectors", dim=128, drop_if_exists=True,
                                      cache_chunks=0, cache_query=False, chunk_size=10_0000)

d = 128
n = 100_0000
current_rows = collection.shape[0]

query = np.random.random(d).astype(np.float32)

t0 = time.perf_counter()
with collection.insert_session() as session:
    for i in range(current_rows, current_rows + n):
        if i == 0:
            vec = query
        else:
            vec = np.random.random(d).astype(np.float32)
        session.add_item(vec, id=i, field={"test": f"test_{i // 1000}", "order": i})
t1 = time.perf_counter()
print(f"Insert {n} vectors: {t1 - t0:.1f}s")
print(f"Shape: {collection.shape}")

# Warmup
collection.search(query, k=10)

# Benchmark search
times = []
for _ in range(10):
    s = time.perf_counter()
    result = collection.search(query, k=10)
    e = time.perf_counter()
    times.append((e - s) * 1000)

print(f"Search times (ms): {[f'{t:.2f}' for t in times]}")
print(f"Median: {sorted(times)[5]:.2f}ms")
print(f"Mean: {sum(times)/len(times):.2f}ms")
