"""Smoke test: local mode uses direct Rust backend, no HTTP."""
import tempfile
import numpy as np
from lynse import VectorDBClient

with tempfile.TemporaryDirectory() as tmpdir:
    client = VectorDBClient(tmpdir)
    print("Client:", client)
    print("Is remote:", client._is_remote)
    print("Manager type:", type(client._manager).__name__)

    db = client.create_database("test_db")
    print("DB type:", type(db).__name__)

    dbs = client.list_databases()
    print("Databases:", dbs)

    coll = db.require_collection("test_coll", dim=128, chunk_size=10000)
    print("Collection type:", type(coll).__name__)

    vectors = np.random.randn(100, 128).astype(np.float32)
    coll.bulk_add_binary(vectors, enable_progress_bar=False)
    print("Shape:", coll.shape)

    coll.commit()
    print("Committed")

    query = np.random.randn(128).astype(np.float32)
    ids, dists, fields = coll.search(query, k=5)
    print("Search ids:", ids)

    coll.build_index("Flat-IP")
    print("Index:", coll.index_mode)

    ids2, dists2, _ = coll.search(query, k=5)
    print("Index search ids:", ids2)

    queries = np.random.randn(3, 128).astype(np.float32)
    results = coll.batch_search(queries, k=5)
    print("Batch:", len(results), "results")

    client.drop_database("test_db")
    print("After drop:", client.list_databases())
    print("ALL PASSED")
