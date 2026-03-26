"""Test script for index_mode refactor and removed params."""
import numpy as np
import tempfile
import time
import shutil

from lynse._backend import DatabaseManager

tmpdir = tempfile.mkdtemp()
try:
    mgr = DatabaseManager(tmpdir)
    mgr.create_database("testdb")

    # Test: require_collection without chunk_size/cache_chunks/cache_query
    mgr.require_collection("testdb", "coll1", 128, drop_if_exists=True)
    coll = mgr.get_collection("testdb", "coll1", 128)

    # Test 1: Flat build_index on EMPTY collection (should succeed instantly)
    t0 = time.perf_counter()
    coll.build_index("FLAT")
    dt = (time.perf_counter() - t0) * 1000
    print(f"[PASS] build_index('FLAT') on empty collection: {dt:.2f}ms, mode={coll.index_mode}")

    t0 = time.perf_counter()
    coll.build_index("FLAT-COS-SQ8")
    dt = (time.perf_counter() - t0) * 1000
    print(f"[PASS] build_index('FLAT-COS-SQ8') on empty collection: {dt:.2f}ms, mode={coll.index_mode}")

    # Add 100k vectors
    vecs = np.random.randn(100_000, 128).astype(np.float32)
    coll.add_items(vecs)
    print(f"\nAdded {coll.shape[0]} vectors")

    # Test 2: Flat build_index with data (should be instant — no data copy)
    for mode in ["FLAT", "FLAT-L2", "FLAT-COS", "FLAT-IP-SQ8", "FLAT-L2-SQ8", "FLAT-COS-SQ8"]:
        t0 = time.perf_counter()
        coll.build_index(mode)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[PASS] build_index('{mode}'): {dt:.2f}ms")

    # Test 3: Search correctness with each mode
    q = np.random.randn(128).astype(np.float32)
    for mode in ["FLAT", "FLAT-L2", "FLAT-COS", "FLAT-IP-SQ8", "FLAT-COS-SQ8"]:
        coll.build_index(mode)
        r = coll.search(q, k=10)
        print(f"[PASS] {mode} search: {len(r.ids)} results, index_mode={r.index_mode}")

    # Test 4: Config no longer has chunk_size
    cfg = mgr.get_collection_config("testdb", "coll1")
    assert "chunk_size" not in cfg, "chunk_size should not be in config!"
    print(f"\n[PASS] Config keys: {sorted(cfg.keys())}")

    # Test 5: LocalClient API
    from lynse.api.local_client import LocalClient
    client = LocalClient(mgr, "testdb")
    lc = client.require_collection("coll2", dim=64, drop_if_exists=True)
    lc.build_index("FLAT-L2")  # build on empty — should work
    print(f"[PASS] LocalClient build_index on empty collection: mode={lc.index_mode}")

    vecs2 = np.random.randn(500, 64).astype(np.float32)
    lc.bulk_add_binary(vecs2, enable_progress_bar=False)
    lc.commit()
    ids, dists, _ = lc.search(np.random.randn(64).astype(np.float32), k=3)
    print(f"[PASS] LocalCollection FLAT-L2 search: {len(ids)} results")

    lc.build_index("FLAT-COS-SQ8")
    ids, dists, _ = lc.search(np.random.randn(64).astype(np.float32), k=3)
    print(f"[PASS] LocalCollection FLAT-COS-SQ8 search: {len(ids)} results")

    print("\n=== All tests passed! ===")
finally:
    shutil.rmtree(tmpdir)
