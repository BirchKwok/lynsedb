"""DiskANN IP in-place insert/delete/upsert (no full rebuild)."""

from __future__ import annotations

import numpy as np

from .conftest import DIM


def test_diskann_ip_add_delete_upsert_no_rebuild(collection):
    rng = np.random.default_rng(0)
    n, dim = 120, DIM
    data = rng.standard_normal((n, dim)).astype(np.float32)
    ids = list(range(n))
    collection.add(ids=ids, vectors=data)
    collection.commit()
    collection.build_index("DISKANN-L2")

    # Incremental insert after index build must succeed (previously raised).
    extra = rng.standard_normal((10, dim)).astype(np.float32)
    extra_ids = list(range(n, n + 10))
    collection.add(ids=extra_ids, vectors=extra)
    collection.commit()

    # Soft-delete + IP graph delete.
    collection.delete([5, 6, 7])
    res = collection.search(vector=data[5], k=5)
    assert 5 not in list(res.ids)

    # Upsert existing id should re-link without full rebuild.
    new_vec = rng.standard_normal(dim).astype(np.float32)
    collection.upsert(ids=[20], vectors=new_vec)
    collection.commit()
    res2 = collection.search(vector=new_vec, k=5)
    assert 20 in list(res2.ids)
