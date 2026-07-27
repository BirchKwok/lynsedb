#!/usr/bin/env bash
# Local-only canary smoke for LynseDB.
#
# Intentionally NOT wired into GitHub Actions. Run on the developer machine
# (or a dedicated local runner) after installing the current tree.
#
# Usage:
#   scripts/run_local_canary.sh
#   PYTHON=python3.11 scripts/run_local_canary.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"

echo "==> Local canary (machine-only; not for CI)"
"$PYTHON" - <<'PY'
import shutil
import tempfile
from pathlib import Path

import numpy as np

import lynse

root = Path(tempfile.mkdtemp(prefix="lynse-canary-"))
try:
    client = lynse.VectorDBClient(uri=str(root))
    db = client.create_database("canary_db", drop_if_exists=True)

    # ── FLAT path: add / filter / delete ──────────────────────────────────────
    coll = db.require_collection(
        "canary_col",
        dim=8,
        drop_if_exists=True,
        default_index="FLAT-IP",
    )

    rng = np.random.default_rng(7)
    vectors = rng.random((64, 8), dtype=np.float32)
    vectors[0] = np.ones(8, dtype=np.float32)
    ids = list(range(64))
    fields = [{"tag": "a" if i % 2 == 0 else "b", "score": i} for i in ids]
    coll.add(ids=ids, vectors=vectors, fields=fields)
    coll.commit()

    hit = coll.search(vector=vectors[0], k=3)
    assert len(hit) > 0, "search returned empty ResultView"
    assert int(hit.ids[0]) == 0, f"expected nearest id 0, got {hit.ids}"

    filtered = coll.search(vector=vectors[1], k=5, where="tag = 'b'")
    assert len(filtered) > 0, "filtered search returned empty"
    assert all(fields[int(i)]["tag"] == "b" for i in filtered.ids), filtered.ids

    batch = coll.batch_search(vectors[:4], k=3)
    assert len(batch) == 4, batch
    assert int(batch[0].ids[0]) == 0, batch[0].ids

    filtered_batch = coll.batch_search(vectors[:2], k=5, where="tag = 'a'")
    for result in filtered_batch:
        assert all(fields[int(i)]["tag"] == "a" for i in result.ids), result.ids

    coll.delete([0])
    after_delete = coll.search(vector=vectors[0], k=3)
    assert 0 not in [int(i) for i in after_delete.ids], after_delete.ids

    # ── HNSW filtered parity with single search ──────────────────────────────
    hnsw = db.require_collection(
        "canary_hnsw",
        dim=8,
        drop_if_exists=True,
        default_index="HNSW-IP",
    )
    hnsw.add(ids=ids, vectors=vectors, fields=fields)
    hnsw.commit()
    hnsw.build_index("HNSW-IP")
    single = hnsw.search(vector=vectors[0], k=5, where="tag = 'b'")
    batched = hnsw.batch_search(vectors[:1], k=5, where="tag = 'b'")
    assert list(single.ids) == list(batched[0].ids), (single.ids, batched[0].ids)
    assert all(fields[int(i)]["tag"] == "b" for i in single.ids), single.ids

    print("canary ok: add/search/filter/batch/delete + hnsw filtered parity")
finally:
    shutil.rmtree(root, ignore_errors=True)
PY
