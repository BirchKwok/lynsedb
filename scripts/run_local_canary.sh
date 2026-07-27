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

    coll.delete([0])
    after_delete = coll.search(vector=vectors[0], k=3)
    assert 0 not in [int(i) for i in after_delete.ids], after_delete.ids

    print("canary ok: add/search/filter/delete")
finally:
    shutil.rmtree(root, ignore_errors=True)
PY
