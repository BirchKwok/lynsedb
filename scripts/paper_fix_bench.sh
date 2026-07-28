#!/usr/bin/env bash
# Paper-fidelity fix regression bench on ANN_SIFT1M / sift_learn.
#
# After aligning HNSW SQ/non-SQ search beams, measure recall/latency on real
# SIFT vectors under ``GATE_SIFT_DIR`` (default: ~/Downloads/sift).
#
# Usage:
#   scripts/paper_fix_bench.sh self              # one-shot SIFT run
#   scripts/paper_fix_bench.sh record            # save as paper-fix-pre.json
#   scripts/paper_fix_bench.sh check             # compare vs recorded pre
#   GATE_ROWS=1000000 scripts/paper_fix_bench.sh self   # full SIFT1M base
#
# Default rows=100000 uses sift_learn.fvecs when present.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-}"
if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  sed -n '2,18p' "$0"
  exit 0
fi

PYTHON="${PYTHON:-python3}"
ROWS="${GATE_ROWS:-100000}"
BATCH="${GATE_BATCH_SIZE:-50000}"
WARMUPS="${GATE_WARMUPS:-2}"
TRIALS="${GATE_TRIALS:-20}"
NPROBE="${GATE_NPROBE:-64}"
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export RAYON_NUM_THREADS

SIFT_DIR="${GATE_SIFT_DIR:sift}"
if [[ ! -d "$SIFT_DIR" ]]; then
  echo "SIFT dir not found: $SIFT_DIR (set GATE_SIFT_DIR)" >&2
  exit 2
fi

MODES="${GATE_INDEX_MODES:-}"
if [[ -z "$MODES" ]]; then
  MODES="$(
    "$PYTHON" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("benchmarks").resolve()))
from gate_index_modes import SIFT_PAPER_FIX_INDEX_MODES
print(",".join(SIFT_PAPER_FIX_INDEX_MODES))
PY
  )"
fi

BASELINE_JSON="$ROOT/benchmarks/baselines/paper-fix-pre.json"
CANDIDATE_JSON="$ROOT/benchmarks/baselines/paper-fix-post.json"
SELF_JSON="$ROOT/benchmarks/baselines/paper-fix-sift-self.json"
REPORT_JSON="$ROOT/benchmarks/baselines/paper-fix-report.json"
DATA_DIR="${GATE_MATRIX_DIR:-/tmp/lynse_sift_paper_fix}"

mkdir -p "$ROOT/benchmarks/baselines"

echo "==> Install editable LynseDB (locked maturin build)"
"$PYTHON" -m pip install -q --upgrade pip
"$PYTHON" -m pip install -q --config-settings=maturin.build-args=--locked .

run_sift() {
  local output="$1"
  local side="$2"
  echo "==> SIFT paper-fix matrix → $output"
  echo "    sift=$SIFT_DIR rows=$ROWS nprobe=$NPROBE modes=$MODES"
  "$PYTHON" "$ROOT/benchmarks/sift_paper_fix_bench.py" \
    --sift-dir "$SIFT_DIR" \
    --rows "$ROWS" \
    --batch-size "$BATCH" \
    --warmups "$WARMUPS" \
    --trials "$TRIALS" \
    --nprobe "$NPROBE" \
    --modes "$MODES" \
    --data-dir "$DATA_DIR" \
    --side "$side" \
    --git-ref "sift-local" \
    --output "$output"
}

case "$MODE" in
  self)
    run_sift "$SELF_JSON" self
    ;;
  record)
    run_sift "$SELF_JSON" baseline
    cp "$SELF_JSON" "$BASELINE_JSON"
    echo "Wrote baseline $BASELINE_JSON"
    ;;
  check)
    if [[ ! -f "$BASELINE_JSON" ]]; then
      echo "Missing baseline $BASELINE_JSON — run: $0 record" >&2
      exit 2
    fi
    # Fresh data dir so we do not reuse a dirty collection from record.
    DATA_DIR="${GATE_MATRIX_DIR:-/tmp/lynse_sift_paper_fix_post}"
    run_sift "$CANDIDATE_JSON" candidate
    "$PYTHON" - <<PY
import json
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
# Lightweight compare: reuse perf_gate_local.evaluate if profiles match enough.
from perf_gate_local import evaluate, print_cases
import argparse

class A: pass
args = A()
args.relative_budget = float("${GATE_RELATIVE_BUDGET:-0.15}")
args.search_absolute_budget_ms = float("${GATE_SEARCH_ABS_MS:-5.0}")
args.build_absolute_budget_ms = float("${GATE_BUILD_ABS_MS:-60000.0}")
args.ingest_absolute_budget_ms = 60000.0
args.upsert_absolute_budget_ms = 5.0

base = json.loads(Path("$BASELINE_JSON").read_text())
cand = json.loads(Path("$CANDIDATE_JSON").read_text())
# Soften profile match: compare core numeric fields only via evaluate's profile_core.
cases = evaluate(base, cand, args)
failed = print_cases(cases)
Path("$REPORT_JSON").write_text(json.dumps({"schema_version": 3, "mode": "sift-check", "warning": failed, "cases": cases}, indent=2, sort_keys=True) + "\n")
print("Report: $REPORT_JSON")
raise SystemExit(1 if failed else 0)
PY
    ;;
  *)
    echo "Unknown mode: $MODE (expected self|record|check)" >&2
    exit 2
    ;;
esac
