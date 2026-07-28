#!/usr/bin/env bash
# Local-only quality + performance gates for LynseDB.
#
# Performance gate defaults to an isolated A/B run:
#   - baseline git ref in its own venv + data dir
#   - current worktree (including uncommitted changes) in another
#   - 1M x 128 dense index×quant matrix + sparse/hybrid/BM25 paths
#
# These gates are intentionally NOT part of GitHub Actions.
#
# Usage:
#   scripts/run_local_gates.sh
#   scripts/run_local_gates.sh --baseline-ref HEAD~1
#   scripts/run_local_gates.sh --modes FLAT-IP,HNSW-IP
#   scripts/run_local_gates.sh --skip-perf
#   scripts/run_local_gates.sh --skip-matrix
#   scripts/run_local_gates.sh --quick
#   scripts/run_local_canary.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_PERF=0
SKIP_MATRIX=0
QUICK=0
PYTHON="${PYTHON:-python3}"
BASELINE_REF="${GATE_BASELINE_REF:-}"
MODES="${GATE_INDEX_MODES:-}"
KEEP_AB=0
EXTRA_PERF_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-perf) SKIP_PERF=1; shift ;;
    --skip-matrix|--skip-extended-perf) SKIP_MATRIX=1; shift ;;
    --quick) QUICK=1; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    --baseline-ref) BASELINE_REF="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --keep-ab) KEEP_AB=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export GATE_ROWS="${GATE_ROWS:-1000000}"
export GATE_DIM="${GATE_DIM:-128}"
export GATE_BATCH_SIZE="${GATE_BATCH_SIZE:-100000}"

echo "==> Rust correctness regressions"
if [[ "$QUICK" -eq 1 ]]; then
  for test_name in \
    validate_resource_name_rejects_path_escape \
    create_database_rejects_path_traversal_names \
    filtered_search_empty_probe_does_not_leak_unfiltered_ids \
    merge_search_blocks_with_fields_keeps_true_topk \
    blacklisted_in_query_falls_back_to_sql \
    merge_search_blocks_obeys_metric_order \
    batch_search_hnsw_where_matches_single_search \
    batch_search_flat_pq_where_matches_single_search \
    diskann_ip_search_returns_max_inner_product \
    diskann_filtered_empty_graph_does_not_leak_unfiltered_ids \
    approx_hybrid_ip_adaptive_pool_keeps_true_topk
  do
    cargo test --locked "$test_name"
  done
else
  cargo test --locked
fi

need_python=0
if [[ "$QUICK" -eq 0 || "$SKIP_PERF" -eq 0 ]]; then
  need_python=1
fi

if [[ "$need_python" -eq 1 ]]; then
  echo "==> Install editable package for Python gates"
  "$PYTHON" -m pip install -q --upgrade pip
  "$PYTHON" -m pip install -q pytest
  "$PYTHON" -m pip install -q --config-settings=maturin.build-args=--locked .
fi

if [[ "$QUICK" -eq 0 ]]; then
  echo "==> Python API tests"
  "$PYTHON" -m pytest tests/standard_tests tests/test_explicit_api_parameters.py -q
fi

if [[ "$SKIP_PERF" -eq 1 ]]; then
  echo "==> Performance gate skipped"
  exit 0
fi

PERF_ARGS=(ab --python "$PYTHON")
if [[ -n "$BASELINE_REF" ]]; then
  PERF_ARGS+=(--baseline-ref "$BASELINE_REF")
fi
if [[ -n "$MODES" ]]; then
  PERF_ARGS+=(--modes "$MODES")
fi
if [[ "$SKIP_MATRIX" -eq 1 ]]; then
  PERF_ARGS+=(--skip-matrix)
fi
if [[ "$KEEP_AB" -eq 1 ]]; then
  PERF_ARGS+=(--keep-ab)
fi
PERF_ARGS+=("${EXTRA_PERF_ARGS[@]}")

echo "==> Isolated A/B performance gate (1M x 128 dense matrix + sparse/hybrid)"
echo "    This installs baseline and candidate into separate venvs/data dirs and can take hours."
"$PYTHON" "$ROOT/scripts/perf_gate_local.py" "${PERF_ARGS[@]}"

if [[ "${SKIP_CANARY:-0}" != "1" ]]; then
  echo "==> Local canary"
  "$ROOT/scripts/run_local_canary.sh"
fi

echo "All local gates passed."
