#!/usr/bin/env bash
# Local-only quality + performance gates for LynseDB.
#
# These gates are intentionally NOT part of GitHub Actions. Baselines are
# machine-specific; run on the developer workstation (or a fixed local box).
#
# Usage:
#   scripts/run_local_gates.sh              # correctness + perf check (or record if no baseline)
#   scripts/run_local_gates.sh --record-perf
#   scripts/run_local_gates.sh --skip-perf
#   scripts/run_local_gates.sh --quick      # focused Rust regression tests only + skip heavy python
#   scripts/run_local_canary.sh             # separate local canary smoke

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD_PERF=0
SKIP_PERF=0
QUICK=0
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --record-perf) RECORD_PERF=1; shift ;;
    --skip-perf) SKIP_PERF=1; shift ;;
    --quick) QUICK=1; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
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

echo "==> Rust correctness regressions"
if [[ "$QUICK" -eq 1 ]]; then
  for test_name in \
    validate_resource_name_rejects_path_escape \
    create_database_rejects_path_traversal_names \
    filtered_search_empty_probe_does_not_leak_unfiltered_ids \
    merge_search_blocks_with_fields_keeps_true_topk \
    blacklisted_in_query_falls_back_to_sql \
    merge_search_blocks_obeys_metric_order
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

BASELINE="$ROOT/benchmarks/baselines/local-perf-baseline.json"
if [[ "$RECORD_PERF" -eq 1 || ! -f "$BASELINE" ]]; then
  echo "==> Recording local performance baseline -> $BASELINE"
  "$PYTHON" "$ROOT/scripts/perf_gate_local.py" record --python "$PYTHON"
  if [[ ! -f "$BASELINE" ]]; then
    echo "Failed to write baseline" >&2
    exit 1
  fi
  if [[ "$RECORD_PERF" -eq 1 ]]; then
    echo "Baseline recorded. Re-run without --record-perf to check."
    exit 0
  fi
  echo "No prior baseline existed; recorded one for this machine. Next run will compare."
  exit 0
fi

echo "==> Local performance gate (check vs $BASELINE)"
"$PYTHON" "$ROOT/scripts/perf_gate_local.py" check --python "$PYTHON"

if [[ "${SKIP_CANARY:-0}" != "1" ]]; then
  echo "==> Local canary"
  "$ROOT/scripts/run_local_canary.sh"
fi

echo "All local gates passed."
