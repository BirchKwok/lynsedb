#!/usr/bin/env python3
"""Local-only performance gate for LynseDB.

Not for GitHub Actions. Baselines are machine-specific and live under
``benchmarks/baselines/local-perf-*.json`` (gitignored).

Modes:
  record   Run benches and write a machine-local baseline JSON.
  check    Run benches and compare against the saved baseline (fail on budget breach).
  self     Install is assumed current; just print absolute timings (no compare).

Default sizes are sized for a laptop. Override with env vars or CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "benchmarks" / "baselines" / "local-perf-baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("record", "check", "self"),
        help="record baseline, check against baseline, or print timings only",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--rows", type=int, default=int(os.environ.get("GATE_ROWS", "20000")))
    parser.add_argument("--dim", type=int, default=int(os.environ.get("GATE_DIM", "64")))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("GATE_BATCH_SIZE", "5000")),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--upsert-trials", type=int, default=3)
    parser.add_argument(
        "--relative-budget",
        type=float,
        default=float(os.environ.get("GATE_RELATIVE_BUDGET", "0.10")),
        help="fail when relative regression exceeds this AND absolute budget",
    )
    parser.add_argument(
        "--search-absolute-budget-ms",
        type=float,
        default=float(os.environ.get("GATE_SEARCH_ABS_MS", "0.20")),
    )
    parser.add_argument(
        "--ingest-absolute-budget-s",
        type=float,
        default=float(os.environ.get("GATE_INGEST_ABS_S", "0.05")),
    )
    parser.add_argument(
        "--upsert-absolute-budget-ms",
        type=float,
        default=float(os.environ.get("GATE_UPSERT_ABS_MS", "0.50")),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter that already has LynseDB installed",
    )
    return parser.parse_args()


def run_json(cmd: list[str]) -> dict:
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if completed.stdout.strip():
        # Benchmarks already print JSON; prefer the --output file path content.
        pass
    return completed


def bench_flat(args: argparse.Namespace, output: Path) -> dict:
    cmd = [
        args.python,
        str(ROOT / "benchmarks" / "flat_search_bench.py"),
        "--rows",
        str(args.rows),
        "--dim",
        str(args.dim),
        "--batch-size",
        str(args.batch_size),
        "--warmups",
        str(args.warmups),
        "--trials",
        str(args.trials),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(output.read_text())


def bench_upsert(args: argparse.Namespace, output: Path) -> dict:
    cmd = [
        args.python,
        str(ROOT / "benchmarks" / "upsert_bench.py"),
        "--rows",
        str(args.rows),
        "--dim",
        str(args.dim),
        "--ingest-batch-size",
        str(args.batch_size),
        "--trials",
        str(args.upsert_trials),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(output.read_text())


def collect(args: argparse.Namespace) -> dict:
    with tempfile.TemporaryDirectory(prefix="lynse-local-gate-") as tmp:
        tmp_path = Path(tmp)
        flat = bench_flat(args, tmp_path / "flat.json")
        upsert = bench_upsert(args, tmp_path / "upsert.json")
    return {
        "schema_version": 1,
        "profile": {
            "rows": args.rows,
            "dim": args.dim,
            "batch_size": args.batch_size,
            "warmups": args.warmups,
            "trials": args.trials,
            "upsert_trials": args.upsert_trials,
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
        },
        "flat": {
            "median_ms": flat["median_ms"],
            "ingest_seconds": flat["ingest_seconds"],
        },
        "upsert": {
            size: {"median_ms": payload["median_ms"]}
            for size, payload in upsert.get("updates", {}).items()
        },
    }


def evaluate(baseline: dict, candidate: dict, args: argparse.Namespace) -> list[dict]:
    cases = [
        {
            "name": "Flat search median",
            "unit": "ms",
            "baseline": baseline["flat"]["median_ms"],
            "candidate": candidate["flat"]["median_ms"],
            "relative_budget": args.relative_budget,
            "absolute_budget": args.search_absolute_budget_ms,
        },
        {
            "name": "Flat ingest",
            "unit": "s",
            "baseline": baseline["flat"]["ingest_seconds"],
            "candidate": candidate["flat"]["ingest_seconds"],
            "relative_budget": args.relative_budget,
            "absolute_budget": args.ingest_absolute_budget_s,
        },
    ]
    for size in sorted(baseline.get("upsert", {}), key=lambda item: int(item)):
        if size not in candidate.get("upsert", {}):
            continue
        cases.append(
            {
                "name": f"Upsert {size} row(s)",
                "unit": "ms",
                "baseline": baseline["upsert"][size]["median_ms"],
                "candidate": candidate["upsert"][size]["median_ms"],
                "relative_budget": max(args.relative_budget, 0.10),
                "absolute_budget": args.upsert_absolute_budget_ms,
            }
        )

    for case in cases:
        absolute_change = case["candidate"] - case["baseline"]
        relative_change = case["candidate"] / case["baseline"] - 1.0 if case["baseline"] else 0.0
        case["absolute_change"] = absolute_change
        case["relative_change"] = relative_change
        case["warning"] = (
            relative_change > case["relative_budget"]
            and absolute_change > case["absolute_budget"]
        )
    return cases


def main() -> int:
    args = parse_args()
    os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("RAYON_NUM_THREADS", "4"))

    candidate = collect(args)
    print(json.dumps(candidate, indent=2, sort_keys=True))

    if args.mode == "self":
        return 0

    if args.mode == "record":
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
        print(f"Wrote baseline: {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(
            f"Baseline missing: {args.baseline}\n"
            "Run: python scripts/perf_gate_local.py record",
            file=sys.stderr,
        )
        return 2

    baseline = json.loads(args.baseline.read_text())
    if baseline.get("profile") != candidate.get("profile"):
        print(
            "WARNING: baseline profile differs from current run; "
            "compare may be noisy.\n"
            f"  baseline={baseline.get('profile')}\n"
            f"  candidate={candidate.get('profile')}",
            file=sys.stderr,
        )

    cases = evaluate(baseline, candidate, args)
    failed = False
    print("\n=== Local performance gate ===")
    for case in cases:
        mark = "FAIL" if case["warning"] else "ok"
        print(
            f"[{mark}] {case['name']}: {case['relative_change']:+.2%}, "
            f"{case['absolute_change']:+.3f} {case['unit']} "
            f"(budgets {case['relative_budget']:.0%} / "
            f"{case['absolute_budget']:.3f} {case['unit']})"
        )
        failed = failed or case["warning"]

    report = {
        "schema_version": 1,
        "warning": failed,
        "baseline": str(args.baseline),
        "cases": cases,
        "candidate": candidate,
    }
    report_path = ROOT / "benchmarks" / "baselines" / "local-perf-last-check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote report: {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
