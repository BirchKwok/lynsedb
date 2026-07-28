#!/usr/bin/env python3
"""Local-only performance gate for LynseDB with isolated A/B environments.

Default mode ``ab`` installs a baseline git ref and the current working tree into
separate virtualenvs, runs the same 1M×128 matrix bench against each install
with isolated data directories, then compares latency/quality budgets.

Not for GitHub Actions. Artifacts land under ``benchmarks/baselines/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "benchmarks" / "baselines" / "local-perf-last-check.json"
DEFAULT_AB_ROOT = Path(os.environ.get("GATE_AB_ROOT", "/tmp/lynse_perf_ab"))
MATRIX_BENCH = ROOT / "benchmarks" / "gate_matrix_bench.py"

sys.path.insert(0, str(ROOT / "benchmarks"))
from gate_index_modes import recall_floor_for_mode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("ab", "record", "check", "self"),
        help="ab=isolated baseline vs candidate (default path); "
        "record/check/self keep single-env debugging",
    )
    parser.add_argument(
        "--baseline-ref",
        default=os.environ.get("GATE_BASELINE_REF", ""),
        help="Git ref for baseline side (default: merge-base with origin/main or HEAD~1)",
    )
    parser.add_argument("--ab-root", type=Path, default=DEFAULT_AB_ROOT)
    parser.add_argument("--keep-ab", action="store_true", help="keep isolated env trees")
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=ROOT / "benchmarks" / "baselines" / "local-perf-baseline-side.json",
    )
    parser.add_argument(
        "--candidate-result",
        type=Path,
        default=ROOT / "benchmarks" / "baselines" / "local-perf-candidate-side.json",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--rows", type=int, default=int(os.environ.get("GATE_ROWS", "1000000")))
    parser.add_argument("--dim", type=int, default=int(os.environ.get("GATE_DIM", "128")))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("GATE_BATCH_SIZE", "100000")),
    )
    parser.add_argument("--warmups", type=int, default=int(os.environ.get("GATE_WARMUPS", "2")))
    parser.add_argument("--trials", type=int, default=int(os.environ.get("GATE_TRIALS", "5")))
    parser.add_argument("--nprobe", type=int, default=int(os.environ.get("GATE_NPROBE", "32")))
    parser.add_argument(
        "--modes",
        default=os.environ.get("GATE_INDEX_MODES", ""),
        help="Optional comma-separated index mode subset",
    )
    parser.add_argument("--skip-matrix", action="store_true", help="only upsert + sparse shared")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--skip-shared", action="store_true")
    parser.add_argument("--skip-upsert", action="store_true")
    parser.add_argument(
        "--relative-budget",
        type=float,
        default=float(os.environ.get("GATE_RELATIVE_BUDGET", "0.15")),
    )
    parser.add_argument(
        "--search-absolute-budget-ms",
        type=float,
        default=float(os.environ.get("GATE_SEARCH_ABS_MS", "5.0")),
    )
    parser.add_argument(
        "--build-absolute-budget-ms",
        type=float,
        default=float(os.environ.get("GATE_BUILD_ABS_MS", "30000.0")),
    )
    parser.add_argument(
        "--ingest-absolute-budget-ms",
        type=float,
        default=float(os.environ.get("GATE_INGEST_ABS_MS", "60000.0")),
    )
    parser.add_argument(
        "--upsert-absolute-budget-ms",
        type=float,
        default=float(os.environ.get("GATE_UPSERT_ABS_MS", "5.0")),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Host interpreter used to create isolated venvs",
    )
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def resolve_baseline_ref(explicit: str) -> str:
    if explicit:
        return explicit
    try:
        out = subprocess.check_output(
            ["git", "merge-base", "HEAD", "origin/main"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return out
    except subprocess.CalledProcessError:
        pass
    return subprocess.check_output(["git", "rev-parse", "HEAD~1"], cwd=ROOT, text=True).strip()


def current_dirty_export(dest: Path) -> str:
    """Export HEAD plus uncommitted tracked/untracked project files into dest."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT)
    subprocess.run(["tar", "-xf", "-"], input=archive, cwd=dest, check=True)

    # Tracked modifications.
    diff = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if diff.strip():
        subprocess.run(["git", "apply", "--whitespace=nowarn"], input=diff, cwd=dest, check=True)

    # Untracked / unignored files relevant to the build and gate.
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for rel in untracked:
        if not rel:
            continue
        src = ROOT / rel
        if not src.is_file():
            continue
        if not (
            rel.startswith("src/")
            or rel.startswith("python/")
            or rel.startswith("benchmarks/")
            or rel.startswith("scripts/")
            or rel in {"Cargo.toml", "Cargo.lock", "pyproject.toml", "README.md"}
        ):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return f"WORKDIR+{head[:12]}"


def checkout_ref(dest: Path, ref: str) -> str:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    resolved = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=tar", resolved], cwd=ROOT)
    subprocess.run(["tar", "-xf", "-"], input=archive, cwd=dest, check=True)
    return resolved


def create_venv(venv_dir: Path, host_python: str) -> Path:
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    # Use host interpreter to create the venv for matching platform tags.
    run([host_python, "-m", "venv", str(venv_dir)])
    return venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"


def install_lynse(python: Path, src: Path) -> None:
    run([str(python), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-q",
            "--config-settings=maturin.build-args=--locked",
            ".",
        ],
        cwd=src,
    )


def isolated_env(side_root: Path, data_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    # Drop ambient Python path contamination.
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env["GATE_MATRIX_DIR"] = str(data_dir)
    env["GATE_MATRIX_REUSE"] = "0"
    env.setdefault("RAYON_NUM_THREADS", os.environ.get("RAYON_NUM_THREADS", "4"))
    env["PATH"] = str(side_root / "venv" / ("Scripts" if os.name == "nt" else "bin")) + os.pathsep + env.get(
        "PATH", ""
    )
    return env


def matrix_cmd(args: argparse.Namespace, output: Path, *, side: str, git_ref: str) -> list[str]:
    cmd = [
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
        "--nprobe",
        str(args.nprobe),
        "--side",
        side,
        "--git-ref",
        git_ref,
        "--output",
        str(output),
        "--no-reuse",
    ]
    if args.modes:
        cmd.extend(["--modes", args.modes])
    if args.skip_matrix:
        cmd.append("--skip-modes")
    if args.skip_sparse:
        cmd.append("--skip-sparse")
    if args.skip_shared:
        cmd.append("--skip-shared")
    if args.skip_upsert:
        cmd.append("--skip-upsert")
    return cmd


def run_matrix_side(
    args: argparse.Namespace,
    *,
    side: str,
    src: Path,
    git_ref: str,
    output: Path,
    host_python: str,
) -> dict:
    side_root = args.ab_root / args.run_id / side
    data_dir = side_root / "data"
    python = create_venv(side_root / "venv", host_python)
    print(f"==> Installing LynseDB into {side} env from {src} ({git_ref})", flush=True)
    install_lynse(python, src)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    env = isolated_env(side_root, data_dir)
    env["GATE_SIDE"] = side
    env["GATE_GIT_REF"] = git_ref
    cmd = [str(python), str(MATRIX_BENCH), *matrix_cmd(args, output, side=side, git_ref=git_ref)]
    print(f"==> Running matrix bench for {side}", flush=True)
    run(cmd, env=env)
    return json.loads(output.read_text())

def profile_core(profile: dict) -> dict:
    keys = (
        "rows",
        "dim",
        "batch_size",
        "k",
        "warmups",
        "trials",
        "batch_queries",
        "nprobe",
        "n_clusters",
        "seed",
        "sparse_nnz",
        "sparse_dims",
        "rayon_threads",
    )
    return {k: profile.get(k) for k in keys}


def add_latency_case(
    cases: list[dict],
    *,
    name: str,
    baseline: float,
    candidate: float,
    unit: str,
    relative_budget: float,
    absolute_budget: float,
) -> None:
    absolute_change = candidate - baseline
    relative_change = candidate / baseline - 1.0 if baseline else 0.0
    cases.append(
        {
            "name": name,
            "unit": unit,
            "baseline": baseline,
            "candidate": candidate,
            "relative_budget": relative_budget,
            "absolute_budget": absolute_budget,
            "absolute_change": absolute_change,
            "relative_change": relative_change,
            "warning": relative_change > relative_budget and absolute_change > absolute_budget,
        }
    )


def evaluate(baseline: dict, candidate: dict, args: argparse.Namespace) -> list[dict]:
    cases: list[dict] = []
    if profile_core(baseline.get("profile", {})) != profile_core(candidate.get("profile", {})):
        cases.append(
            {
                "name": "Profile match",
                "unit": "profile",
                "baseline": profile_core(baseline.get("profile", {})),
                "candidate": profile_core(candidate.get("profile", {})),
                "relative_budget": 0.0,
                "absolute_budget": 0.0,
                "absolute_change": 0.0,
                "relative_change": 0.0,
                "warning": True,
            }
        )
        return cases

    for size, base_row in sorted((baseline.get("upsert") or {}).items(), key=lambda kv: int(kv[0])):
        cand_row = (candidate.get("upsert") or {}).get(size)
        if not cand_row:
            continue
        add_latency_case(
            cases,
            name=f"Upsert {size}",
            baseline=base_row["median_ms"],
            candidate=cand_row["median_ms"],
            unit="ms",
            relative_budget=max(args.relative_budget, 0.10),
            absolute_budget=args.upsert_absolute_budget_ms,
        )

    sparse_b = baseline.get("sparse") or {}
    sparse_c = candidate.get("sparse") or {}
    if sparse_c.get("status") == "error":
        cases.append(
            {
                "name": "Sparse status",
                "unit": "status",
                "baseline": sparse_b.get("status", "ok"),
                "candidate": sparse_c.get("error", "error"),
                "relative_budget": 0.0,
                "absolute_budget": 0.0,
                "absolute_change": 0.0,
                "relative_change": 0.0,
                "warning": True,
            }
        )
    else:
        if sparse_c.get("filter_bad_ids", 0) > 0:
            cases.append(
                {
                    "name": "Sparse filter correctness",
                    "unit": "ids",
                    "baseline": 0,
                    "candidate": sparse_c.get("filter_bad_ids", 0),
                    "relative_budget": 0.0,
                    "absolute_budget": 0.0,
                    "absolute_change": sparse_c.get("filter_bad_ids", 0),
                    "relative_change": 0.0,
                    "warning": True,
                }
            )
        for key, abs_budget in (
            ("ingest_ms", args.ingest_absolute_budget_ms),
            ("search_p50_ms", args.search_absolute_budget_ms),
            ("filter_p50_ms", args.search_absolute_budget_ms),
        ):
            if key in sparse_b and key in sparse_c:
                add_latency_case(
                    cases,
                    name=f"Sparse {key}",
                    baseline=sparse_b[key],
                    candidate=sparse_c[key],
                    unit="ms",
                    relative_budget=args.relative_budget,
                    absolute_budget=abs_budget,
                )

    shared_keys = [
        "query_exact_p50_ms",
        "query_1pct_p50_ms",
        "query_10pct_p50_ms",
        "search_p50_ms",
        "search_filter_p50_ms",
        "approx_p50_ms",
        "bm25_p50_ms",
        "bm25_filter_p50_ms",
        "hybrid_p50_ms",
        "hybrid_filter_p50_ms",
        "batch_p50_ms",
        "batch_filter_p50_ms",
        "named_dense_p50_ms",
        "commit_median_ms",
        "tombstone_search_median_ms",
    ]
    shared_b = baseline.get("shared") or {}
    shared_c = candidate.get("shared") or {}
    for key in shared_keys:
        if key in shared_b and key in shared_c:
            add_latency_case(
                cases,
                name=f"Shared {key}",
                baseline=shared_b[key],
                candidate=shared_c[key],
                unit="ms",
                relative_budget=max(args.relative_budget, 0.15),
                absolute_budget=args.search_absolute_budget_ms,
            )

    modes_b = baseline.get("modes") or {}
    modes_c = candidate.get("modes") or {}
    for mode in sorted(set(modes_b) | set(modes_c)):
        if mode == "__NONE__":
            continue
        row_b = modes_b.get(mode)
        row_c = modes_c.get(mode)
        if row_c is None:
            cases.append(
                {
                    "name": f"{mode} missing in candidate",
                    "unit": "status",
                    "baseline": "present",
                    "candidate": "missing",
                    "relative_budget": 0.0,
                    "absolute_budget": 0.0,
                    "absolute_change": 0.0,
                    "relative_change": 0.0,
                    "warning": True,
                }
            )
            continue
        if row_c.get("status") == "error":
            cases.append(
                {
                    "name": f"{mode} status",
                    "unit": "status",
                    "baseline": (row_b or {}).get("status", "ok"),
                    "candidate": row_c.get("error", "error"),
                    "relative_budget": 0.0,
                    "absolute_budget": 0.0,
                    "absolute_change": 0.0,
                    "relative_change": 0.0,
                    "warning": True,
                }
            )
            continue
        if row_c.get("filter_bad_ids", 0) > 0:
            cases.append(
                {
                    "name": f"{mode} filter correctness",
                    "unit": "ids",
                    "baseline": 0,
                    "candidate": row_c["filter_bad_ids"],
                    "relative_budget": 0.0,
                    "absolute_budget": 0.0,
                    "absolute_change": row_c["filter_bad_ids"],
                    "relative_change": 0.0,
                    "warning": True,
                }
            )
        floor = recall_floor_for_mode(mode)
        recall = float(row_c.get("recall_at_k", 0.0))
        cases.append(
            {
                "name": f"{mode} recall@k floor",
                "unit": "recall",
                "baseline": floor,
                "candidate": recall,
                "relative_budget": 0.0,
                "absolute_budget": 0.0,
                "absolute_change": recall - floor,
                "relative_change": recall - floor,
                "warning": recall < floor,
            }
        )
        if row_b and row_b.get("status") == "ok":
            for key, abs_budget in (
                ("build_ms", args.build_absolute_budget_ms),
                ("search_p50_ms", args.search_absolute_budget_ms),
                ("filter_p50_ms", args.search_absolute_budget_ms),
                ("batch_p50_ms", args.search_absolute_budget_ms),
                ("batch_filter_p50_ms", args.search_absolute_budget_ms),
            ):
                if key in row_b and key in row_c:
                    add_latency_case(
                        cases,
                        name=f"{mode} {key}",
                        baseline=row_b[key],
                        candidate=row_c[key],
                        unit="ms",
                        relative_budget=args.relative_budget
                        if key != "build_ms"
                        else max(args.relative_budget, 0.20),
                        absolute_budget=abs_budget,
                    )
    return cases


def print_cases(cases: list[dict]) -> bool:
    failed = False
    print("\n=== Local performance gate (isolated A/B) ===")
    for case in cases:
        mark = "FAIL" if case["warning"] else "ok"
        failed = failed or case["warning"]
        if case["unit"] == "recall":
            print(
                f"[{mark}] {case['name']}: {case['candidate']:.3f} "
                f"(floor {case['baseline']:.3f})"
            )
        elif case["unit"] in {"status", "profile", "ids"}:
            print(f"[{mark}] {case['name']}: {case['candidate']!r} (baseline {case['baseline']!r})")
        else:
            print(
                f"[{mark}] {case['name']}: {case['relative_change']:+.2%}, "
                f"{case['absolute_change']:+.3f} {case['unit']} "
                f"(budgets {case['relative_budget']:.0%} / "
                f"{case['absolute_budget']:.3f} {case['unit']})"
            )
    return failed


def run_ab(args: argparse.Namespace) -> int:
    args.run_id = time.strftime("%Y%m%d-%H%M%S")
    ab_run = args.ab_root / args.run_id
    ab_run.mkdir(parents=True, exist_ok=True)

    baseline_ref = resolve_baseline_ref(args.baseline_ref)
    baseline_src = ab_run / "baseline" / "src"
    candidate_src = ab_run / "candidate" / "src"
    print(f"Baseline ref: {baseline_ref}", flush=True)
    resolved_baseline = checkout_ref(baseline_src, baseline_ref)
    candidate_label = current_dirty_export(candidate_src)
    print(f"Candidate export: {candidate_label}", flush=True)

    # Ensure gate scripts from the live workspace are available even if the
    # candidate archive is an older tree without them — measurement protocol
    # always comes from ROOT.
    for rel in (
        "benchmarks/gate_matrix_bench.py",
        "benchmarks/gate_index_modes.py",
    ):
        src = ROOT / rel
        for side_src in (baseline_src, candidate_src):
            # Not required inside package install; present for debugging only.
            target = side_src / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    args.baseline_result.parent.mkdir(parents=True, exist_ok=True)
    baseline_json = run_matrix_side(
        args,
        side="baseline",
        src=baseline_src,
        git_ref=resolved_baseline,
        output=args.baseline_result,
        host_python=args.python,
    )
    candidate_json = run_matrix_side(
        args,
        side="candidate",
        src=candidate_src,
        git_ref=candidate_label,
        output=args.candidate_result,
        host_python=args.python,
    )

    cases = evaluate(baseline_json, candidate_json, args)
    failed = print_cases(cases)
    report = {
        "schema_version": 3,
        "mode": "ab",
        "warning": failed,
        "baseline_ref": resolved_baseline,
        "candidate_ref": candidate_label,
        "ab_root": str(ab_run),
        "baseline_result": str(args.baseline_result),
        "candidate_result": str(args.candidate_result),
        "cases": cases,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote report: {args.report}")

    if not args.keep_ab:
        # Keep result JSON copies under benchmarks/baselines; drop bulky venvs/data.
        for side in ("baseline", "candidate"):
            for name in ("venv", "data", "src"):
                path = ab_run / side / name
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
    return 1 if failed else 0


def run_single(args: argparse.Namespace) -> int:
    """Debug helper: run matrix once in the current interpreter environment."""
    output = ROOT / "benchmarks" / "baselines" / "local-perf-self.json"
    data_dir = Path(os.environ.get("GATE_MATRIX_DIR", "/tmp/lynse_gate_matrix_self"))
    cmd = [
        args.python,
        str(MATRIX_BENCH),
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
        "--nprobe",
        str(args.nprobe),
        "--data-dir",
        str(data_dir),
        "--side",
        "self",
        "--git-ref",
        "local",
        "--output",
        str(output),
    ]
    if args.modes:
        cmd.extend(["--modes", args.modes])
    if args.skip_matrix:
        cmd.append("--skip-modes")
    if args.skip_sparse:
        cmd.append("--skip-sparse")
    if args.skip_shared:
        cmd.append("--skip-shared")
    if args.skip_upsert:
        cmd.append("--skip-upsert")
    run(cmd)
    payload = json.loads(output.read_text())
    print(json.dumps({"schema_version": 3, "side": "self", "output": str(output)}, indent=2))
    if args.mode == "self":
        return 0
    if args.mode == "record":
        args.baseline_result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote baseline-side snapshot: {args.baseline_result}")
        return 0
    if not args.baseline_result.exists():
        print(f"Missing baseline snapshot: {args.baseline_result}", file=sys.stderr)
        return 2
    baseline = json.loads(args.baseline_result.read_text())
    cases = evaluate(baseline, payload, args)
    failed = print_cases(cases)
    args.report.write_text(
        json.dumps(
            {"schema_version": 3, "mode": args.mode, "warning": failed, "cases": cases},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("RAYON_NUM_THREADS", "4"))
    if args.mode == "ab":
        return run_ab(args)
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
