"""Repository-level entrypoint with dependency-aware pipeline stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _ensure_src_on_path(repo_root: Path) -> None:
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _import_config(repo_root: Path):
    _ensure_src_on_path(repo_root)
    from core import project_config as config  # type: ignore

    return config


def _script_path(repo_root: Path, relative_path: str) -> Path:
    return repo_root / relative_path


def _run_script(script: Path, repo_root: Path) -> int:
    print(f"[pipeline] run: {script.relative_to(repo_root)}", flush=True)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(repo_root))
    return int(proc.returncode)


def _require_files(paths: list[Path], stage_name: str) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"[pipeline] stage '{stage_name}' missing required artifacts:\n{lines}\n"
            "Run '--steps train' first, or provide the expected files."
        )


def _stage_train(repo_root: Path) -> list[Path]:
    return [_script_path(repo_root, "src/pipelines/stage_train.py")]


def _stage_analysis(repo_root: Path) -> list[Path]:
    return [_script_path(repo_root, "src/analysis/stage_analysis.py")]


def _stage_figures(repo_root: Path) -> list[Path]:
    return [_script_path(repo_root, "src/figures/stage_figures.py")]


def _build_stage_map(repo_root: Path):
    return {
        "train": _stage_train(repo_root),
        "analysis": _stage_analysis(repo_root),
        "figures": _stage_figures(repo_root),
    }


def _check_stage_dependencies(stage: str, config_module) -> None:
    core_required = [
        Path(config_module.DATA_NPZ_PATH),
        Path(config_module.RF_SUMMER_MODEL_PATH),
    ]
    if stage in {"analysis", "figures"}:
        _require_files(core_required, stage)
    if stage == "analysis":
        spring_required = [
            Path(config_module.SPRING_DATA_NPZ_PATH),
            Path(config_module.RF_SPRING_MODEL_PATH),
        ]
        _require_files(spring_required, stage)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dependency-aware runner for this repo. "
            "Default order: train -> analysis -> figures."
        )
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["train", "analysis", "figures"],
        default=["train", "analysis", "figures"],
        help="Pipeline stages to run in the given order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan only; do not run scripts.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one script fails.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent
    config_module = _import_config(repo_root)
    stage_map = _build_stage_map(repo_root)
    t0 = time.perf_counter()

    execution_plan: list[tuple[str, Path]] = []
    for stage in args.steps:
        for script in stage_map[stage]:
            execution_plan.append((stage, script))

    # De-duplicate while preserving order.
    unique_execution_plan: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for stage, script in execution_plan:
        if script not in seen:
            unique_execution_plan.append((stage, script))
            seen.add(script)

    print("[pipeline] execution order:")
    for idx, (_, script) in enumerate(unique_execution_plan, start=1):
        print(f"  {idx:02d}. {script.relative_to(repo_root)}")

    if args.dry_run:
        print("[pipeline] dry-run enabled; no scripts executed.")
        return 0

    failures: list[tuple[Path, int]] = []
    checked_stages: set[str] = set()
    for stage, script in unique_execution_plan:
        if stage != "train" and stage not in checked_stages:
            _check_stage_dependencies(stage, config_module)
            checked_stages.add(stage)
        code = _run_script(script, repo_root)
        if code != 0:
            failures.append((script, code))
            print(f"[pipeline] FAILED ({code}): {script.relative_to(repo_root)}", flush=True)
            if args.stop_on_error:
                break
        else:
            print(f"[pipeline] OK: {script.relative_to(repo_root)}", flush=True)

    print("[pipeline] summary:")
    print(f"  total:   {len(unique_execution_plan)}")
    print(f"  success: {len(unique_execution_plan) - len(failures)}")
    print(f"  failed:  {len(failures)}")
    print(f"  elapsed: {time.perf_counter() - t0:.1f}s")
    if failures:
        print("  failed scripts:")
        for script, code in failures:
            print(f"    - {script.relative_to(repo_root)} (exit={code})")
        print("  hint: use '--stop-on-error' to stop at first failure.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
