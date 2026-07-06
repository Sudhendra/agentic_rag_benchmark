from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_experiment import run_experiment  # noqa: E402
from src.utils.suite_config import (  # noqa: E402
    ExperimentSpec,
    expand_suite_config,
    filter_experiments,
    load_suite_config,
)


def prepare_suite_runs(suite_path: Path, filters: list[str] | None = None) -> list[ExperimentSpec]:
    suite = load_suite_config(suite_path)
    experiments = expand_suite_config(suite)
    return filter_experiments(experiments, filters)


def print_dry_run(experiments: list[ExperimentSpec]) -> None:
    print("Suite dry run")
    print(f"Selected runs: {len(experiments)}")
    for index, experiment in enumerate(experiments, start=1):
        architecture = experiment.config.get("architecture", {}).get("name")
        dataset = experiment.config.get("data", {}).get("dataset")
        retriever = experiment.config.get("retrieval", {}).get("method")
        model = experiment.config.get("llm", {}).get("model")
        subset = experiment.config.get("data", {}).get("subset_size")
        print(
            f"{index:03d}. {experiment.name} | "
            f"arch={architecture} dataset={dataset} retriever={retriever} "
            f"model={model} subset={subset}"
        )


async def run_suite(
    suite_path: Path,
    filters: list[str] | None = None,
    skip_existing: bool = False,
) -> list[Path]:
    experiments = prepare_suite_runs(suite_path, filters)
    run_dirs: list[Path] = []
    progress_path = _progress_path(experiments, suite_path)
    progress = _load_progress(progress_path)

    _write_manifest(suite_path, experiments, progress_path.parent)
    for experiment in experiments:
        if skip_existing and experiment.name in progress:
            print(f"[Suite] Skipping existing run: {experiment.name}", flush=True)
            continue
        print(f"[Suite] Running {experiment.name}", flush=True)
        run_dir = await run_experiment(experiment.config)
        run_dirs.append(run_dir)
        progress[experiment.name] = str(run_dir)
        _save_progress(progress_path, progress)

    return run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a declarative benchmark suite")
    parser.add_argument("--suite", required=True, help="Path to suite YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded runs without executing")
    parser.add_argument("--only", action="append", default=[], help="Filter runs by key=value")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs in suite_progress.json")
    parser.add_argument("--yes", action="store_true", help="Execute without interactive confirmation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = Path(args.suite)
    experiments = prepare_suite_runs(suite_path, args.only)

    if args.dry_run:
        print_dry_run(experiments)
        return

    if not args.yes:
        print_dry_run(experiments)
        raise SystemExit("Refusing to execute without --yes. Re-run with --yes after reviewing.")

    asyncio.run(run_suite(suite_path, filters=args.only, skip_existing=args.skip_existing))


def _progress_path(experiments: list[ExperimentSpec], suite_path: Path) -> Path:
    output_dir = None
    if experiments:
        output_dir = experiments[0].config.get("experiment", {}).get("output_dir")
    if output_dir is None:
        output_dir = Path("results") / Path(suite_path).stem
    return Path(output_dir) / "suite_progress.json"


def _load_progress(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_progress(path: Path, progress: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2, sort_keys=True))


def _write_manifest(
    suite_path: Path,
    experiments: list[ExperimentSpec],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "suite_path": str(suite_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "num_runs": len(experiments),
        "runs": [
            {
                "name": experiment.name,
                "architecture": experiment.config.get("architecture", {}).get("name"),
                "dataset": experiment.config.get("data", {}).get("dataset"),
                "retriever": experiment.config.get("retrieval", {}).get("method"),
                "model": experiment.config.get("llm", {}).get("model"),
                "components": {
                    key: str(path) for key, path in experiment.components.items()
                },
            }
            for experiment in experiments
        ],
    }
    (output_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
