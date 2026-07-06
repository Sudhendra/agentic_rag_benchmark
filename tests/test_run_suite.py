import importlib
import sys
from pathlib import Path


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_run_suite_dry_run_filters_without_executing(monkeypatch, tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", "llm:\n  model: gpt-4o-mini\n")
    react = _write(tmp_path / "react.yaml", "architecture:\n  name: react_rag\n")
    rlm = _write(tmp_path / "rlm.yaml", "architecture:\n  name: recursive_lm\n")
    dataset = _write(tmp_path / "hotpotqa.yaml", "data:\n  dataset: hotpotqa\n")
    suite = _write(
        tmp_path / "suite.yaml",
        f"""
suite:
  name: dry_run
  output_dir: {tmp_path / "results"}
  base_config: {base}
  matrix:
    architecture: [{react}, {rlm}]
    dataset: [{dataset}]
""",
    )

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_suite = importlib.import_module("run_suite")
    finally:
        sys.path.remove(str(scripts_dir))

    async def fail_run_experiment(config):
        raise AssertionError("dry run must not execute experiments")

    monkeypatch.setattr(run_suite, "run_experiment", fail_run_experiment)

    selected = run_suite.prepare_suite_runs(suite, filters=["architecture=recursive_lm"])
    run_suite.print_dry_run(selected)

    assert len(selected) == 1
    assert selected[0].config["architecture"]["name"] == "recursive_lm"
