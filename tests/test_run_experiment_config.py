import sys
from pathlib import Path


def test_run_experiment_loads_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
llm:
  model: gpt-4o-mini
""")

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from run_experiment import load_resolved_config

        config = load_resolved_config(config_path)
    finally:
        sys.path.remove(str(scripts_dir))

    assert "llm" in config
