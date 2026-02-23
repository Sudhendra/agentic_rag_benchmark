import sys
import importlib
from pathlib import Path
from types import SimpleNamespace


def test_run_experiment_loads_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
llm:
  model: gpt-4o-mini
""")

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_experiment = importlib.import_module("run_experiment")
        load_resolved_config = run_experiment.load_resolved_config

        config = load_resolved_config(config_path)
    finally:
        sys.path.remove(str(scripts_dir))

    assert "llm" in config


def test_planner_uses_planner_default_context_tokens_when_not_set(monkeypatch) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_experiment = importlib.import_module("run_experiment")
    finally:
        sys.path.remove(str(scripts_dir))

    captured: dict[str, object] = {}

    def fake_create_llm_client(**kwargs):
        return SimpleNamespace(model="test-model")

    def fake_create_retriever(**kwargs):
        return object()

    def fake_create_architecture(name, llm, retriever, config):
        captured["name"] = name
        captured["config"] = config
        return SimpleNamespace()

    monkeypatch.setattr(run_experiment, "create_llm_client", fake_create_llm_client)
    monkeypatch.setattr(run_experiment, "create_retriever", fake_create_retriever)
    monkeypatch.setattr(run_experiment, "create_architecture", fake_create_architecture)

    config = {
        "architecture": {"name": "planner_rag"},
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 1024},
        "retrieval": {"method": "bm25", "top_k": 3},
        "planner": {"max_iterations": 5},
    }

    run_experiment._build_rag(config)

    assert captured["name"] == "planner_rag"
    architecture_config = captured["config"]
    assert isinstance(architecture_config, dict)
    assert architecture_config["top_k"] == 3
    assert "max_context_tokens" not in architecture_config
