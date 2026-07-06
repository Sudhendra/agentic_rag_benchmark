from pathlib import Path

from src.utils.suite_config import expand_suite_config, filter_experiments, load_suite_config


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_suite_expands_matrix_into_deterministic_experiment_configs(tmp_path: Path) -> None:
    base = _write(
        tmp_path / "base.yaml",
        """
experiment:
  seed: 42
llm:
  provider: openai
  model: gpt-4o-mini
retrieval:
  top_k: 5
data:
  split: validation
""",
    )
    vanilla = _write(
        tmp_path / "components" / "vanilla.yaml",
        """
architecture:
  name: vanilla_rag
vanilla:
  prompt_path: prompts/vanilla.txt
""",
    )
    react = _write(
        tmp_path / "components" / "react.yaml",
        """
architecture:
  name: react_rag
react:
  max_iterations: 7
""",
    )
    hotpotqa = _write(
        tmp_path / "components" / "hotpotqa.yaml",
        """
data:
  dataset: hotpotqa
  subset_size: null
""",
    )
    bm25 = _write(
        tmp_path / "components" / "bm25.yaml",
        """
retrieval:
  method: bm25
""",
    )
    dense = _write(
        tmp_path / "components" / "dense.yaml",
        """
retrieval:
  method: dense
""",
    )
    suite = _write(
        tmp_path / "suites" / "main.yaml",
        f"""
suite:
  name: main
  output_dir: results/main
  base_config: {base}
  defaults:
    evaluation:
      max_concurrency: 2
  matrix:
    architecture:
      - {vanilla}
      - {react}
    dataset:
      - {hotpotqa}
    retriever:
      - {bm25}
      - {dense}
""",
    )

    experiments = expand_suite_config(load_suite_config(suite))

    assert [experiment.name for experiment in experiments] == [
        "vanilla_rag__hotpotqa__bm25__gpt-4o-mini__seed42",
        "vanilla_rag__hotpotqa__dense__gpt-4o-mini__seed42",
        "react_rag__hotpotqa__bm25__gpt-4o-mini__seed42",
        "react_rag__hotpotqa__dense__gpt-4o-mini__seed42",
    ]
    assert experiments[0].config["experiment"]["name"] == experiments[0].name
    assert experiments[0].config["experiment"]["output_dir"] == "results/main"
    assert experiments[0].config["evaluation"]["max_concurrency"] == 2
    assert experiments[2].config["react"]["max_iterations"] == 7


def test_suite_excludes_matching_matrix_combinations(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", "llm:\n  model: gpt-4o-mini\n")
    architecture = _write(
        tmp_path / "react.yaml", "architecture:\n  name: react_rag\n"
    )
    dataset = _write(tmp_path / "2wiki.yaml", "data:\n  dataset: 2wikimultihop\n")
    bm25 = _write(tmp_path / "bm25.yaml", "retrieval:\n  method: bm25\n")
    dense = _write(tmp_path / "dense.yaml", "retrieval:\n  method: dense\n")
    suite = _write(
        tmp_path / "suite.yaml",
        f"""
suite:
  name: exclusions
  base_config: {base}
  matrix:
    architecture: [{architecture}]
    dataset: [{dataset}]
    retriever: [{bm25}, {dense}]
  exclude:
    - dataset: 2wikimultihop
      retriever: dense
      reason: defer dense embedding cost
""",
    )

    experiments = expand_suite_config(load_suite_config(suite))

    assert len(experiments) == 1
    assert experiments[0].config["retrieval"]["method"] == "bm25"


def test_filter_experiments_matches_resolved_config_values(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", "llm:\n  model: gpt-4o-mini\n")
    react = _write(tmp_path / "react.yaml", "architecture:\n  name: react_rag\n")
    rlm = _write(tmp_path / "rlm.yaml", "architecture:\n  name: recursive_lm\n")
    dataset = _write(tmp_path / "hotpotqa.yaml", "data:\n  dataset: hotpotqa\n")
    suite = _write(
        tmp_path / "suite.yaml",
        f"""
suite:
  name: filters
  base_config: {base}
  matrix:
    architecture: [{react}, {rlm}]
    dataset: [{dataset}]
""",
    )
    experiments = expand_suite_config(load_suite_config(suite))

    filtered = filter_experiments(experiments, ["architecture=recursive_lm"])

    assert [experiment.config["architecture"]["name"] for experiment in filtered] == [
        "recursive_lm"
    ]


def test_checked_in_dev_smoke_suite_uses_valid_relative_component_paths() -> None:
    experiments = expand_suite_config(load_suite_config(Path("configs/suites/dev_smoke.yaml")))

    assert len(experiments) == 3
    assert [experiment.config["architecture"]["name"] for experiment in experiments] == [
        "vanilla_rag",
        "ircot_rag",
        "recursive_lm",
    ]
    assert {experiment.config["data"]["dataset"] for experiment in experiments} == {"hotpotqa"}
    assert {experiment.config["retrieval"]["method"] for experiment in experiments} == {"bm25"}
