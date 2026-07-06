from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from .config import deep_merge, load_config


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    path: Path
    output_dir: str | None
    base_config: Path | None
    defaults: dict[str, Any]
    matrix: dict[str, list[Path]]
    exclude: list[dict[str, Any]]
    run: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    config: dict[str, Any]
    components: dict[str, Path]


def load_suite_config(path: Path) -> SuiteConfig:
    suite_path = Path(path)
    data = yaml.safe_load(suite_path.read_text()) or {}
    suite_data = data.get("suite")
    if not isinstance(suite_data, dict):
        raise ValueError("Suite YAML must contain a 'suite' mapping")

    matrix_data = suite_data.get("matrix", {})
    if not isinstance(matrix_data, dict) or not matrix_data:
        raise ValueError("Suite YAML must define a non-empty suite.matrix mapping")

    matrix: dict[str, list[Path]] = {}
    for dimension, entries in matrix_data.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"suite.matrix.{dimension} must be a non-empty list")
        matrix[dimension] = [_resolve_path(suite_path, entry) for entry in entries]

    base_config = suite_data.get("base_config")
    return SuiteConfig(
        name=str(suite_data.get("name") or suite_path.stem),
        path=suite_path,
        output_dir=suite_data.get("output_dir"),
        base_config=_resolve_path(suite_path, base_config) if base_config else None,
        defaults=suite_data.get("defaults", {}) or {},
        matrix=matrix,
        exclude=suite_data.get("exclude", []) or [],
        run=suite_data.get("run", {}) or {},
    )


def expand_suite_config(suite: SuiteConfig) -> list[ExperimentSpec]:
    base = load_config(suite.base_config) if suite.base_config else {}
    dimensions = list(suite.matrix.keys())
    experiments: list[ExperimentSpec] = []

    for component_paths in product(*(suite.matrix[dimension] for dimension in dimensions)):
        components = dict(zip(dimensions, component_paths, strict=True))
        config = deep_merge(base, suite.defaults)
        for component_path in component_paths:
            config = deep_merge(config, load_config(component_path))

        if _is_excluded(config, suite.exclude):
            continue

        name = _build_experiment_name(config)
        experiment_config = deep_merge(
            config,
            {
                "experiment": {
                    "name": name,
                    **({"output_dir": suite.output_dir} if suite.output_dir else {}),
                },
                "suite": {
                    "name": suite.name,
                    "path": str(suite.path),
                    "components": {
                        dimension: str(component_path)
                        for dimension, component_path in components.items()
                    },
                },
            },
        )
        experiments.append(ExperimentSpec(name=name, config=experiment_config, components=components))

    return experiments


def filter_experiments(
    experiments: list[ExperimentSpec],
    filters: list[str] | None,
) -> list[ExperimentSpec]:
    if not filters:
        return experiments

    parsed_filters = [_parse_filter(item) for item in filters]
    return [
        experiment
        for experiment in experiments
        if all(_matches_filter(experiment.config, key, value) for key, value in parsed_filters)
    ]


def _resolve_path(suite_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return suite_path.parent / path


def _is_excluded(config: dict[str, Any], exclude_rules: list[dict[str, Any]]) -> bool:
    for rule in exclude_rules:
        criteria = {key: value for key, value in rule.items() if key != "reason"}
        if criteria and all(_matches_filter(config, key, str(value)) for key, value in criteria.items()):
            return True
    return False


def _parse_filter(filter_text: str) -> tuple[str, str]:
    if "=" not in filter_text:
        raise ValueError(f"Filter must use key=value format: {filter_text}")
    key, value = filter_text.split("=", 1)
    return key.strip(), value.strip()


def _matches_filter(config: dict[str, Any], key: str, value: str) -> bool:
    actual = _lookup_config_value(config, key)
    return actual is not None and str(actual) == value


def _lookup_config_value(config: dict[str, Any], key: str) -> Any:
    aliases = {
        "architecture": "architecture.name",
        "arch": "architecture.name",
        "dataset": "data.dataset",
        "retriever": "retrieval.method",
        "retrieval": "retrieval.method",
        "model": "llm.model",
        "seed": "experiment.seed",
    }
    path = aliases.get(key, key)
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _build_experiment_name(config: dict[str, Any]) -> str:
    architecture = _lookup_config_value(config, "architecture") or "unknown_architecture"
    dataset = _lookup_config_value(config, "dataset") or "unknown_dataset"
    retriever = _lookup_config_value(config, "retriever") or "unknown_retriever"
    model = _lookup_config_value(config, "model") or "unknown_model"
    seed = _lookup_config_value(config, "seed")

    name_parts = [architecture, dataset, retriever, model]
    if seed is not None:
        name_parts.append(f"seed{seed}")
    return "__".join(_safe_name(str(part)) for part in name_parts)


def _safe_name(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_")
