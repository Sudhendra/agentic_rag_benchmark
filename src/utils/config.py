from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if "inherits" in data:
        inherits = data.pop("inherits")
        if isinstance(inherits, (str, Path)):
            inherit_paths = [inherits]
        elif isinstance(inherits, list):
            inherit_paths = inherits
        else:
            raise TypeError("inherits must be a string or list of strings")

        base: dict[str, Any] = {}
        for inherit_path in inherit_paths:
            resolved_path = Path(inherit_path)
            if not resolved_path.is_absolute():
                resolved_path = Path(path).parent / resolved_path
            base = deep_merge(base, load_config(resolved_path))
        return deep_merge(base, data)
    return data
