from pathlib import Path

from src.utils.config import load_config


def test_config_inherits_base(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        """
llm:
  model: gpt-4o-mini
retrieval:
  top_k: 5
"""
    )

    child = tmp_path / "child.yaml"
    child.write_text(
        """
inherits: base.yaml
retrieval:
  top_k: 3
"""
    )

    config = load_config(child)

    assert config["llm"]["model"] == "gpt-4o-mini"
    assert config["retrieval"]["top_k"] == 3
