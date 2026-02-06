from pathlib import Path

from src.utils.config import load_config


def test_react_config_loads(tmp_path: Path) -> None:
    config = load_config(Path("configs/react.yaml"))
    assert config["architecture"]["name"] == "react_rag"
