from pathlib import Path


def test_react_prompt_file_exists() -> None:
    assert Path("prompts/react.txt").exists()
