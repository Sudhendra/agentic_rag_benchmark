"""Test that the Self-RAG prompt template exists."""

from pathlib import Path


def test_self_rag_prompt_file_exists() -> None:
    assert Path("prompts/self_rag.txt").exists()
