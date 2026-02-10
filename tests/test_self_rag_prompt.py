"""Tests for the Self-RAG prompt template."""

from pathlib import Path


def test_self_rag_prompt_file_exists() -> None:
    assert Path("prompts/self_rag.txt").exists()


def test_self_rag_prompt_includes_retrieval_token() -> None:
    prompt = Path("prompts/self_rag.txt").read_text()
    assert "[Retrieval]" in prompt
