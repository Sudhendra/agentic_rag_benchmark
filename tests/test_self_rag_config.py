"""Tests for Self-RAG factory and config integration."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.architectures.factory import create_architecture


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.model = "test-model"
    return llm


@pytest.fixture
def mock_retriever():
    return AsyncMock()


def test_create_architecture_self_rag(mock_llm, mock_retriever):
    """Factory should create SelfRAG when name is 'self_rag'."""
    rag = create_architecture(
        "self_rag", mock_llm, mock_retriever, {"self_rag": {"num_candidates": 2}}
    )
    assert rag.get_name() == "self_rag"


def test_self_rag_config_merging(mock_llm, mock_retriever):
    """Factory should merge nested self_rag config into top-level."""
    rag = create_architecture(
        "self_rag",
        mock_llm,
        mock_retriever,
        {"self_rag": {"num_candidates": 5}, "top_k": 10},
    )
    assert rag.config["num_candidates"] == 5
    assert rag.config["top_k"] == 10
