"""Tests for Planner RAG factory and config integration."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.architectures.factory import create_architecture
from src.utils.config import load_config


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.model = "test-model"
    return llm


@pytest.fixture
def mock_retriever():
    return AsyncMock()


def test_create_architecture_planner_rag(mock_llm, mock_retriever):
    rag = create_architecture(
        "planner_rag",
        mock_llm,
        mock_retriever,
        {"planner": {"max_iterations": 4}},
    )
    assert rag.get_name() == "planner_rag"


def test_planner_rag_config_merging(mock_llm, mock_retriever):
    rag = create_architecture(
        "planner_rag",
        mock_llm,
        mock_retriever,
        {
            "top_k": 7,
            "planner": {"max_iterations": 8, "max_branching_factor": 3},
        },
    )
    assert rag.config["top_k"] == 7
    assert rag.config["max_iterations"] == 8
    assert rag.config["max_branching_factor"] == 3


def test_planner_yaml_config_loads():
    config = load_config(Path("configs/planner.yaml"))
    assert config["architecture"]["name"] == "planner_rag"
    assert config["planner"]["max_iterations"] == 5
    assert config["planner"]["rollout_similarity_threshold"] == 0.85
    assert config["planner"]["bridge_refine_enabled"] is True
    assert config["planner"]["bridge_refine_max_attempts"] == 1


def test_planner_max_context_tokens_default_is_consistent():
    config = load_config(Path("configs/planner.yaml"))
    assert config["planner"]["max_context_tokens"] == 4000
