"""Tests for IRCoT factory and config integration."""

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


def test_create_architecture_ircot(mock_llm, mock_retriever):
    rag = create_architecture("ircot_rag", mock_llm, mock_retriever, {"ircot": {"max_steps": 4}})
    assert rag.get_name() == "ircot_rag"


def test_ircot_yaml_config_loads():
    config = load_config(Path("configs/ircot.yaml"))
    assert config["architecture"]["name"] == "ircot_rag"
    assert config["ircot"]["max_steps"] == 4


def test_ircot_subset_config_uses_small_dev_subset():
    config = load_config(Path("configs/ircot.yaml"))
    assert config["data"]["subset_size"] == 100


def test_ircot_full_config_loads():
    config = load_config(Path("configs/ircot_bm25_full.yaml"))
    assert config["architecture"]["name"] == "ircot_rag"
    assert config["data"]["subset_size"] is None
