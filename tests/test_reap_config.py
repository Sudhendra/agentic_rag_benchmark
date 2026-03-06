"""Tests for REAP factory and config integration."""

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


def test_create_architecture_reap(mock_llm, mock_retriever):
    rag = create_architecture("reap_rag", mock_llm, mock_retriever, {"reap": {"max_iterations": 4}})
    assert rag.get_name() == "reap_rag"


def test_reap_yaml_config_loads():
    config = load_config(Path("configs/reap.yaml"))
    assert config["architecture"]["name"] == "reap_rag"
    assert config["reap"]["max_iterations"] == 5


def test_reap_subset_config_uses_small_dev_subset():
    config = load_config(Path("configs/reap.yaml"))
    assert config["data"]["subset_size"] == 100


def test_reap_full_config_uses_null_subset():
    config = load_config(Path("configs/reap_bm25_full.yaml"))
    assert config["data"]["subset_size"] is None
