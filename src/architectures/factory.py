"""Factory for creating RAG architectures."""

from __future__ import annotations

from typing import Any

from .agentic.react_rag import ReActRAG
from .vanilla_rag import VanillaRAG
from ..core.base_rag import BaseRAG
from ..core.llm_client import BaseLLMClient
from ..core.retriever import BaseRetriever


def _resolve_architecture_config(name: str, config: dict[str, Any]) -> dict[str, Any]:
    if name == "vanilla_rag" and "vanilla" in config:
        return config.get("vanilla", {})
    if name == "react_rag" and "react" in config:
        return config.get("react", {})
    return config


def create_architecture(
    name: str,
    llm: BaseLLMClient,
    retriever: BaseRetriever,
    config: dict[str, Any],
) -> BaseRAG:
    """Create a RAG architecture by name."""
    resolved_config = _resolve_architecture_config(name, config)
    if name == "vanilla_rag":
        return VanillaRAG(llm, retriever, resolved_config)
    if name == "react_rag":
        return ReActRAG(llm, retriever, resolved_config)

    raise ValueError(f"Unknown architecture: {name}")
