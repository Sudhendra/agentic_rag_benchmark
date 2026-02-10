"""Factory for creating RAG architectures."""

from __future__ import annotations

from typing import Any

from ..core.base_rag import BaseRAG
from ..core.llm_client import BaseLLMClient
from ..core.retriever import BaseRetriever
from .agentic.react_rag import ReActRAG
from .agentic.self_rag import SelfRAG
from .vanilla_rag import VanillaRAG


def _resolve_architecture_config(name: str, config: dict[str, Any]) -> dict[str, Any]:
    nested_key = None
    if name == "vanilla_rag":
        nested_key = "vanilla"
    elif name == "react_rag":
        nested_key = "react"
    elif name == "self_rag":
        nested_key = "self_rag"

    nested_config = config.get(nested_key, {}) if nested_key else {}
    merged_config = {**config, **nested_config}
    if nested_key and nested_key in merged_config:
        merged_config.pop(nested_key, None)
    return merged_config


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
    if name == "self_rag":
        return SelfRAG(llm, retriever, resolved_config)

    raise ValueError(f"Unknown architecture: {name}")
