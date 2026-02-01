"""Core abstractions: types, base classes, LLM client, retriever interface."""

from .types import (
    Question,
    QuestionType,
    Document,
    RetrievalResult,
    ReasoningStep,
    RAGResponse,
    EvaluationResult,
    BenchmarkResult,
    ArchitectureType,
)

__all__ = [
    "Question",
    "QuestionType",
    "Document",
    "RetrievalResult",
    "ReasoningStep",
    "RAGResponse",
    "EvaluationResult",
    "BenchmarkResult",
    "ArchitectureType",
]
