"""Core type definitions for the RAG benchmark."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np


class QuestionType(Enum):
    """Types of multi-hop questions."""

    BRIDGE = "bridge"  # Entity linked through intermediate
    COMPARISON = "comparison"  # Compare two entities
    COMPOSITIONAL = "compositional"  # Nested sub-questions
    SINGLE_HOP = "single_hop"  # Baseline single retrieval


class ArchitectureType(Enum):
    """RAG architecture categories."""

    VANILLA = "vanilla"
    AGENTIC = "agentic"
    RECURSIVE = "recursive"
    RLM = "rlm"


@dataclass
class Question:
    """Represents a question from the dataset."""

    id: str
    text: str
    type: QuestionType
    gold_answer: str | None = None
    supporting_facts: list[tuple[str, int]] | None = None  # (title, sent_idx)
    decomposition: list[str] | None = None  # Sub-questions if available
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    """Represents a document/passage in the corpus."""

    id: str
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None

    def __post_init__(self):
        """Auto-split text into sentences if not provided."""
        if not self.sentences and self.text:
            # Simple sentence splitting - can be improved
            self.sentences = [s.strip() + "." for s in self.text.split(".") if s.strip()]


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    documents: list[Document]
    scores: list[float]
    query: str
    retrieval_time_ms: float
    method: Literal["bm25", "dense", "hybrid"]


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_id: int
    thought: str
    action: str  # "search", "lookup", "finish", "reflect", "decompose", "recurse"
    action_input: str
    observation: str
    tokens_used: int = 0
    cost_usd: float = 0.0


@dataclass
class RAGResponse:
    """Complete response from a RAG system."""

    answer: str
    reasoning_chain: list[ReasoningStep]
    retrieved_docs: list[RetrievalResult]
    total_tokens: int
    total_cost_usd: float
    latency_ms: float
    num_retrieval_calls: int
    num_llm_calls: int
    model: str
    architecture: str


@dataclass
class EvaluationResult:
    """Evaluation metrics for a single question."""

    question_id: str
    question_type: QuestionType
    exact_match: float
    f1: float
    predicted_answer: str
    gold_answer: str
    supporting_fact_em: float | None = None
    supporting_fact_f1: float | None = None
    joint_em: float | None = None
    joint_f1: float | None = None
    # Efficiency metrics
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    num_retrieval_calls: int = 0
    num_llm_calls: int = 0


@dataclass
class BenchmarkResult:
    """Aggregated results for a benchmark run."""

    architecture: str
    architecture_type: ArchitectureType
    model: str
    dataset: str
    num_questions: int
    # Accuracy metrics
    avg_exact_match: float
    avg_f1: float
    avg_supporting_fact_em: float | None
    avg_supporting_fact_f1: float | None
    # Breakdown by question type
    metrics_by_type: dict[QuestionType, dict[str, float]]
    # Efficiency metrics
    avg_latency_ms: float
    avg_tokens_per_question: float
    avg_retrieval_calls: float
    avg_llm_calls: float
    total_cost_usd: float
    total_tokens: int
    # Raw results
    per_question_results: list[EvaluationResult]
