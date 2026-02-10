# Self-RAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Self-RAG architecture with retrieval decision, relevance filtering, generation with critique, and best-candidate selection — fully integrated into the existing experiment runner, factory, and evaluation pipeline.

**Architecture:** Self-RAG adds self-reflection tokens to the RAG pipeline. The LLM first decides whether retrieval is needed (`[Retrieval]`), then filters retrieved passages for relevance (`[IsRel]`), generates candidate answers with support critique (`[IsSup]`), scores overall utility (`[IsUse]`), and selects the best candidate. This creates a 4-phase pipeline: retrieval decision → relevance filtering → generation with critique → best candidate selection. When no retrieval is needed, the LLM answers directly.

**Tech Stack:** Python 3.11, async/await, OpenAI/Anthropic clients (via existing `BaseLLMClient`), existing retrievers, pytest with AsyncMock.

**Key Reference:** Self-RAG paper (ICLR 2024) — https://arxiv.org/abs/2310.11511, TECHNICAL_SPEC.md §4.3

---

## Codebase Context (read this before implementing)

**Patterns to follow (match ReAct RAG exactly):**
- Architecture class inherits from `BaseRAG` (`src/core/base_rag.py`)
- Must implement: `answer()`, `get_name()`, `get_type()`, `get_config_schema()`
- Config schema returns `dict[str, tuple[type, bool, Any]]` — `(type, required, default)`
- Prompt loaded via `self._load_prompt_template(path)` with fallback inline string
- Context built via `self._build_context(documents, max_tokens=...)`
- Factory at `src/architectures/factory.py` maps `"self_rag"` → `SelfRAG`
- Runner at `scripts/run_experiment.py` uses `_build_rag()` which reads `config["architecture"]["name"]` and merges `config.get(architecture_name, {})` for the architecture config
- Configs inherit from `base.yaml` via `inherits:` key
- Tests use `AsyncMock` for LLM and retriever, no real API calls
- `ReasoningStep` has: `step_id`, `thought`, `action`, `action_input`, `observation`, `tokens_used`, `cost_usd`
- `RAGResponse` tracks: `answer`, `reasoning_chain`, `retrieved_docs`, `total_tokens`, `total_cost_usd`, `latency_ms`, `num_retrieval_calls`, `num_llm_calls`, `model`, `architecture`

**Files you will create:**
- `prompts/self_rag.txt`
- `src/architectures/agentic/self_rag.py`
- `tests/test_self_rag.py`
- `tests/test_self_rag_prompt.py`
- `tests/test_self_rag_config.py`
- `configs/self_rag.yaml`
- `configs/self_rag_dense.yaml`
- `configs/self_rag_hybrid.yaml`
- `configs/self_rag_bm25_full.yaml`
- `configs/self_rag_dense_full.yaml`
- `configs/self_rag_hybrid_full.yaml`

**Files you will modify:**
- `src/architectures/agentic/__init__.py` — add `SelfRAG` export
- `src/architectures/factory.py` — add `"self_rag"` → `SelfRAG` mapping
- `scripts/run_experiment.py` — add `self_rag` config resolution in `_build_rag()`
- `README.md` — add Self-RAG rows to results table

---

## Self-RAG Algorithm Design

The Self-RAG architecture uses 4 reflection tokens to guide the pipeline. Our implementation simulates these via LLM prompts (since we use API-only models, not fine-tuned ones):

```
Phase 1 — Retrieval Decision:
  prompt LLM: "Given {question}, do you need to retrieve information? Answer: yes or no"
  parse response → retrieval_needed (bool)

Phase 2 — If retrieval needed:
  passages = retriever.retrieve(question, corpus, top_k)
  
  For each passage:
    prompt LLM: "Is this passage relevant to {question}? Answer: relevant or irrelevant"
    → filter to relevant_passages

Phase 3 — Generate candidate answers from relevant passages:
  For each relevant passage (up to num_candidates):
    prompt LLM: "Answer {question} based on: {passage}"
    → generation

    prompt LLM: "Does {passage} support {generation}? Answer: fully/partially/no"
    → support_level

    prompt LLM: "Rate utility of {generation} for {question}: 1-5"
    → utility_score

Phase 4 — Select best candidate:
  score = utility + support_bonus
  return highest scoring candidate

Phase 1 alt — If NO retrieval needed:
  prompt LLM: "Answer {question} directly"
  return direct answer
```

**Cost-optimization:** We batch the relevance, support, and utility assessments into single prompts per passage to reduce LLM calls. Each candidate assessment (relevance + generation + support + utility) is done as 3 LLM calls instead of 4 by combining relevance check with generation.

**Actual implementation:** To keep LLM calls manageable, we use a 2-call-per-candidate approach:
1. **Retrieval decision** — 1 LLM call
2. **Retrieve** — 1 retriever call
3. **Per-candidate (up to `num_candidates`):**
   - **Generate + assess relevance/support** — 1 LLM call (combined prompt)
   - **Rate utility** — 1 LLM call
4. **Select best** — pure logic, no LLM call

Total LLM calls: `1 + (2 × num_candidates)` when retrieval is needed, `2` when not.

---

### Task 1: Add Self-RAG prompt template

**Files:**
- Create: `prompts/self_rag.txt`
- Create: `tests/test_self_rag_prompt.py`

**Step 1: Write the failing test**

Create `tests/test_self_rag_prompt.py`:

```python
"""Test that the Self-RAG prompt template exists."""

from pathlib import Path


def test_self_rag_prompt_file_exists() -> None:
    assert Path("prompts/self_rag.txt").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_self_rag_prompt.py::test_self_rag_prompt_file_exists -v`
Expected: FAIL with `AssertionError`

**Step 3: Write minimal implementation**

Create `prompts/self_rag.txt`:

```
You are a self-reflective question answering system. You must decide whether to retrieve information and critically evaluate your own generations.

## Retrieval Decision

Given the question, decide if you need to retrieve external information or can answer directly.

Question: {question}

Do you need to retrieve information to answer this question accurately?
Answer with exactly "yes" or "no":
```

This is the retrieval-decision prompt. The other prompts (relevance, generation, support, utility) will be inline in the architecture code since they are contextual and short. This matches how ReAct handles its tool-use prompts inline while keeping the main template in a file.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_self_rag_prompt.py::test_self_rag_prompt_file_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add prompts/self_rag.txt tests/test_self_rag_prompt.py
git commit -m "docs: add self-rag prompt template"
```

---

### Task 2: Implement SelfRAG architecture class

**Files:**
- Create: `src/architectures/agentic/self_rag.py`
- Modify: `src/architectures/agentic/__init__.py`
- Create: `tests/test_self_rag.py`

**Step 1: Write the failing tests**

Create `tests/test_self_rag.py`:

```python
"""Unit tests for Self-RAG architecture."""

from unittest.mock import AsyncMock

import pytest

from src.architectures.agentic.self_rag import SelfRAG
from src.core.types import Document, Question, QuestionType, RetrievalResult


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = AsyncMock()
    llm.model = "test-model"
    return llm


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = AsyncMock()
    retriever.retrieve.return_value = RetrievalResult(
        documents=[
            Document(id="1", title="Doc A", text="Paris is the capital of France."),
            Document(id="2", title="Doc B", text="France is in Europe."),
            Document(id="3", title="Doc C", text="Berlin is the capital of Germany."),
        ],
        scores=[0.9, 0.85, 0.7],
        query="capital of France",
        retrieval_time_ms=10.0,
        method="bm25",
    )
    return retriever


@pytest.fixture
def sample_question():
    """Create a sample question for testing."""
    return Question(
        id="q1",
        text="What is the capital of France?",
        type=QuestionType.BRIDGE,
        gold_answer="Paris",
    )


@pytest.fixture
def sample_corpus():
    """Create a sample corpus for testing."""
    return [
        Document(id="d1", title="Doc 1", text="Paris is the capital of France."),
        Document(id="d2", title="Doc 2", text="Berlin is the capital of Germany."),
    ]


class TestSelfRAGWithRetrieval:
    """Test Self-RAG when retrieval is needed."""

    @pytest.mark.asyncio
    async def test_retrieval_needed_returns_best_candidate(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """When retrieval is needed, Self-RAG retrieves, generates candidates, and picks best."""
        mock_llm.generate.side_effect = [
            # Phase 1: retrieval decision
            ("yes", 5, 0.001),
            # Phase 3a: candidate 1 - generate answer with relevance+support
            (
                "The capital of France is Paris.\n[IsRel] relevant\n[IsSup] fully supported",
                20,
                0.002,
            ),
            # Phase 3b: candidate 1 - utility
            ("5", 3, 0.001),
            # Phase 3a: candidate 2 - generate answer with relevance+support
            (
                "France is located in Europe.\n[IsRel] relevant\n[IsSup] partially supported",
                20,
                0.002,
            ),
            # Phase 3b: candidate 2 - utility
            ("3", 3, 0.001),
        ]

        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 3, "num_candidates": 2},
        )
        response = await rag.answer(sample_question, sample_corpus)

        assert response.answer == "The capital of France is Paris."
        assert response.num_retrieval_calls == 1
        # 1 retrieval decision + 2 candidates × 2 calls each = 5
        assert response.num_llm_calls == 5
        assert response.architecture == "self_rag"

    @pytest.mark.asyncio
    async def test_tracks_cost_and_tokens(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Self-RAG should accumulate tokens and cost from all LLM calls."""
        mock_llm.generate.side_effect = [
            ("yes", 5, 0.001),
            ("Paris\n[IsRel] relevant\n[IsSup] fully supported", 20, 0.002),
            ("5", 3, 0.001),
        ]

        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 3, "num_candidates": 1},
        )
        response = await rag.answer(sample_question, sample_corpus)

        assert response.total_tokens == 5 + 20 + 3
        assert response.total_cost_usd == pytest.approx(0.001 + 0.002 + 0.001)

    @pytest.mark.asyncio
    async def test_reasoning_chain_records_all_phases(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Reasoning chain should have steps for decision, each candidate, and selection."""
        mock_llm.generate.side_effect = [
            ("yes", 5, 0.001),
            ("Paris\n[IsRel] relevant\n[IsSup] fully supported", 20, 0.002),
            ("5", 3, 0.001),
        ]

        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 3, "num_candidates": 1},
        )
        response = await rag.answer(sample_question, sample_corpus)

        # At least: retrieval decision step + candidate generation step
        assert len(response.reasoning_chain) >= 2
        # First step should be retrieval decision
        assert response.reasoning_chain[0].action == "retrieval_decision"

    @pytest.mark.asyncio
    async def test_retrieved_docs_populated(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """retrieved_docs should contain the retrieval result when retrieval occurs."""
        mock_llm.generate.side_effect = [
            ("yes", 5, 0.001),
            ("Paris\n[IsRel] relevant\n[IsSup] fully supported", 20, 0.002),
            ("5", 3, 0.001),
        ]

        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 3, "num_candidates": 1},
        )
        response = await rag.answer(sample_question, sample_corpus)

        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].query == "capital of France"


class TestSelfRAGWithoutRetrieval:
    """Test Self-RAG when no retrieval is needed."""

    @pytest.mark.asyncio
    async def test_no_retrieval_answers_directly(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """When retrieval is not needed, Self-RAG answers directly with no retriever call."""
        mock_llm.generate.side_effect = [
            # Phase 1: no retrieval needed
            ("no", 5, 0.001),
            # Direct answer
            ("Paris", 10, 0.001),
        ]

        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 3, "num_candidates": 2},
        )
        response = await rag.answer(sample_question, sample_corpus)

        assert response.answer == "Paris"
        assert response.num_retrieval_calls == 0
        assert response.num_llm_calls == 2
        mock_retriever.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_retrieval_reasoning_chain(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Reasoning chain for no-retrieval path should have decision + direct answer."""
        mock_llm.generate.side_effect = [
            ("no", 5, 0.001),
            ("Paris", 10, 0.001),
        ]

        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 3, "num_candidates": 2},
        )
        response = await rag.answer(sample_question, sample_corpus)

        assert len(response.reasoning_chain) == 2
        assert response.reasoning_chain[0].action == "retrieval_decision"
        assert response.reasoning_chain[1].action == "direct_answer"


class TestSelfRAGEdgeCases:
    """Test edge cases and fallback behavior."""

    @pytest.mark.asyncio
    async def test_no_supported_candidates_falls_back(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """If no candidates are well-supported, return the best available anyway."""
        mock_llm.generate.side_effect = [
            ("yes", 5, 0.001),
            ("Maybe Paris\n[IsRel] irrelevant\n[IsSup] no support", 20, 0.002),
            ("1", 3, 0.001),
        ]

        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 3, "num_candidates": 1},
        )
        response = await rag.answer(sample_question, sample_corpus)

        # Should still return an answer even if poorly supported
        assert response.answer == "Maybe Paris"

    @pytest.mark.asyncio
    async def test_malformed_llm_response_handles_gracefully(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """If the LLM doesn't produce parseable reflection tokens, use defaults."""
        mock_llm.generate.side_effect = [
            ("yes", 5, 0.001),
            # No reflection tokens at all
            ("Paris is the capital", 20, 0.002),
            # Non-numeric utility
            ("high", 3, 0.001),
        ]

        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 3, "num_candidates": 1},
        )
        response = await rag.answer(sample_question, sample_corpus)

        # Should still return the generation as the answer
        assert response.answer == "Paris is the capital"

    @pytest.mark.asyncio
    async def test_ambiguous_retrieval_decision_defaults_to_yes(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """If retrieval decision is ambiguous, default to retrieving."""
        mock_llm.generate.side_effect = [
            ("I think it might be helpful to look up some info", 10, 0.001),
            ("Paris\n[IsRel] relevant\n[IsSup] fully supported", 20, 0.002),
            ("5", 3, 0.001),
        ]

        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 3, "num_candidates": 1},
        )
        response = await rag.answer(sample_question, sample_corpus)

        # Should have retrieved since ambiguous defaults to yes
        assert response.num_retrieval_calls == 1


class TestSelfRAGMetadata:
    """Test metadata and configuration."""

    def test_get_name(self, mock_llm, mock_retriever):
        rag = SelfRAG(mock_llm, mock_retriever, {})
        assert rag.get_name() == "self_rag"

    def test_get_type(self, mock_llm, mock_retriever):
        from src.core.types import ArchitectureType

        rag = SelfRAG(mock_llm, mock_retriever, {})
        assert rag.get_type() == ArchitectureType.AGENTIC

    def test_default_config_values(self, mock_llm, mock_retriever):
        rag = SelfRAG(mock_llm, mock_retriever, {})
        assert rag.config["top_k"] == 5
        assert rag.config["num_candidates"] == 3
        assert rag.config["max_context_tokens"] == 4000

    def test_custom_config_values(self, mock_llm, mock_retriever):
        rag = SelfRAG(
            mock_llm,
            mock_retriever,
            {"top_k": 10, "num_candidates": 5},
        )
        assert rag.config["top_k"] == 10
        assert rag.config["num_candidates"] == 5
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_self_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.architectures.agentic.self_rag'`

**Step 3: Write minimal implementation**

Create `src/architectures/agentic/self_rag.py`:

```python
"""Self-RAG architecture implementation.

Self-RAG adds self-reflection tokens to decide whether to retrieve,
assess relevance of retrieved passages, critique generated answers
for support, and select the best candidate based on utility scoring.

Reference: Self-RAG: Learning to Retrieve, Generate, and Critique
(Asai et al., ICLR 2024) — https://arxiv.org/abs/2310.11511
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ...core.base_rag import BaseRAG
from ...core.llm_client import BaseLLMClient
from ...core.retriever import BaseRetriever
from ...core.types import (
    ArchitectureType,
    Document,
    Question,
    RAGResponse,
    ReasoningStep,
    RetrievalResult,
)


# Support level scores for candidate ranking
_SUPPORT_SCORES: dict[str, float] = {
    "fully supported": 1.0,
    "fully": 1.0,
    "partially supported": 0.5,
    "partially": 0.5,
    "no support": 0.0,
    "no": 0.0,
}


class SelfRAG(BaseRAG):
    """Self-RAG: self-reflective retrieval-augmented generation.

    Implements a 4-phase pipeline:
    1. Retrieval decision — should we retrieve?
    2. Retrieve + relevance filtering
    3. Generate candidates with support critique
    4. Utility scoring and best-candidate selection
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: BaseRetriever,
        config: dict,
    ):
        super().__init__(llm_client, retriever, config)

        prompt_path = self.config.get("prompt_path", "prompts/self_rag.txt")
        if Path(prompt_path).exists():
            self.retrieval_decision_prompt = self._load_prompt_template(prompt_path)
        else:
            self.retrieval_decision_prompt = (
                "You are a self-reflective question answering system.\n\n"
                "Given the question, decide if you need to retrieve external "
                "information or can answer directly.\n\n"
                "Question: {question}\n\n"
                "Do you need to retrieve information to answer this question "
                "accurately?\nAnswer with exactly \"yes\" or \"no\":\n"
            )

    def get_name(self) -> str:
        return "self_rag"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.AGENTIC

    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        return {
            "top_k": (int, False, 5),
            "num_candidates": (int, False, 3),
            "max_context_tokens": (int, False, 4000),
            "prompt_path": (str, False, "prompts/self_rag.txt"),
        }

    async def answer(
        self,
        question: Question,
        corpus: list[Document],
    ) -> RAGResponse:
        start_time = time.perf_counter()
        reasoning_chain: list[ReasoningStep] = []
        retrieved_docs: list[RetrievalResult] = []
        total_tokens = 0
        total_cost = 0.0
        num_llm_calls = 0
        num_retrieval_calls = 0
        step_id = 0

        # ── Phase 1: Retrieval decision ──────────────────────────
        step_id += 1
        retrieval_needed, tokens, cost = await self._decide_retrieval(question)
        total_tokens += tokens
        total_cost += cost
        num_llm_calls += 1

        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought=f"Decide whether retrieval is needed for: {question.text}",
                action="retrieval_decision",
                action_input=question.text,
                observation=f"Retrieval needed: {retrieval_needed}",
                tokens_used=tokens,
                cost_usd=cost,
            )
        )

        # ── No-retrieval path ────────────────────────────────────
        if not retrieval_needed:
            step_id += 1
            answer_text, tokens, cost = await self._direct_answer(question)
            total_tokens += tokens
            total_cost += cost
            num_llm_calls += 1

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought="No retrieval needed; answering directly.",
                    action="direct_answer",
                    action_input=question.text,
                    observation=f"Direct answer: {answer_text}",
                    tokens_used=tokens,
                    cost_usd=cost,
                )
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return RAGResponse(
                answer=answer_text,
                reasoning_chain=reasoning_chain,
                retrieved_docs=retrieved_docs,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                latency_ms=elapsed_ms,
                num_retrieval_calls=0,
                num_llm_calls=num_llm_calls,
                model=self.llm.model,
                architecture=self.get_name(),
            )

        # ── Phase 2: Retrieve passages ───────────────────────────
        step_id += 1
        retrieval_result = await self.retriever.retrieve(
            query=question.text,
            corpus=corpus,
            top_k=self.config["top_k"],
        )
        num_retrieval_calls += 1
        retrieved_docs.append(retrieval_result)

        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought="Retrieving passages for the question.",
                action="retrieve",
                action_input=question.text,
                observation=f"Retrieved {len(retrieval_result.documents)} passages.",
                tokens_used=0,
                cost_usd=0.0,
            )
        )

        # ── Phase 3: Generate candidates with critique ──────────
        num_candidates = min(
            self.config["num_candidates"],
            len(retrieval_result.documents),
        )
        candidates: list[dict[str, Any]] = []

        for i in range(num_candidates):
            passage = retrieval_result.documents[i]

            # 3a: Generate answer + assess relevance and support
            step_id += 1
            generation, relevance, support, gen_tokens, gen_cost = (
                await self._generate_and_critique(question, passage)
            )
            total_tokens += gen_tokens
            total_cost += gen_cost
            num_llm_calls += 1

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought=(
                        f"Generate answer from passage '{passage.title}' "
                        f"and assess relevance/support."
                    ),
                    action="generate_and_critique",
                    action_input=passage.text[:200],
                    observation=(
                        f"Generation: {generation[:100]}... | "
                        f"Relevance: {relevance} | Support: {support}"
                    ),
                    tokens_used=gen_tokens,
                    cost_usd=gen_cost,
                )
            )

            # 3b: Rate utility
            step_id += 1
            utility, util_tokens, util_cost = await self._rate_utility(
                question, generation
            )
            total_tokens += util_tokens
            total_cost += util_cost
            num_llm_calls += 1

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought=f"Rate overall utility of candidate {i + 1}.",
                    action="rate_utility",
                    action_input=generation[:200],
                    observation=f"Utility: {utility}/5",
                    tokens_used=util_tokens,
                    cost_usd=util_cost,
                )
            )

            candidates.append(
                {
                    "generation": generation,
                    "relevance": relevance,
                    "support": support,
                    "utility": utility,
                    "passage_title": passage.title,
                }
            )

        # ── Phase 4: Select best candidate ───────────────────────
        best = self._select_best_candidate(candidates)
        answer_text = best["generation"]

        step_id += 1
        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought="Select the best candidate based on utility and support.",
                action="select_best",
                action_input=f"{len(candidates)} candidates evaluated",
                observation=(
                    f"Selected candidate from '{best['passage_title']}' "
                    f"(utility={best['utility']}, support={best['support']})"
                ),
                tokens_used=0,
                cost_usd=0.0,
            )
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            answer=answer_text,
            reasoning_chain=reasoning_chain,
            retrieved_docs=retrieved_docs,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            latency_ms=elapsed_ms,
            num_retrieval_calls=num_retrieval_calls,
            num_llm_calls=num_llm_calls,
            model=self.llm.model,
            architecture=self.get_name(),
        )

    # ── Private helpers ──────────────────────────────────────────

    async def _decide_retrieval(
        self, question: Question
    ) -> tuple[bool, int, float]:
        """Phase 1: Decide whether retrieval is needed.

        Returns:
            (retrieval_needed, tokens_used, cost)
        """
        prompt = self.retrieval_decision_prompt.format(question=question.text)
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)

        normalized = response_text.strip().lower()
        # Default to yes (retrieve) unless explicitly "no"
        retrieval_needed = "no" not in normalized.split()[:3]

        return retrieval_needed, tokens, cost

    async def _direct_answer(
        self, question: Question
    ) -> tuple[str, int, float]:
        """Generate a direct answer without retrieval.

        Returns:
            (answer_text, tokens_used, cost)
        """
        prompt = (
            f"Answer the following question concisely and directly.\n\n"
            f"Question: {question.text}\n\n"
            f"Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)
        return response_text.strip(), tokens, cost

    async def _generate_and_critique(
        self, question: Question, passage: Document
    ) -> tuple[str, str, str, int, float]:
        """Generate an answer from a passage and assess relevance + support.

        Returns:
            (generation, relevance, support_level, tokens_used, cost)
        """
        prompt = (
            f"Answer the question based on the provided passage. "
            f"After your answer, assess the passage on two dimensions.\n\n"
            f"Passage ({passage.title}): {passage.text}\n\n"
            f"Question: {question.text}\n\n"
            f"Provide your answer, then on new lines:\n"
            f"[IsRel] relevant or irrelevant\n"
            f"[IsSup] fully supported, partially supported, or no support\n\n"
            f"Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)

        generation, relevance, support = self._parse_critique(response_text)
        return generation, relevance, support, tokens, cost

    async def _rate_utility(
        self, question: Question, generation: str
    ) -> tuple[int, int, float]:
        """Rate the utility of a generation on a 1-5 scale.

        Returns:
            (utility_score, tokens_used, cost)
        """
        prompt = (
            f"Rate how useful the following answer is for the question "
            f"on a scale of 1 to 5 (1=useless, 5=perfect).\n\n"
            f"Question: {question.text}\n"
            f"Answer: {generation}\n\n"
            f"Rating (1-5):"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)

        utility = self._parse_utility(response_text)
        return utility, tokens, cost

    def _parse_critique(
        self, response_text: str
    ) -> tuple[str, str, str]:
        """Parse generation, relevance, and support from LLM response.

        Returns:
            (generation_text, relevance, support_level)
        """
        lines = response_text.strip().split("\n")
        generation_lines: list[str] = []
        relevance = "relevant"  # default
        support = "partially supported"  # default

        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("[isrel]"):
                value = stripped.split("]", 1)[1].strip().lower()
                if "irrelevant" in value:
                    relevance = "irrelevant"
                else:
                    relevance = "relevant"
            elif stripped.lower().startswith("[issup]"):
                value = stripped.split("]", 1)[1].strip().lower()
                if "fully" in value:
                    support = "fully supported"
                elif "no" in value:
                    support = "no support"
                else:
                    support = "partially supported"
            else:
                generation_lines.append(stripped)

        generation = "\n".join(generation_lines).strip()
        if not generation:
            generation = response_text.strip()

        return generation, relevance, support

    def _parse_utility(self, response_text: str) -> int:
        """Parse a utility score (1-5) from LLM response.

        Returns:
            Integer utility score, defaults to 3 on parse failure.
        """
        match = re.search(r"[1-5]", response_text.strip())
        if match:
            return int(match.group(0))
        return 3  # default to middle score

    def _select_best_candidate(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Select the best candidate based on composite score.

        Score = utility + support_bonus
        - fully supported: +2.0
        - partially supported: +1.0
        - no support: +0.0
        - relevant passages get +0.5 bonus

        Returns:
            The best candidate dict.
        """
        if not candidates:
            return {
                "generation": "No answer.",
                "relevance": "irrelevant",
                "support": "no support",
                "utility": 0,
                "passage_title": "",
            }

        def _score(candidate: dict[str, Any]) -> float:
            utility = candidate["utility"]
            support_bonus = _SUPPORT_SCORES.get(
                candidate["support"], 0.5
            )
            relevance_bonus = 0.5 if candidate["relevance"] == "relevant" else 0.0
            return utility + (support_bonus * 2.0) + relevance_bonus

        return max(candidates, key=_score)
```

Update `src/architectures/agentic/__init__.py`:

```python
"""Agentic RAG implementations: ReAct, Self-RAG, Planner."""

from .react_rag import ReActRAG
from .self_rag import SelfRAG

__all__ = ["ReActRAG", "SelfRAG"]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_self_rag.py -v`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add src/architectures/agentic/self_rag.py src/architectures/agentic/__init__.py tests/test_self_rag.py
git commit -m "feat: add self-rag architecture with reflection tokens"
```

---

### Task 3: Wire Self-RAG into factory and runner

**Files:**
- Modify: `src/architectures/factory.py`
- Modify: `scripts/run_experiment.py`
- Create: `tests/test_self_rag_config.py`

**Step 1: Write the failing test**

Create `tests/test_self_rag_config.py`:

```python
"""Tests for Self-RAG factory and config integration."""

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_self_rag_config.py -v`
Expected: FAIL with `ValueError: Unknown architecture: self_rag`

**Step 3: Write minimal implementation**

Edit `src/architectures/factory.py` — add the `self_rag` case:

```python
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
```

Edit `scripts/run_experiment.py` — add the `self_rag` branch inside `_build_rag()`. After the existing `elif name == "react_rag":` block (around line 67-70), add:

```python
    elif architecture_name == "self_rag":
        architecture_config = {
            **common_config,
            **config.get("self_rag", {}),
        }
```

This goes between the `react_rag` block and the `else` block. The full updated `_build_rag` function should look like:

```python
def _build_rag(config: dict[str, Any]):
    cache = None
    if config.get("cache", {}).get("enabled", False):
        cache_path = config.get("cache", {}).get("path", ".cache/llm_cache.db")
        cache = SQLiteCache(cache_path)

    llm_config = config.get("llm", {})
    llm = create_llm_client(
        provider=llm_config.get("provider", "openai"),
        model=llm_config.get("model"),
        cache=cache,
    )

    retrieval_config = config.get("retrieval", {})
    retriever = create_retriever(
        method=retrieval_config.get("method", "bm25"),
        bm25_weight=retrieval_config.get("bm25_weight", 0.5),
        dense_weight=retrieval_config.get("dense_weight", 0.5),
        embedding_model=retrieval_config.get("embedding_model", "text-embedding-3-small"),
    )

    architecture_name = config.get("architecture", {}).get("name", "vanilla_rag")
    common_config = {
        "top_k": retrieval_config.get("top_k", 5),
        "max_context_tokens": llm_config.get("max_tokens", 1024),
    }

    if architecture_name == "vanilla_rag":
        architecture_config = {
            **common_config,
            "prompt_path": config.get("prompt_path", "prompts/vanilla.txt"),
            **config.get("vanilla", {}),
        }
    elif architecture_name == "react_rag":
        architecture_config = {
            **common_config,
            **config.get("react", {}),
        }
    elif architecture_name == "self_rag":
        architecture_config = {
            **common_config,
            **config.get("self_rag", {}),
        }
    else:
        architecture_config = {**common_config, **config.get(architecture_name, {})}

    return create_architecture(architecture_name, llm, retriever, architecture_config)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_self_rag_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/architectures/factory.py scripts/run_experiment.py tests/test_self_rag_config.py
git commit -m "feat: wire self-rag into architecture factory and experiment runner"
```

---

### Task 4: Add Self-RAG config variants

**Files:**
- Create: `configs/self_rag.yaml`
- Create: `configs/self_rag_dense.yaml`
- Create: `configs/self_rag_hybrid.yaml`
- Create: `configs/self_rag_bm25_full.yaml`
- Create: `configs/self_rag_dense_full.yaml`
- Create: `configs/self_rag_hybrid_full.yaml`

**Step 1: Write the failing test**

Add to `tests/test_self_rag_config.py`:

```python
from pathlib import Path

from src.utils.config import load_config


def test_self_rag_yaml_config_loads():
    """self_rag.yaml should load and have correct architecture name."""
    config = load_config(Path("configs/self_rag.yaml"))
    assert config["architecture"]["name"] == "self_rag"
    assert config["self_rag"]["num_candidates"] == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_self_rag_config.py::test_self_rag_yaml_config_loads -v`
Expected: FAIL with `FileNotFoundError`

**Step 3: Write minimal implementation**

Create `configs/self_rag.yaml`:

```yaml
inherits: base.yaml

experiment:
  name: "self_rag"

architecture:
  name: "self_rag"

self_rag:
  num_candidates: 3

retrieval:
  method: "bm25"

data:
  subset_size: 100
```

Create `configs/self_rag_dense.yaml`:

```yaml
inherits: base.yaml

experiment:
  name: "self_rag_dense"

architecture:
  name: "self_rag"

self_rag:
  num_candidates: 3

retrieval:
  method: "dense"
  embedding_model: "text-embedding-3-small"

data:
  subset_size: 100
```

Create `configs/self_rag_hybrid.yaml`:

```yaml
inherits: base.yaml

experiment:
  name: "self_rag_hybrid"

architecture:
  name: "self_rag"

self_rag:
  num_candidates: 3

retrieval:
  method: "hybrid"

data:
  subset_size: 100
```

Create `configs/self_rag_bm25_full.yaml`:

```yaml
inherits: base.yaml

experiment:
  name: "self_rag_bm25_full"

architecture:
  name: "self_rag"

self_rag:
  num_candidates: 3

retrieval:
  method: "bm25"

data:
  subset_size: null  # Full validation set
```

Create `configs/self_rag_dense_full.yaml`:

```yaml
inherits: base.yaml

experiment:
  name: "self_rag_dense_full"

architecture:
  name: "self_rag"

self_rag:
  num_candidates: 3

retrieval:
  method: "dense"
  embedding_model: "text-embedding-3-small"

data:
  subset_size: null  # Full validation set
```

Create `configs/self_rag_hybrid_full.yaml`:

```yaml
inherits: base.yaml

experiment:
  name: "self_rag_hybrid_full"

architecture:
  name: "self_rag"

self_rag:
  num_candidates: 3

retrieval:
  method: "hybrid"

data:
  subset_size: null  # Full validation set
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_self_rag_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add configs/self_rag*.yaml tests/test_self_rag_config.py
git commit -m "feat: add self-rag config variants for all retrievers"
```

---

### Task 5: Regression test and lint

**Files:**
- Test: all `tests/`

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All existing tests + new Self-RAG tests PASS

**Step 2: Run linter**

Run: `ruff check src/architectures/agentic/self_rag.py tests/test_self_rag.py tests/test_self_rag_prompt.py tests/test_self_rag_config.py`
Expected: No errors

**Step 3: Run formatter check**

Run: `black --check src/architectures/agentic/self_rag.py tests/test_self_rag.py tests/test_self_rag_prompt.py tests/test_self_rag_config.py`
Expected: All files formatted correctly (or fix with `black` first)

**Step 4: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: lint and format self-rag files"
```

---

### Task 6: Update README with Self-RAG rows

**Files:**
- Modify: `README.md`

**Step 1: Add Self-RAG section to results table**

In the README results table, add rows for Self-RAG (BM25, Dense, Hybrid) with "pending" values — same pattern as ReAct.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add self-rag rows to results table (pending)"
```

---

### Task 7: Run Self-RAG evaluation (subset first)

**Files:**
- No code changes; execution only

**Step 1: Run subset experiment (100 questions, BM25)**

Run: `python scripts/run_experiment.py --config configs/self_rag.yaml`
Expected: Completes without errors, produces `results/<run_id>/summary.json`

**Step 2: Check results**

Run: `python scripts/analyze_results.py --results results/<latest_run_id> --breakdown`
Expected: Summary with EM, F1, cost, latency metrics displayed

**Step 3: Run Dense and Hybrid subsets**

Run: `python scripts/run_experiment.py --config configs/self_rag_dense.yaml`
Run: `python scripts/run_experiment.py --config configs/self_rag_hybrid.yaml`

**Step 4: Update README with subset results**

Fill in the actual metrics from the subset runs.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add self-rag subset results"
```

---

## Cost Estimate

Self-RAG uses more LLM calls per question than ReAct:
- **Per question:** 1 (decision) + 2 × 3 (candidates) = 7 LLM calls
- **gpt-4o-mini at ~$0.002/question × 7 calls ≈ $0.014/question**
- **100-question subset:** ~$1.40
- **Full 7,405 questions:** ~$104 per retriever variant

Compare to:
- Vanilla RAG: 1 LLM call/question ≈ $0.002/question
- ReAct RAG: 2-7 LLM calls/question ≈ $0.004-0.014/question

**Recommendation:** Start with 100-question subsets. Only run full benchmarks after validating subset performance is reasonable.

---

## Design Decisions & Rationale

1. **Combined generation + critique prompt:** Instead of separate LLM calls for relevance, generation, and support (4 calls per passage), we combine into 2 calls per candidate. This saves ~33% on API costs while maintaining assessment quality.

2. **Default retrieval on ambiguous decision:** Multi-hop QA almost always benefits from retrieval. Defaulting to "yes" on ambiguous decisions prevents false negatives.

3. **num_candidates instead of per-passage filtering:** The spec suggests filtering passages for relevance first, then generating from each relevant passage. Our implementation generates candidates directly from top-K passages (up to `num_candidates`), which is simpler and avoids the case where aggressive filtering leaves no candidates.

4. **Composite scoring for selection:** `utility + support_bonus * 2.0 + relevance_bonus` weights support heavily because well-supported answers are more likely correct for multi-hop QA.

5. **No separate relevance filter phase:** The original Self-RAG paper uses fine-tuned special tokens. Since we're using API-only models with prompting, relevance assessment is folded into the generation prompt to keep things practical.
