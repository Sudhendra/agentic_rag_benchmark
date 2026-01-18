# AGENTS.md - AI Coding Assistant Guide

This document provides context for AI coding assistants (Cursor, Windsurf, Aider, GitHub Copilot, etc.) working on the Agentic RAG Benchmark project.

## Project Overview

**Goal:** Build a benchmarking framework comparing three paradigms for multi-hop question answering:
1. **Agentic RAG** - LLM agents with iterative retrieval (ReAct, Self-RAG, Planner)
2. **Recursive RAG** - Interleaved retrieval with reasoning (IRCoT, REAP)
3. **Recursive Language Models (RLM)** - Programmatic self-recursion

**Target:** ACL/NAACL 2026 publication
**Constraint:** API-only compute (OpenAI, Anthropic)

## Key Documents

- `docs/TECHNICAL_SPEC.md` - Complete technical specification with all interfaces
- `README.md` - Project overview
- `pyproject.toml` - Dependencies (when created)

## Architecture Overview

```
src/
├── core/           # Base abstractions (types, LLM client, retriever)
├── architectures/  # RAG implementations (6 architectures)
├── retrieval/      # BM25, Dense, Hybrid retrievers
├── data/           # Dataset loaders (HotpotQA, MuSiQue, 2WikiMultiHop)
├── evaluation/     # Metrics and evaluation runner
└── utils/          # Cache, cost tracking, logging
```

## Coding Standards

### Python Version
- Python 3.11+
- Use type hints everywhere
- Use `async/await` for all I/O operations

### Style
- Follow PEP 8
- Use Black formatter (line length 100)
- Use Ruff for linting
- Docstrings in Google style

### Imports
```python
# Standard library
from dataclasses import dataclass, field
from typing import Optional, Literal
from abc import ABC, abstractmethod
import asyncio

# Third-party
import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Local
from ..core.types import Question, Document, RAGResponse
from ..core.base_rag import BaseRAG
```

## Key Interfaces

### BaseRAG (all architectures inherit from this)
```python
class BaseRAG(ABC):
    def __init__(self, llm_client: BaseLLMClient, retriever: BaseRetriever, config: dict):
        ...
    
    @abstractmethod
    async def answer(self, question: Question, corpus: list[Document]) -> RAGResponse:
        """Main entry point - must be implemented by all architectures."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return architecture name like 'react_rag'."""
        pass
    
    @abstractmethod
    def get_type(self) -> ArchitectureType:
        """Return category: VANILLA, AGENTIC, RECURSIVE, or RLM."""
        pass
```

### BaseLLMClient
```python
class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None
    ) -> tuple[str, int, float]:
        """Returns (response_text, tokens_used, cost_usd)."""
        pass
```

### BaseRetriever
```python
class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        corpus: list[Document],
        top_k: int = 5
    ) -> RetrievalResult:
        pass
    
    @abstractmethod
    async def index(self, corpus: list[Document]) -> None:
        pass
```

## Core Data Types

```python
@dataclass
class Question:
    id: str
    text: str
    type: QuestionType  # BRIDGE, COMPARISON, COMPOSITIONAL, SINGLE_HOP
    gold_answer: Optional[str] = None
    supporting_facts: Optional[list[tuple[str, int]]] = None

@dataclass
class Document:
    id: str
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

@dataclass
class RAGResponse:
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
```

## Implementation Order

When implementing, follow this order to satisfy dependencies:

### Phase 1: Core (implement first)
1. `src/core/types.py` - All dataclasses
2. `src/utils/cache.py` - SQLite cache for LLM responses
3. `src/core/llm_client.py` - OpenAI + Anthropic clients
4. `src/core/retriever.py` - Base retriever interface
5. `src/retrieval/bm25.py` - BM25 implementation
6. `src/retrieval/dense.py` - Dense with OpenAI embeddings
7. `src/retrieval/hybrid.py` - Hybrid retriever
8. `src/core/base_rag.py` - Abstract base class

### Phase 2: Architectures
9. `src/architectures/vanilla_rag.py` - Baseline
10. `src/architectures/agentic/react_rag.py` - ReAct agent
11. `src/architectures/agentic/self_rag.py` - Self-reflection
12. `src/architectures/agentic/planner_rag.py` - Question decomposition
13. `src/architectures/recursive/ircot.py` - Interleaved CoT
14. `src/architectures/recursive/reap.py` - Recursive planning
15. `src/architectures/rlm/recursive_lm.py` - Recursive LM

### Phase 3: Data & Evaluation
16. `src/data/base_loader.py` - Abstract loader
17. `src/data/hotpotqa.py` - HotpotQA
18. `src/evaluation/metrics.py` - EM, F1
19. `src/evaluation/evaluator.py` - Runner

### Phase 4: Scripts & Config
20. `configs/base.yaml` + architecture configs
21. `scripts/run_experiment.py`
22. `tests/` - Unit tests

## Architecture Implementation Patterns

### Vanilla RAG (simplest)
```python
class VanillaRAG(BaseRAG):
    async def answer(self, question, corpus):
        # 1. Retrieve
        result = await self.retriever.retrieve(question.text, corpus, top_k)
        # 2. Build context
        context = self._build_context(result.documents)
        # 3. Generate
        response, tokens, cost = await self.llm.generate(prompt)
        # 4. Return RAGResponse
        return RAGResponse(...)
```

### ReAct (iterative)
```python
class ReActRAG(BaseRAG):
    async def answer(self, question, corpus):
        scratchpad = []
        for _ in range(max_iterations):
            prompt = self._build_react_prompt(question, scratchpad)
            response, tokens, cost = await self.llm.generate(prompt, stop=["Observation:"])
            thought, action, action_input = self._parse(response)
            
            if action == "finish":
                return RAGResponse(answer=action_input, ...)
            
            if action == "search":
                result = await self.retriever.retrieve(action_input, corpus)
                observation = self._format_results(result)
            
            scratchpad.append((thought, action, action_input, observation))
```

### IRCoT (recursive with retrieval)
```python
class IRCoTRAG(BaseRAG):
    async def answer(self, question, corpus):
        cot_steps = []
        for _ in range(max_steps):
            prompt = self._build_ircot_prompt(question, cot_steps)
            step, tokens, cost = await self.llm.generate(prompt)
            
            if "[ANSWER]" in step:
                return RAGResponse(answer=self._extract_answer(step), ...)
            
            # Retrieve based on CoT step
            query = self._extract_query(step)
            result = await self.retriever.retrieve(query, corpus)
            cot_steps.append({"thought": step, "retrieval": result})
```

### RLM (recursive decomposition)
```python
class RecursiveLM(BaseRAG):
    async def answer(self, question, corpus, depth=0, memo=None):
        if memo is None:
            memo = {}
        
        cache_key = hash(question.text)
        if cache_key in memo:
            return memo[cache_key]
        
        if depth > self.config["max_depth"]:
            return await self._direct_answer(question, corpus)
        
        # Generate decomposition decision
        program = await self._generate_program(question)
        
        if program.is_direct:
            answer = program.answer
        else:
            # Recursively solve sub-questions
            sub_answers = []
            for sq in program.sub_questions:
                sub_q = Question(id=f"{question.id}_sub", text=sq, ...)
                sub_answer = await self.answer(sub_q, corpus, depth+1, memo)
                sub_answers.append(sub_answer)
            # Combine
            answer = await self._combine(question, sub_answers, program.combine_instruction)
        
        memo[cache_key] = answer
        return RAGResponse(answer=answer, ...)
```

## Testing Patterns

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=BaseLLMClient)
    llm.generate.return_value = ("Test answer", 100, 0.001)
    llm.model = "test-model"
    return llm

@pytest.fixture
def mock_retriever():
    retriever = AsyncMock(spec=BaseRetriever)
    retriever.retrieve.return_value = RetrievalResult(
        documents=[Document(id="1", title="Test", text="Test content")],
        scores=[0.9],
        query="test",
        retrieval_time_ms=10,
        method="bm25"
    )
    return retriever

@pytest.mark.asyncio
async def test_vanilla_rag(mock_llm, mock_retriever):
    rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 5})
    question = Question(id="1", text="Test?", type=QuestionType.SINGLE_HOP)
    
    response = await rag.answer(question, [])
    
    assert response.answer == "Test answer"
    mock_retriever.retrieve.assert_called_once()
    mock_llm.generate.assert_called_once()
```

## Common Tasks

### Adding a New Architecture
1. Create file in `src/architectures/<category>/<name>.py`
2. Inherit from `BaseRAG`
3. Implement `answer()`, `get_name()`, `get_type()`, `get_config_schema()`
4. Add to `scripts/run_experiment.py` ARCHITECTURES dict
5. Create config in `configs/<name>.yaml`
6. Add tests in `tests/test_architectures.py`

### Adding a New Dataset
1. Create file in `src/data/<name>.py`
2. Inherit from base loader pattern
3. Return `tuple[list[Question], list[Document]]`
4. Handle subset_size for development
5. Add to data loading in run script

### Adding a New Metric
1. Add function to `src/evaluation/metrics.py`
2. Update `EvaluationResult` dataclass if needed
3. Integrate into `Evaluator.evaluate()`
4. Update `BenchmarkResult` aggregation

## API Cost Awareness

**IMPORTANT:** This project uses paid APIs. Always:

1. **Use caching** - SQLiteCache for all LLM calls
2. **Use cheap models for dev** - gpt-4o-mini, claude-3-haiku
3. **Test on subsets** - Use `subset_size: 100` during development
4. **Track costs** - All LLM clients track `total_cost`

Cost estimates per question:
- gpt-4o-mini: ~$0.002
- gpt-4o: ~$0.03
- claude-3.5-sonnet: ~$0.04

## Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
)
async def generate(self, ...):
    ...
```

## File Naming Conventions

- Snake_case for all Python files: `react_rag.py`, `llm_client.py`
- Lowercase with underscores for modules: `src/core/`, `src/architectures/`
- UPPERCASE for markdown: `README.md`, `AGENTS.md`, `TECHNICAL_SPEC.md`
- Lowercase for configs: `configs/react.yaml`

## Git Workflow

- Main branch: `main`
- Feature branches: `feature/<name>`
- Commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`

## Environment Variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
ruff check src/ tests/

# Run experiment
python scripts/run_experiment.py --config configs/react.yaml

# Run with subset for development
python scripts/run_experiment.py --config configs/react.yaml --subset 100
```

## Questions for Human

When implementing, ask about:
1. Specific prompt wording for architectures
2. Edge case handling preferences
3. Metric calculation details
4. Config parameter values

## References

- Technical Spec: `docs/TECHNICAL_SPEC.md`
- ReAct Paper: https://arxiv.org/abs/2210.03629
- Self-RAG Paper: https://arxiv.org/abs/2310.11511
- IRCoT Paper: https://arxiv.org/abs/2212.10509
- REAP Paper: https://arxiv.org/abs/2511.09966
- RLM Paper: https://arxiv.org/abs/2512.24601
- HotpotQA: https://hotpotqa.github.io/
