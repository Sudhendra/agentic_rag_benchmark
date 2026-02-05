# Next Steps: Completing Vanilla RAG

**Date:** February 6, 2026  
**Status:** Phase 1 Complete - Full Validation Results Available  
**Author:** Research Team

---

## Current Status Summary

### What's Done

| Component | File | Status |
|-----------|------|--------|
| Core Types | `src/core/types.py` | ✅ Complete |
| LLM Client (OpenAI) | `src/core/llm_client.py` | ✅ Complete |
| Base RAG | `src/core/base_rag.py` | ✅ Complete |
| Base Retriever | `src/core/retriever.py` | ✅ Complete |
| BM25 Retriever | `src/retrieval/bm25.py` | ✅ Complete |
| Dense Retriever | `src/retrieval/dense.py` | ✅ Complete |
| Hybrid Retriever | `src/retrieval/hybrid.py` | ✅ Complete |
| Vanilla RAG | `src/architectures/vanilla_rag.py` | ✅ Complete |
| HotpotQA Loader | `src/data/hotpotqa.py` | ✅ Complete |
| Metrics (EM, F1) | `src/evaluation/metrics.py` | ✅ Complete |
| Evaluator | `src/evaluation/evaluator.py` | ✅ Complete |
| SQLite Cache | `src/utils/cache.py` | ✅ Complete |
| Config Loader | `src/utils/config.py` | ✅ Complete |
| Results Saver | `src/utils/results.py` | ✅ Complete |
| Logging | `src/utils/logging.py` | ✅ Complete |
| Experiment Runner | `scripts/run_experiment.py` | ✅ Complete |
| MLflow Integration | `scripts/run_experiment.py` | ✅ Complete |
| Configs | `configs/base.yaml`, `vanilla.yaml`, `vanilla_dense.yaml`, `vanilla_hybrid.yaml`, `*_full.yaml` | ✅ Complete |
| Unit Tests | `tests/test_*.py` (97 tests) | ✅ Complete |
| Analysis Script | `scripts/analyze_results.py` | ✅ Complete |

### Benchmark Results - Full Validation Set (7,405 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost |
|-----------|-------------|----------|--------------|------|
| **Dense** | **45.0%** | **59.5%** | 1411 | $0.79 |
| Hybrid    | 44.1%       | 58.6%    | 2127         | $0.76 |
| BM25      | 38.2%       | 51.5%    | 2321         | $0.77 |

### Comparison: 100-sample vs Full Validation

| Retriever | EM (100) | EM (7405) | F1 (100) | F1 (7405) |
|-----------|----------|-----------|----------|-----------|
| Dense     | 55.0%    | 45.0%     | 69.0%    | 59.5%     |
| Hybrid    | 49.0%    | 44.1%     | 61.0%    | 58.6%     |
| BM25      | 46.0%    | 38.2%     | 57.2%    | 51.5%     |

**Key Findings:**
- Dense retrieval outperforms BM25 (+6.8% EM, +8.0% F1 on full set)
- Hybrid performs between Dense and BM25 (RRF fusion provides marginal benefit over BM25)
- 100-sample results were optimistic; full set shows realistic performance
- Total cost for all three runs: ~$2.32

**Breakdown by Question Type (Dense, Full Set):**
| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 39.6% | 55.4% |
| Comparison | 1,487 | 66.3% | 75.9% |

**Observation:** Comparison questions are significantly easier than bridge questions (+27% EM).

**Recommendation:** Use Dense retrieval as the default for Vanilla RAG baseline.

---

## Remaining Tasks for Vanilla RAG Completion

### Priority 1: Anthropic Client Implementation (High Impact)

**File:** `src/core/llm_client.py`

Add the AnthropicClient class after OpenAIClient:

```python
import anthropic

class AnthropicClient(BaseLLMClient):
    """Anthropic API client with caching and cost tracking."""

    # Pricing per 1M tokens (as of Jan 2026)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        cache: Optional[SQLiteCache] = None,
        track_costs: bool = True,
        api_key: Optional[str] = None,
    ):
        super().__init__(model, cache, track_costs)
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(self.model, self.PRICING["claude-3-5-sonnet-20241022"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError)),
    )
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
    ) -> tuple[str, int, float]:
        # Check cache first
        cache_key = self._make_cache_key(messages, temperature, max_tokens, stop)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return tuple(cached)

        # Convert OpenAI message format to Anthropic format
        system_msg = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append(msg)

        # Build API call kwargs
        kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if stop:
            kwargs["stop_sequences"] = stop

        response = await self.client.messages.create(**kwargs)

        # Extract response and metrics
        content = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)

        # Update stats
        if self.track_costs:
            self.total_tokens += total_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.call_count += 1

        result = (content, total_tokens, cost)

        # Cache the result
        if self.cache:
            self.cache.set(cache_key, list(result))

        return result
```

**Update `create_llm_client()` factory:**

```python
def create_llm_client(
    provider: str = "openai",
    model: Optional[str] = None,
    cache: Optional[SQLiteCache] = None,
    **kwargs,
) -> BaseLLMClient:
    if provider == "openai":
        model = model or "gpt-4o-mini"
        return OpenAIClient(model=model, cache=cache, **kwargs)
    elif provider == "anthropic":
        model = model or "claude-3-5-sonnet-20241022"
        return AnthropicClient(model=model, cache=cache, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

**Add to pyproject.toml dependencies:**
```toml
"anthropic>=0.18",
```

---

### Priority 2: Test with Different Retrievers

**Task:** Run vanilla RAG with hybrid retriever to compare performance.

**Create `configs/vanilla_hybrid.yaml`:**
```yaml
inherits: base.yaml

experiment:
  name: "vanilla_hybrid"

retrieval:
  method: "hybrid"
  bm25_weight: 0.5
  dense_weight: 0.5

data:
  subset_size: 100
```

**Create `configs/vanilla_dense.yaml`:**
```yaml
inherits: base.yaml

experiment:
  name: "vanilla_dense"

retrieval:
  method: "dense"

data:
  subset_size: 100
```

**Run comparisons:**
```bash
# BM25 (already done)
python scripts/run_experiment.py --config configs/vanilla.yaml --subset 100

# Dense
python scripts/run_experiment.py --config configs/vanilla_dense.yaml --subset 100

# Hybrid
python scripts/run_experiment.py --config configs/vanilla_hybrid.yaml --subset 100
```

---

### Priority 3: Unit Tests

**File:** `tests/test_vanilla_rag.py`

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.architectures.vanilla_rag import VanillaRAG
from src.core.types import Document, Question, QuestionType, RetrievalResult


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.model = "test-model"
    llm.generate.return_value = ("Test answer", 100, 0.001)
    return llm


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock()
    retriever.retrieve.return_value = RetrievalResult(
        documents=[Document(id="1", title="Test Doc", text="Test content.")],
        scores=[0.9],
        query="test query",
        retrieval_time_ms=10.0,
        method="bm25",
    )
    return retriever


@pytest.fixture
def sample_question():
    return Question(
        id="q1",
        text="What is the test?",
        type=QuestionType.SINGLE_HOP,
        gold_answer="Test answer",
    )


@pytest.fixture
def sample_corpus():
    return [
        Document(id="d1", title="Doc 1", text="This is document one."),
        Document(id="d2", title="Doc 2", text="This is document two."),
    ]


@pytest.mark.asyncio
async def test_vanilla_rag_returns_answer(mock_llm, mock_retriever, sample_question, sample_corpus):
    rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 5, "max_context_tokens": 4000})
    
    response = await rag.answer(sample_question, sample_corpus)
    
    assert response.answer == "Test answer"
    assert response.num_retrieval_calls == 1
    assert response.num_llm_calls == 1
    assert response.architecture == "vanilla_rag"


@pytest.mark.asyncio
async def test_vanilla_rag_calls_retriever(mock_llm, mock_retriever, sample_question, sample_corpus):
    rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 3})
    
    await rag.answer(sample_question, sample_corpus)
    
    mock_retriever.retrieve.assert_called_once()
    call_args = mock_retriever.retrieve.call_args
    assert call_args.kwargs["query"] == sample_question.text
    assert call_args.kwargs["top_k"] == 3


@pytest.mark.asyncio
async def test_vanilla_rag_tracks_cost(mock_llm, mock_retriever, sample_question, sample_corpus):
    rag = VanillaRAG(mock_llm, mock_retriever, {})
    
    response = await rag.answer(sample_question, sample_corpus)
    
    assert response.total_cost_usd == 0.001
    assert response.total_tokens == 100


def test_vanilla_rag_get_name(mock_llm, mock_retriever):
    rag = VanillaRAG(mock_llm, mock_retriever, {})
    assert rag.get_name() == "vanilla_rag"


def test_vanilla_rag_get_type(mock_llm, mock_retriever):
    from src.core.types import ArchitectureType
    rag = VanillaRAG(mock_llm, mock_retriever, {})
    assert rag.get_type() == ArchitectureType.VANILLA
```

**File:** `tests/test_metrics.py`

```python
import pytest
from src.evaluation.metrics import normalize_answer, exact_match, f1_score, supporting_fact_metrics


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("HELLO") == "hello"

    def test_remove_articles(self):
        assert normalize_answer("the quick brown fox") == "quick brown fox"
        assert normalize_answer("a cat and an apple") == "cat and apple"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_whitespace_normalization(self):
        assert normalize_answer("hello   world") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert exact_match("PARIS", "paris") == 1.0

    def test_with_articles(self):
        assert exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0

    def test_different(self):
        assert exact_match("Paris", "London") == 0.0


class TestF1Score:
    def test_identical(self):
        assert f1_score("quick brown fox", "quick brown fox") == 1.0

    def test_partial_overlap(self):
        score = f1_score("quick brown fox", "quick red fox")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert f1_score("hello", "world") == 0.0

    def test_empty_strings(self):
        assert f1_score("", "") == 1.0


class TestSupportingFactMetrics:
    def test_exact_match(self):
        pred = [("Doc1", 0), ("Doc2", 1)]
        gold = [("Doc1", 0), ("Doc2", 1)]
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 1.0
        assert f1 == 1.0

    def test_partial_match(self):
        pred = [("Doc1", 0), ("Doc3", 2)]
        gold = [("Doc1", 0), ("Doc2", 1)]
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 0.0
        assert 0.0 < f1 < 1.0

    def test_no_match(self):
        pred = [("Doc3", 2)]
        gold = [("Doc1", 0)]
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 0.0
        assert f1 == 0.0
```

**File:** `tests/test_llm_client.py`

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.llm_client import OpenAIClient
from src.utils.cache import SQLiteCache


@pytest.fixture
def mock_cache():
    cache = MagicMock(spec=SQLiteCache)
    cache.get.return_value = None
    return cache


@pytest.mark.asyncio
async def test_openai_client_caches_response(mock_cache):
    with patch("src.core.llm_client.openai.AsyncOpenAI") as mock_openai:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="cached response"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(model="gpt-4o-mini", cache=mock_cache, api_key="test-key")
        
        # First call - should hit API
        result1 = await client.generate([{"role": "user", "content": "test"}])
        
        # Verify cache was checked and set
        mock_cache.get.assert_called()
        mock_cache.set.assert_called()


def test_openai_client_calculates_cost():
    with patch("src.core.llm_client.openai.AsyncOpenAI"):
        client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
        
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = client._calculate_cost(input_tokens=1000, output_tokens=500)
        
        expected = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
        assert abs(cost - expected) < 0.0001


def test_openai_client_tracks_stats():
    with patch("src.core.llm_client.openai.AsyncOpenAI"):
        client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
        
        assert client.total_tokens == 0
        assert client.call_count == 0
        
        stats = client.get_stats()
        assert stats["model"] == "gpt-4o-mini"
        assert stats["call_count"] == 0
```

**File:** `tests/test_hotpotqa.py`

```python
import pytest
from unittest.mock import patch, MagicMock

from src.data.hotpotqa import HotpotQALoader, load_hotpotqa
from src.core.types import QuestionType


@pytest.fixture
def mock_dataset():
    return [
        {
            "id": "q1",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {
                "title": ["France", "Paris"],
                "sent_id": [0, 1],
            },
            "context": {
                "title": ["France", "Germany"],
                "sentences": [
                    ["France is a country.", "Its capital is Paris."],
                    ["Germany is a country.", "Its capital is Berlin."],
                ],
            },
        },
    ]


def test_loader_parses_questions(mock_dataset):
    with patch("src.data.hotpotqa.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        
        loader = HotpotQALoader(setting="distractor", split="validation")
        questions, corpus = loader.load()
        
        assert len(questions) == 1
        assert questions[0].id == "q1"
        assert questions[0].text == "What is the capital of France?"
        assert questions[0].gold_answer == "Paris"
        assert questions[0].type == QuestionType.BRIDGE


def test_loader_parses_corpus(mock_dataset):
    with patch("src.data.hotpotqa.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        
        loader = HotpotQALoader(setting="distractor", split="validation")
        questions, corpus = loader.load()
        
        assert len(corpus) == 2  # Two documents from context
        assert any(doc.title == "France" for doc in corpus)
        assert any(doc.title == "Germany" for doc in corpus)


def test_loader_parses_supporting_facts(mock_dataset):
    with patch("src.data.hotpotqa.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        
        loader = HotpotQALoader(setting="distractor", split="validation")
        questions, corpus = loader.load()
        
        assert questions[0].supporting_facts == [("France", 0), ("Paris", 1)]


def test_loader_respects_subset_size(mock_dataset):
    with patch("src.data.hotpotqa.load_dataset") as mock_load:
        # Create a mock that supports .select()
        mock_ds = MagicMock()
        mock_ds.__iter__ = lambda self: iter(mock_dataset)
        mock_ds.__len__ = lambda self: len(mock_dataset)
        mock_ds.select.return_value = mock_dataset[:1]
        mock_load.return_value = mock_ds
        
        loader = HotpotQALoader(subset_size=1)
        mock_ds.select.assert_not_called()  # Not called until load()
```

---

### Priority 4: Supporting Fact Evaluation

**Update `src/evaluation/evaluator.py`** to compute supporting fact metrics when enabled.

The current implementation sets `supporting_fact_em` and `supporting_fact_f1` to `None`. To enable:

1. Add a `compute_supporting_facts` flag to the Evaluator
2. Extract predicted supporting facts from the RAGResponse's retrieved docs
3. Compare against gold supporting facts

This requires vanilla_rag to return which sentences were used - a more complex change that may be deferred to Phase 2 architectures.

---

### Priority 5: Analysis Script

**File:** `scripts/analyze_results.py`

```python
#!/usr/bin/env python3
"""Analyze and compare benchmark results."""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import pandas as pd


def load_results(results_dir: Path) -> dict:
    """Load summary and predictions from a results directory."""
    summary_path = results_dir / "summary.json"
    predictions_path = results_dir / "predictions.jsonl"
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            predictions.append(json.loads(line))
    
    return {"summary": summary, "predictions": predictions}


def breakdown_by_question_type(predictions: list[dict]) -> pd.DataFrame:
    """Compute metrics breakdown by question type."""
    by_type = defaultdict(lambda: {"em": [], "f1": [], "count": 0})
    
    for pred in predictions:
        qtype = pred["question_type"]
        by_type[qtype]["em"].append(pred["exact_match"])
        by_type[qtype]["f1"].append(pred["f1"])
        by_type[qtype]["count"] += 1
    
    rows = []
    for qtype, metrics in by_type.items():
        rows.append({
            "question_type": qtype,
            "count": metrics["count"],
            "avg_em": sum(metrics["em"]) / len(metrics["em"]),
            "avg_f1": sum(metrics["f1"]) / len(metrics["f1"]),
        })
    
    return pd.DataFrame(rows)


def extract_errors(predictions: list[dict], threshold: float = 0.5) -> list[dict]:
    """Extract predictions with F1 below threshold for error analysis."""
    errors = []
    for pred in predictions:
        if pred["f1"] < threshold:
            errors.append({
                "question_id": pred["question_id"],
                "predicted": pred["predicted_answer"],
                "gold": pred["gold_answer"],
                "f1": pred["f1"],
            })
    return sorted(errors, key=lambda x: x["f1"])


def compare_runs(run_dirs: list[Path]) -> pd.DataFrame:
    """Compare metrics across multiple runs."""
    rows = []
    for run_dir in run_dirs:
        results = load_results(run_dir)
        summary = results["summary"]
        rows.append({
            "run": run_dir.name,
            "architecture": summary["architecture"],
            "model": summary["model"],
            "num_questions": summary["num_questions"],
            "exact_match": summary["avg_exact_match"],
            "f1": summary["avg_f1"],
            "latency_ms": summary["avg_latency_ms"],
            "cost_usd": summary["total_cost_usd"],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze RAG benchmark results")
    parser.add_argument("--results", required=True, help="Path to results directory or parent of multiple runs")
    parser.add_argument("--compare", action="store_true", help="Compare multiple runs")
    parser.add_argument("--errors", action="store_true", help="Show error analysis")
    parser.add_argument("--breakdown", action="store_true", help="Show breakdown by question type")
    args = parser.parse_args()
    
    results_path = Path(args.results)
    
    if args.compare:
        # Find all run directories
        run_dirs = [d for d in results_path.iterdir() if d.is_dir() and (d / "summary.json").exists()]
        df = compare_runs(run_dirs)
        print("\n=== Run Comparison ===")
        print(df.to_string(index=False))
    else:
        results = load_results(results_path)
        
        print("\n=== Summary ===")
        for key, value in results["summary"].items():
            print(f"{key}: {value}")
        
        if args.breakdown:
            print("\n=== Breakdown by Question Type ===")
            df = breakdown_by_question_type(results["predictions"])
            print(df.to_string(index=False))
        
        if args.errors:
            print("\n=== Error Analysis (F1 < 0.5) ===")
            errors = extract_errors(results["predictions"])
            for err in errors[:10]:  # Show top 10
                print(f"\nQ: {err['question_id']}")
                print(f"  Predicted: {err['predicted']}")
                print(f"  Gold: {err['gold']}")
                print(f"  F1: {err['f1']:.3f}")


if __name__ == "__main__":
    main()
```

---

## Run Commands Checklist

```bash
# 1. Run tests
pytest tests/ -v

# 2. Run vanilla with BM25 (100 questions)
python scripts/run_experiment.py --config configs/vanilla.yaml --subset 100

# 3. Run vanilla with hybrid retriever (after creating config)
python scripts/run_experiment.py --config configs/vanilla_hybrid.yaml --subset 100

# 4. View MLflow results
mlflow ui

# 5. Analyze results
python scripts/analyze_results.py --results results/<run_id> --breakdown --errors

# 6. Compare runs
python scripts/analyze_results.py --results results/ --compare
```

---

## Phase 2 Preparation Notes

Before starting agentic architectures (ReAct, Self-RAG, IRCoT):

1. **Refactor prompt handling** - Consider a PromptManager class for loading/formatting
2. **Add batch_retrieve to retrievers** - Needed for parallel sub-question retrieval
3. **Extend ReasoningStep** - Add fields for tool calls, reflection tokens
4. **Create architecture registry** - Factory pattern for loading architectures by name
5. **Add progress bars** - tqdm for long-running evaluations

---

## Estimated Effort

| Task | Effort | Priority |
|------|--------|----------|
| AnthropicClient implementation | 2 hours | P1 |
| Create retriever config variants | 30 min | P1 |
| Unit tests (vanilla_rag, metrics, llm_client, hotpotqa) | 3 hours | P2 |
| Analysis script | 2 hours | P2 |
| Run full 500-question benchmark | 1 hour (runtime) | P2 |
| Supporting fact evaluation | 2 hours | P3 |

**Total estimated effort:** ~10 hours

---

## Questions for Team Discussion

1. Should we run the 500-question validation before or after implementing Anthropic?
2. Do we want to track retrieval metrics separately (recall@k, precision@k)?
3. Should we add a cost budget limit to the experiment runner?
4. Do we need fullwiki setting for HotpotQA, or is distractor sufficient for the paper?
