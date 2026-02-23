# Agentic RAG Benchmark: Technical Implementation Specification

## 1. Executive Summary

### 1.1 Research Goal
This document specifies the implementation of a benchmarking framework for comparing three paradigms of retrieval-augmented generation for multi-hop question answering:

1. **Agentic RAG**: LLM agents with tool-use capabilities for iterative retrieval
2. **Recursive RAG**: Methods that interleave retrieval with reasoning steps
3. **Recursive Language Models (RLM)**: Programmatic self-recursion over prompt snippets

**Target Publication:** ACL/NAACL 2026
**Timeline:** 6 months
**Compute Constraint:** API-only (OpenAI, Anthropic)

### 1.2 Research Questions

1. How do state-of-the-art agentic RAG architectures compare in performance across different reasoning complexity levels?
2. What are the fundamental trade-offs between Agentic RAG, Recursive RAG, and Recursive LMs?
3. Under what conditions does each paradigm excel?
4. Can we identify optimal architecture selection criteria based on question characteristics?

### 1.3 Key Papers

| Category | Paper | Venue | Key Contribution |
|----------|-------|-------|------------------|
| Agentic | ReAct | ICLR 2023 | Reason + Act paradigm |
| Agentic | Self-RAG | ICLR 2024 | Self-reflection tokens |
| Agentic | RAGShaper | arXiv 2601.08699 | Automated trajectory synthesis |
| Agentic | TreePS-RAG | arXiv 2601.06922 | Tree-structured policy search actions |
| Recursive RAG | IRCoT | ACL 2023 | Interleaved retrieval + CoT |
| Recursive RAG | REAP | AAAI 2026 | Sub-task planning + fact extraction |
| Recursive RAG | HIRO | arXiv 2406.09979 | Hierarchical DFS retrieval |
| RLM | RLM | arXiv 2512.24601 | Programmatic self-recursion |

---

## 2. Project Structure

```
agentic_rag_benchmark/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py              # Dataclasses for all types
│   │   ├── base_rag.py           # Abstract base class
│   │   ├── llm_client.py         # LLM API clients (OpenAI, Anthropic)
│   │   └── retriever.py          # Retrieval interface
│   ├── architectures/
│   │   ├── __init__.py
│   │   ├── vanilla_rag.py        # Baseline: retrieve-then-read
│   │   ├── agentic/
│   │   │   ├── __init__.py
│   │   │   ├── react_rag.py      # ReAct-style agent
│   │   │   ├── self_rag.py       # Self-RAG with reflection
│   │   │   └── planner_rag.py    # Planner-based decomposition
│   │   ├── recursive/
│   │   │   ├── __init__.py
│   │   │   ├── ircot.py          # Interleaved Retrieval CoT
│   │   │   └── reap.py           # Recursive Eval + Adaptive Planning
│   │   └── rlm/
│   │       ├── __init__.py
│   │       └── recursive_lm.py   # Recursive Language Model
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25.py               # Sparse retrieval
│   │   ├── dense.py              # Dense retrieval (OpenAI embeddings)
│   │   └── hybrid.py             # Hybrid BM25 + Dense
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base_loader.py        # Abstract data loader
│   │   ├── hotpotqa.py           # HotpotQA loader
│   │   ├── musique.py            # MuSiQue loader
│   │   └── wikimultihop.py       # 2WikiMultiHopQA loader
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # EM, F1, Supporting Facts
│   │   └── evaluator.py          # Unified evaluation runner
│   └── utils/
│       ├── __init__.py
│       ├── cache.py              # SQLite response cache
│       ├── cost_tracker.py       # API cost monitoring
│       └── logging.py            # Structured logging
├── configs/
│   ├── base.yaml                 # Base configuration
│   ├── vanilla.yaml
│   ├── react.yaml
│   ├── self_rag.yaml
│   ├── ircot.yaml
│   ├── reap.yaml
│   └── rlm.yaml
├── prompts/
│   ├── vanilla.txt
│   ├── react.txt
│   ├── self_rag.txt
│   ├── ircot.txt
│   └── rlm.txt
├── scripts/
│   ├── run_experiment.py         # Main experiment runner
│   ├── download_data.py          # Dataset download script
│   └── analyze_results.py        # Results analysis
├── tests/
│   ├── __init__.py
│   ├── test_types.py
│   ├── test_llm_client.py
│   ├── test_retriever.py
│   ├── test_architectures.py
│   └── test_metrics.py
├── docs/
│   ├── TECHNICAL_SPEC.md         # This document
│   └── AGENTS.md                 # AI coding assistant guide
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 3. Core Abstractions

### 3.1 Type Definitions (`src/core/types.py`)

```python
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import numpy as np

class QuestionType(Enum):
    """Types of multi-hop questions."""
    BRIDGE = "bridge"           # Entity linked through intermediate
    COMPARISON = "comparison"   # Compare two entities
    COMPOSITIONAL = "compositional"  # Nested sub-questions
    SINGLE_HOP = "single_hop"   # Baseline single retrieval

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
    gold_answer: Optional[str] = None
    supporting_facts: Optional[list[tuple[str, int]]] = None  # (title, sent_idx)
    decomposition: Optional[list[str]] = None  # Sub-questions if available
    metadata: dict = field(default_factory=dict)

@dataclass
class Document:
    """Represents a document/passage in the corpus."""
    id: str
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.sentences and self.text:
            self.sentences = [s.strip() + '.' for s in self.text.split('.') if s.strip()]

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
    supporting_fact_em: Optional[float] = None
    supporting_fact_f1: Optional[float] = None
    joint_em: Optional[float] = None
    joint_f1: Optional[float] = None
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
    avg_supporting_fact_em: Optional[float]
    avg_supporting_fact_f1: Optional[float]
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
```

### 3.2 Base RAG Interface (`src/core/base_rag.py`)

```python
from abc import ABC, abstractmethod
from typing import Optional
from .types import Question, Document, RAGResponse, ArchitectureType
from .llm_client import BaseLLMClient
from .retriever import BaseRetriever

class BaseRAG(ABC):
    """Abstract base class for all RAG architectures."""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: BaseRetriever,
        config: dict
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.config = config
        self._validate_config()
    
    @abstractmethod
    async def answer(
        self,
        question: Question,
        corpus: list[Document]
    ) -> RAGResponse:
        """
        Answer a question using the RAG architecture.
        
        Args:
            question: The question to answer
            corpus: List of documents to retrieve from
            
        Returns:
            RAGResponse with answer, reasoning chain, and metadata
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the architecture name (e.g., 'react_rag')."""
        pass
    
    @abstractmethod
    def get_type(self) -> ArchitectureType:
        """Return the architecture type category."""
        pass
    
    @abstractmethod
    def get_config_schema(self) -> dict:
        """Return the config schema for validation."""
        pass
    
    def _validate_config(self) -> None:
        """Validate configuration against schema."""
        schema = self.get_config_schema()
        for key, (key_type, required, default) in schema.items():
            if key not in self.config:
                if required:
                    raise ValueError(f"Missing required config key: {key}")
                self.config[key] = default
            elif not isinstance(self.config[key], key_type):
                raise TypeError(f"Config key {key} must be {key_type}")
```

### 3.3 LLM Client Interface (`src/core/llm_client.py`)

```python
from abc import ABC, abstractmethod
from typing import Optional
import asyncio
import hashlib
import json
import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from ..utils.cache import SQLiteCache

class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""
    
    def __init__(
        self,
        model: str,
        cache: Optional[SQLiteCache] = None,
        track_costs: bool = True
    ):
        self.model = model
        self.cache = cache
        self.track_costs = track_costs
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
    
    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None
    ) -> tuple[str, int, float]:
        """
        Generate a response from the LLM.
        
        Returns:
            Tuple of (response_text, tokens_used, cost_usd)
        """
        pass
    
    async def batch_generate(
        self,
        batch: list[list[dict]],
        **kwargs
    ) -> list[tuple[str, int, float]]:
        """Generate responses for a batch of prompts in parallel."""
        tasks = [self.generate(messages, **kwargs) for messages in batch]
        return await asyncio.gather(*tasks)
    
    def _make_cache_key(self, messages: list[dict], **kwargs) -> str:
        """Create deterministic cache key."""
        data = {"model": self.model, "messages": messages, **kwargs}
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "model": self.model,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0


class OpenAIClient(BaseLLMClient):
    """OpenAI API client with caching and cost tracking."""
    
    PRICING = {
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    }
    
    EMBEDDING_PRICING = {
        "text-embedding-3-small": 0.02 / 1_000_000,
        "text-embedding-3-large": 0.13 / 1_000_000,
    }
    
    def __init__(
        self,
        model: str = "gpt-4o",
        cache: Optional[SQLiteCache] = None,
        track_costs: bool = True
    ):
        super().__init__(model, cache, track_costs)
        self.client = openai.AsyncOpenAI()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60)
    )
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None
    ) -> tuple[str, int, float]:
        # Check cache
        cache_key = self._make_cache_key(messages, temperature=temperature, max_tokens=max_tokens)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # API call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop
        )
        
        # Extract metrics
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = input_tokens + output_tokens
        
        pricing = self.PRICING.get(self.model, self.PRICING["gpt-4o"])
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        
        # Update stats
        if self.track_costs:
            self.total_tokens += total_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.call_count += 1
        
        result = (response.choices[0].message.content, total_tokens, cost)
        
        # Cache
        if self.cache:
            self.cache.set(cache_key, result)
        
        return result


class AnthropicClient(BaseLLMClient):
    """Anthropic API client with caching and cost tracking."""
    
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-3-5-haiku-20241022": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
        "claude-3-haiku-20240307": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    }
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        cache: Optional[SQLiteCache] = None,
        track_costs: bool = True
    ):
        super().__init__(model, cache, track_costs)
        self.client = anthropic.AsyncAnthropic()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60)
    )
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None
    ) -> tuple[str, int, float]:
        # Check cache
        cache_key = self._make_cache_key(messages, temperature=temperature, max_tokens=max_tokens)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Convert messages format (OpenAI -> Anthropic)
        system_msg = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append(msg)
        
        # API call
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
        
        # Extract metrics
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        
        pricing = self.PRICING.get(self.model, self.PRICING["claude-3-5-sonnet-20241022"])
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        
        # Update stats
        if self.track_costs:
            self.total_tokens += total_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.call_count += 1
        
        result = (response.content[0].text, total_tokens, cost)
        
        # Cache
        if self.cache:
            self.cache.set(cache_key, result)
        
        return result
```

### 3.4 Retriever Interface (`src/core/retriever.py`)

```python
from abc import ABC, abstractmethod
from typing import Optional
from .types import Document, RetrievalResult

class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    def __init__(self):
        self.is_indexed = False
        self.corpus_size = 0
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        corpus: list[Document],
        top_k: int = 5
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query."""
        pass
    
    @abstractmethod
    async def index(self, corpus: list[Document]) -> None:
        """Pre-index a corpus for faster retrieval."""
        pass
    
    @abstractmethod
    async def batch_retrieve(
        self,
        queries: list[str],
        corpus: list[Document],
        top_k: int = 5
    ) -> list[RetrievalResult]:
        """Retrieve for multiple queries (can be parallelized)."""
        pass
```

---

## 4. Architecture Implementations

### 4.1 Vanilla RAG (Baseline)

**Category:** Baseline
**Description:** Simple retrieve-then-read approach. Single retrieval, single generation.

**Algorithm:**
1. Encode question as query
2. Retrieve top-k documents
3. Concatenate documents into context
4. Generate answer with single LLM call

**Config Schema:**
```yaml
vanilla:
  top_k: 5
  max_context_tokens: 4000
```

**Expected Performance:**
- Fast (1 retrieval, 1 LLM call)
- Lower accuracy on multi-hop questions
- Baseline for comparison

---

### 4.2 ReAct RAG (Agentic)

**Category:** Agentic
**Paper:** ReAct: Synergizing Reasoning and Acting in Language Models (ICLR 2023)

**Description:** Implements Reason+Act loop where the LLM generates thoughts and actions iteratively.

**Algorithm:**
```
for iteration in range(max_iterations):
    prompt = build_react_prompt(question, scratchpad)
    response = llm.generate(prompt, stop=["Observation:"])
    thought, action, action_input = parse_response(response)
    
    if action == "finish":
        return action_input as answer
    elif action == "search":
        observation = retriever.retrieve(action_input)
        scratchpad.append((thought, action, action_input, observation))
    elif action == "lookup":
        observation = lookup_in_retrieved(action_input)
        scratchpad.append(...)
```

**Tools:**
- `search[query]`: Retrieve passages matching query
- `lookup[term]`: Find term in previously retrieved passages
- `finish[answer]`: Return final answer

**Config Schema:**
```yaml
react:
  top_k: 5
  max_iterations: 7
  max_context_tokens: 4000
```

---

### 4.3 Self-RAG (Agentic)

**Category:** Agentic
**Paper:** Self-RAG: Learning to Retrieve, Generate, and Critique (ICLR 2024)

**Description:** Adds self-reflection tokens to decide retrieval and critique generations.

**Reflection Tokens:**
- `[Retrieval]`: yes/no - Should I retrieve?
- `[IsRel]`: relevant/irrelevant - Is passage relevant to query?
- `[IsSup]`: fully/partially/no support - Does passage support generation?
- `[IsUse]`: 1-5 rating - Overall utility

**Algorithm:**
```
# Phase 1: Decide retrieval
retrieval_decision = llm.generate(question + "[Retrieval]")

if retrieval_decision == "yes":
    passages = retriever.retrieve(question)
    
    # Phase 2: Filter relevant passages
    relevant_passages = []
    for p in passages:
        relevance = llm.generate(question + p + "[IsRel]")
        if relevance == "relevant":
            relevant_passages.append(p)
    
    # Phase 3: Generate with critique
    candidates = []
    for p in relevant_passages:
        generation = llm.generate(question + p)
        support = llm.generate(question + p + generation + "[IsSup]")
        utility = llm.generate(question + generation + "[IsUse]")
        candidates.append((generation, support, utility))
    
    # Select best candidate
    answer = select_best(candidates)
else:
    answer = llm.generate(question)  # No retrieval needed
```

**Config Schema:**
```yaml
self_rag:
  top_k: 5
  num_candidates: 3
  relevance_threshold: 0.5
  support_threshold: "partial"
```

---

### 4.4 Planner RAG (Agentic)

**Category:** Agentic
**Inspired by:** RAGShaper, TreePS-RAG

**Description:** Uses an explicit planner action loop over a dynamic reasoning tree with
bounded search budget. The planner chooses one action at a time (`TRAVERSE`, `SELECT`,
`ROLLOUT`, `BACKTRACK`, `STOP`) and the solver retrieves evidence + answers the selected node.
Pre-full-run policy: use subset validation during optimization passes; run full validation after
merging planner changes to the integration branch.

**Algorithm:**
```
# Stage A: Gate (direct vs recursive)
gate = llm.generate(gate_prompt(question))  # JSON: {"direct_answer": bool}
if gate.direct_answer:
    return llm.generate(direct_answer_prompt(question))

# Stage B: Tree planning loop
nodes = {"root": Node(question=question, depth=0, status="open")}
active = "root"

for i in range(max_iterations):
    if root_is_confident(nodes["root"], min_stop_confidence):
        break

    # Planner picks next action
    action = llm.generate(planner_prompt(tree_state(nodes), active))
    # action JSON: {"action": "SELECT|ROLLOUT|TRAVERSE|BACKTRACK|STOP", "node_id": "..."}

    if action.type == "STOP":
        break
    elif action.type in {"SELECT", "TRAVERSE"}:
        target = resolve_target(action.node_id, active, nodes)
        retrieval = retriever.retrieve(build_query(target, nodes), top_k=top_k)
        answer = llm.generate(solver_prompt(target.question, retrieval))
        confidence = llm.generate(confidence_prompt(target.question, answer, retrieval))
        update_node(target, answer, confidence)
        active = target.id
    elif action.type == "ROLLOUT":
        target = resolve_target(action.node_id, active, nodes)
        children = llm.generate(rollout_prompt(target))
        add_children(target, children[:max_branching_factor], max_depth=max_depth)
        active = target.id
    elif action.type == "BACKTRACK":
        prune(active)
        active = parent(active)

# Final synthesis from solved nodes
answer = llm.generate(synthesis_prompt(question, solved_nodes(nodes)))

# Bridge/compositional hardening (bounded +1 LLM call)
if question_type in {"bridge", "compositional"} and is_invalid_or_partial(answer):
    answer = llm.generate(bridge_refine_prompt(question, answer, solved_nodes(nodes)))
    if is_invalid_or_partial(answer):
        answer = deterministic_entity_fallback(solved_nodes(nodes))
```

**Config Schema:**
```yaml
planner:
  top_k: 3
  max_iterations: 5
  max_branching_factor: 2
  rollout_similarity_threshold: 0.85
  max_depth: 3
  min_stop_confidence: 0.8
  allow_direct_answer: true
  max_context_tokens: 4000
  planner_prompt_path: "prompts/planner_action.txt"
  solver_prompt_path: "prompts/planner_solve.txt"
  synthesis_prompt_path: "prompts/planner_synthesize.txt"
  bridge_refine_enabled: true
  bridge_refine_max_attempts: 1
  bridge_refine_prompt_path: "prompts/planner_bridge_refine.txt"
  bridge_generic_answers: ["yes", "no", "unknown", "none", "n/a"]
```

---

### 4.5 IRCoT (Recursive RAG)

**Category:** Recursive RAG
**Paper:** Interleaving Retrieval with Chain-of-Thought Reasoning (ACL 2023)

**Description:** Interleaves retrieval with each step of chain-of-thought reasoning.

**Algorithm:**
```
cot_steps = []
for step in range(max_steps):
    # Generate next CoT step
    prompt = build_ircot_prompt(question, cot_steps)
    cot_step = llm.generate(prompt, stop=["[RETRIEVAL]", "[ANSWER]"])
    
    if "[ANSWER]" in cot_step:
        answer = extract_answer(cot_step)
        return answer
    
    # Retrieve based on CoT step
    query = extract_query_from_cot(cot_step)
    passages = retriever.retrieve(query)
    
    cot_steps.append({
        "thought": cot_step,
        "retrieval": passages
    })

# Max steps reached - force answer
answer = force_answer(question, cot_steps)
```

**Key Insight:** Each reasoning step informs what to retrieve next, and retrieved content informs next reasoning step.

**Config Schema:**
```yaml
ircot:
  top_k: 3
  max_steps: 5
  retrieval_trigger: "[RETRIEVAL]"
  answer_trigger: "[ANSWER]"
```

---

### 4.6 REAP (Recursive RAG)

**Category:** Recursive RAG
**Paper:** REAP: Recursive Evaluation and Adaptive Planning (AAAI 2026)

**Description:** Uses a Fact Extractor to identify missing information and adaptively plans retrieval.

**Components:**
1. **Sub-task Planner**: Decomposes question into retrieval sub-tasks
2. **Fact Extractor**: Extracts atomic facts from retrieved passages
3. **Completeness Evaluator**: Checks if enough facts gathered
4. **Answer Generator**: Synthesizes final answer

**Algorithm:**
```
facts = []
retrieval_history = []

while not is_complete(facts, question):
    # Plan next retrieval
    sub_task = planner.plan(question, facts, retrieval_history)
    
    if sub_task is None:
        break  # No more useful retrievals
    
    # Execute retrieval
    passages = retriever.retrieve(sub_task.query)
    retrieval_history.append(sub_task)
    
    # Extract facts
    new_facts = fact_extractor.extract(passages, question)
    facts.extend(new_facts)
    
    # Check completeness
    if completeness_evaluator.is_complete(facts, question):
        break

# Generate answer from facts
answer = answer_generator.generate(question, facts)
```

**Config Schema:**
```yaml
reap:
  top_k: 5
  max_iterations: 5
  completeness_threshold: 0.8
  fact_extraction_prompt: "prompts/reap_fact_extract.txt"
```

---

### 4.7 Recursive Language Model (RLM)

**Category:** RLM
**Paper:** Recursive Language Models (arXiv 2512.24601)

**Description:** Programmatic self-recursion over prompt snippets. The LLM generates code that calls itself recursively on sub-problems.

**Key Concepts:**
- **Prompt Snippets**: Reusable prompt components
- **Self-Recursion**: LLM generates calls to itself with modified inputs
- **Depth Control**: Maximum recursion depth to prevent infinite loops
- **Memoization**: Cache recursive call results

**Algorithm:**
```python
def rlm_answer(question, depth=0, memo={}):
    if depth > max_depth:
        return direct_answer(question)
    
    cache_key = hash(question)
    if cache_key in memo:
        return memo[cache_key]
    
    # Generate recursive program
    program = llm.generate(f"""
    Question: {question}
    
    You can either:
    1. Answer directly if simple enough
    2. Decompose into sub-questions and call RECURSE(sub_question)
    
    Output format:
    DIRECT: <answer>
    or
    DECOMPOSE:
    - RECURSE(<sub_question_1>)
    - RECURSE(<sub_question_2>)
    COMBINE: <how to combine results>
    """)
    
    if program.startswith("DIRECT:"):
        answer = program.split("DIRECT:")[1].strip()
    else:
        sub_questions = extract_sub_questions(program)
        sub_answers = [rlm_answer(sq, depth+1, memo) for sq in sub_questions]
        combine_instruction = extract_combine(program)
        answer = llm.generate(combine_instruction.format(*sub_answers))
    
    memo[cache_key] = answer
    return answer
```

**Advantages over Agentic RAG:**
- Handles inputs 2 orders of magnitude beyond context window
- More systematic decomposition
- Natural memoization

**Config Schema:**
```yaml
rlm:
  max_depth: 3
  top_k: 5
  memoization: true
  decomposition_strategy: "binary"  # or "multi"
```

---

## 5. Retrieval Components

### 5.1 BM25 Retriever (`src/retrieval/bm25.py`)

**Description:** Classic sparse retrieval using BM25 scoring.

```python
from rank_bm25 import BM25Okapi

class BM25Retriever(BaseRetriever):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or (lambda x: x.lower().split())
        self.bm25 = None
        self.corpus = None
    
    async def index(self, corpus: list[Document]) -> None:
        self.corpus = corpus
        tokenized = [self.tokenizer(doc.text) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.is_indexed = True
        self.corpus_size = len(corpus)
    
    async def retrieve(self, query, corpus, top_k=5) -> RetrievalResult:
        if not self.is_indexed or self.corpus != corpus:
            await self.index(corpus)
        
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        
        return RetrievalResult(
            documents=[self.corpus[i] for i in top_indices],
            scores=[float(scores[i]) for i in top_indices],
            query=query,
            retrieval_time_ms=...,
            method="bm25"
        )
```

### 5.2 Dense Retriever (`src/retrieval/dense.py`)

**Description:** Dense retrieval using OpenAI embeddings API.

```python
class DenseRetriever(BaseRetriever):
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.client = openai.AsyncOpenAI()
        self.embeddings = None
    
    async def index(self, corpus: list[Document]) -> None:
        texts = [doc.text for doc in corpus]
        # Batch embed (max 2048 per request)
        all_embeddings = []
        for i in range(0, len(texts), 2048):
            batch = texts[i:i+2048]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            all_embeddings.extend([e.embedding for e in response.data])
        
        self.embeddings = np.array(all_embeddings)
        self.corpus = corpus
        self.is_indexed = True
    
    async def retrieve(self, query, corpus, top_k=5) -> RetrievalResult:
        # Embed query
        response = await self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        query_emb = np.array(response.data[0].embedding)
        
        # Cosine similarity
        scores = np.dot(self.embeddings, query_emb)
        top_indices = scores.argsort()[-top_k:][::-1]
        
        return RetrievalResult(...)
```

### 5.3 Hybrid Retriever (`src/retrieval/hybrid.py`)

**Description:** Combines BM25 and dense retrieval using Reciprocal Rank Fusion (RRF).

```python
class HybridRetriever(BaseRetriever):
    def __init__(self, bm25_weight=0.5, dense_weight=0.5, rrf_k=60):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k  # RRF constant
    
    async def retrieve(self, query, corpus, top_k=5) -> RetrievalResult:
        # Get both results
        bm25_result = await self.bm25.retrieve(query, corpus, top_k=top_k*2)
        dense_result = await self.dense.retrieve(query, corpus, top_k=top_k*2)
        
        # Reciprocal Rank Fusion
        doc_scores = {}
        for rank, doc in enumerate(bm25_result.documents):
            doc_scores[doc.id] = self.bm25_weight / (self.rrf_k + rank + 1)
        for rank, doc in enumerate(dense_result.documents):
            doc_scores[doc.id] = doc_scores.get(doc.id, 0) + self.dense_weight / (self.rrf_k + rank + 1)
        
        # Sort and return top-k
        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)[:top_k]
        ...
```

---

## 6. Data Loading

### 6.1 HotpotQA (`src/data/hotpotqa.py`)

**Settings:**
- **Distractor**: 10 paragraphs provided (2 gold + 8 distractors)
- **Fullwiki**: Open retrieval from Wikipedia

```python
from datasets import load_dataset

class HotpotQALoader:
    def __init__(self, setting="distractor", split="validation", subset_size=None):
        self.setting = setting
        self.split = split
        self.subset_size = subset_size
    
    def load(self) -> tuple[list[Question], list[Document]]:
        if self.setting == "distractor":
            dataset = load_dataset("hotpot_qa", "distractor", split=self.split)
        else:
            dataset = load_dataset("hotpot_qa", "fullwiki", split=self.split)
        
        if self.subset_size:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))
        
        questions = []
        corpus = {}
        
        for item in dataset:
            q = Question(
                id=item["id"],
                text=item["question"],
                type=QuestionType(item["type"]) if item["type"] in ["bridge", "comparison"] else QuestionType.BRIDGE,
                gold_answer=item["answer"],
                supporting_facts=list(zip(
                    item["supporting_facts"]["title"],
                    item["supporting_facts"]["sent_id"]
                ))
            )
            questions.append(q)
            
            # Build corpus from context
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                doc_id = f"{item['id']}_{title}"
                corpus[doc_id] = Document(
                    id=doc_id,
                    title=title,
                    text=" ".join(sentences),
                    sentences=sentences
                )
        
        return questions, list(corpus.values())
```

### 6.2 MuSiQue (`src/data/musique.py`)

**Description:** Multi-hop questions with explicit decomposition.

```python
class MuSiQueLoader:
    def load(self) -> tuple[list[Question], list[Document]]:
        dataset = load_dataset("musi_que", split=self.split)
        # MuSiQue has explicit decomposed sub-questions
        for item in dataset:
            q = Question(
                ...
                decomposition=item.get("decomposition", None)
            )
```

### 6.3 2WikiMultiHopQA (`src/data/wikimultihop.py`)

**Description:** Wikipedia-based multi-hop QA with evidence chains.

---

## 7. Evaluation Framework

### 7.1 Metrics (`src/evaluation/metrics.py`)

```python
import re
import string
from collections import Counter

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def supporting_fact_metrics(
    pred_facts: list[tuple[str, int]],
    gold_facts: list[tuple[str, int]]
) -> tuple[float, float]:
    """Compute EM and F1 for supporting facts."""
    pred_set = set(pred_facts)
    gold_set = set(gold_facts)
    
    em = float(pred_set == gold_set)
    
    if len(pred_set) == 0 or len(gold_set) == 0:
        f1 = float(pred_set == gold_set)
    else:
        precision = len(pred_set & gold_set) / len(pred_set)
        recall = len(pred_set & gold_set) / len(gold_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return em, f1
```

### 7.2 Novel Metrics (Research Contribution)

**Reasoning Chain Fidelity:**
```python
def reasoning_chain_fidelity(
    reasoning_chain: list[ReasoningStep],
    retrieved_docs: list[RetrievalResult]
) -> float:
    """Measure if each reasoning step is grounded in retrieved documents."""
    grounded_steps = 0
    for step in reasoning_chain:
        if is_grounded(step.thought, retrieved_docs):
            grounded_steps += 1
    return grounded_steps / len(reasoning_chain)
```

**Hallucination Rate:**
```python
def hallucination_rate(
    answer: str,
    reasoning_chain: list[ReasoningStep],
    retrieved_docs: list[RetrievalResult]
) -> float:
    """Percentage of claims in answer not supported by retrieved docs."""
    claims = extract_claims(answer)
    unsupported = sum(1 for c in claims if not is_supported(c, retrieved_docs))
    return unsupported / len(claims) if claims else 0.0
```

**Retrieval Efficiency:**
```python
def retrieval_efficiency(
    f1_score: float,
    num_retrieval_calls: int
) -> float:
    """F1 per retrieval call - measures retrieval ROI."""
    return f1_score / num_retrieval_calls if num_retrieval_calls > 0 else f1_score
```

---

## 8. Configuration System

### 8.1 Base Config (`configs/base.yaml`)

```yaml
# Base configuration - all other configs inherit from this
experiment:
  name: "base"
  seed: 42
  output_dir: "results"

llm:
  provider: "openai"  # openai, anthropic
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 1024

retrieval:
  method: "hybrid"  # bm25, dense, hybrid
  top_k: 5
  bm25_weight: 0.5
  dense_weight: 0.5
  embedding_model: "text-embedding-3-small"

data:
  dataset: "hotpotqa"
  setting: "distractor"
  split: "validation"
  subset_size: 500  # null for full

evaluation:
  max_concurrency: 5
  compute_supporting_facts: false
  save_predictions: true

cache:
  enabled: true
  path: ".cache/llm_cache.db"

logging:
  level: "INFO"
  file: "logs/experiment.log"
```

### 8.2 Architecture Configs

**configs/react.yaml:**
```yaml
inherits: base.yaml

experiment:
  name: "react"

architecture:
  name: "react_rag"

react:
  max_iterations: 7
```

**configs/planner.yaml:**
```yaml
inherits: base.yaml

experiment:
  name: "planner"

architecture:
  name: "planner_rag"

planner:
  max_iterations: 5
  max_branching_factor: 2
  rollout_similarity_threshold: 0.85
  max_depth: 3
  min_stop_confidence: 0.8
  allow_direct_answer: true
  max_context_tokens: 4000
  bridge_refine_enabled: true
  bridge_refine_max_attempts: 1
  bridge_refine_prompt_path: "prompts/planner_bridge_refine.txt"
  bridge_generic_answers: ["yes", "no", "unknown", "none", "n/a"]

retrieval:
  method: "bm25"
  top_k: 3
```

**configs/ircot.yaml:**
```yaml
inherits: base.yaml

experiment:
  name: "ircot"

architecture:
  name: "ircot"

ircot:
  max_steps: 5
  top_k: 3
  retrieval_trigger: "[RETRIEVAL]"
  answer_trigger: "[ANSWER]"
```

**configs/rlm.yaml:**
```yaml
inherits: base.yaml

experiment:
  name: "rlm"

architecture:
  name: "recursive_lm"

rlm:
  max_depth: 3
  memoization: true
  decomposition_strategy: "adaptive"
```

---

## 9. API Cost Management

### 9.1 Cost Estimates

| Model | Input ($/1M) | Output ($/1M) | Est./Question |
|-------|--------------|---------------|---------------|
| gpt-4o | $2.50 | $10.00 | $0.02-0.05 |
| gpt-4o-mini | $0.15 | $0.60 | $0.001-0.003 |
| claude-3.5-sonnet | $3.00 | $15.00 | $0.03-0.06 |
| claude-3.5-haiku | $0.80 | $4.00 | $0.005-0.01 |
| text-embedding-3-small | $0.02 | - | $0.0001 |

### 9.2 Budget Planning

**Development Phase (cheap models):**
- 500 questions × 4 architectures × 5 iterations = 10,000 runs
- At $0.002/run = $20

**Full Benchmark (full models):**
- 7,400 questions × 6 architectures × 2 models = 88,800 runs
- At $0.04/run = ~$3,500

### 9.3 Cost Optimization Strategies

1. **SQLite Caching**: Cache all LLM responses
2. **Development Models**: Use mini/haiku during development
3. **Subset Testing**: Test on 500 questions first
4. **Early Stopping**: Stop unpromising experiments early
5. **Batch Embedding**: Embed corpus once, reuse across experiments

---

## 10. Experiment Scripts

### 10.1 Main Runner (`scripts/run_experiment.py`)

```python
#!/usr/bin/env python3
import asyncio
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import json

async def main(config_path: str):
    config = load_config(config_path)
    
    # Setup
    cache = SQLiteCache(config["cache"]["path"]) if config["cache"]["enabled"] else None
    llm = create_llm_client(config["llm"], cache)
    retriever = create_retriever(config["retrieval"])
    rag = create_architecture(config["architecture"], llm, retriever, config)
    
    # Load data
    questions, corpus = load_data(config["data"])
    await retriever.index(corpus)
    
    # Run evaluation
    evaluator = Evaluator(rag, max_concurrency=config["evaluation"]["max_concurrency"])
    results = await evaluator.evaluate(questions, corpus)
    
    # Save results
    save_results(results, config)
    print_summary(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.config))
```

### 10.2 Analysis Script (`scripts/analyze_results.py`)

```python
#!/usr/bin/env python3
"""Analyze and compare benchmark results."""

def load_results(results_dir: str) -> dict:
    """Load all result files from directory."""
    ...

def compare_architectures(results: dict) -> pd.DataFrame:
    """Create comparison table."""
    ...

def plot_performance_vs_cost(results: dict):
    """Plot accuracy vs API cost tradeoff."""
    ...

def breakdown_by_question_type(results: dict):
    """Analyze performance by question type (bridge vs comparison)."""
    ...
```

---

## 11. Testing

### 11.1 Test Structure

```python
# tests/test_types.py
def test_question_creation():
    q = Question(id="1", text="test", type=QuestionType.BRIDGE)
    assert q.id == "1"

# tests/test_llm_client.py
@pytest.mark.asyncio
async def test_openai_client_caching():
    cache = SQLiteCache(":memory:")
    client = OpenAIClient(model="gpt-4o-mini", cache=cache)
    # First call
    r1, _, _ = await client.generate([{"role": "user", "content": "Hi"}])
    # Second call should hit cache
    r2, _, _ = await client.generate([{"role": "user", "content": "Hi"}])
    assert r1 == r2

# tests/test_architectures.py
@pytest.mark.asyncio
async def test_vanilla_rag():
    # Mock LLM and retriever
    ...
```

---

## 12. Implementation Phases

### Phase 1: MVP (Weeks 1-2)
- [x] Core types and interfaces
- [ ] LLM clients (OpenAI + Anthropic)
- [x] SQLite cache
- [x] Hybrid retriever
- [x] Vanilla RAG
- [x] ReAct RAG
- [x] HotpotQA loader
- [x] Basic metrics (EM, F1)
- [x] Main experiment script

### Phase 2: Full Agentic + Recursive (Weeks 3-4)
- [x] Self-RAG
- [x] Planner RAG (inference + bridge quality hardening)
- [ ] IRCoT
- [ ] REAP
- [ ] Supporting fact metrics
- [ ] MuSiQue loader

### Phase 3: RLM + Analysis (Weeks 5-6)
- [ ] Recursive LM implementation
- [ ] Novel metrics (chain fidelity, hallucination)
- [ ] 2WikiMultiHopQA loader
- [ ] Analysis scripts
- [ ] Result visualization

### Phase 4: Full Benchmark (Weeks 7-8)
- [ ] Run all architectures on all datasets
- [ ] Full model experiments (gpt-4o, claude-3.5-sonnet)
- [ ] Statistical significance tests
- [ ] Error analysis

### Phase 5: Paper Writing (Weeks 9-12)
- [ ] Introduction + Related Work
- [ ] Method description
- [ ] Experimental results
- [ ] Analysis + Discussion
- [ ] Submission preparation

---

## 13. Dependencies

```toml
[project]
name = "agentic-rag-benchmark"
version = "0.1.0"
description = "Benchmarking Agentic RAG vs Recursive RAG vs RLM"
requires-python = ">=3.11"

dependencies = [
    "openai>=1.0",
    "anthropic>=0.18",
    "rank-bm25>=0.2.2",
    "datasets>=2.14",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "tqdm>=4.65",
    "tenacity>=8.2",
    "tiktoken>=0.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
analysis = [
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "scipy>=1.10",
]
```

---

## Appendix A: Prompt Templates

### A.1 Vanilla RAG Prompt
```
Answer the question based on the provided context. Be concise and direct.

Context:
{context}

Question: {question}

Answer:
```

### A.2 ReAct Prompt
```
Answer the following question by searching for relevant information.

Available tools:
- search[query]: Search for passages related to the query
- lookup[term]: Look up a term in previously retrieved passages
- finish[answer]: Return your final answer

Question: {question}

{scratchpad}

Thought:
```

### A.3 IRCoT Prompt
```
Answer the following question step by step. After each reasoning step, indicate if you need to retrieve more information with [RETRIEVAL] or provide your final answer with [ANSWER].

Question: {question}

{cot_steps}

Step {n}:
```

### A.4 RLM Decomposition Prompt
```
You are solving a complex question by decomposition.

Question: {question}

You can either:
1. Answer directly if simple: OUTPUT: DIRECT: <answer>
2. Decompose and recurse: OUTPUT: DECOMPOSE: [sub_q1, sub_q2, ...] COMBINE: <instruction>

Choose wisely based on complexity.

OUTPUT:
```

---

## Appendix B: HotpotQA Evaluation Protocol

Following the official HotpotQA evaluation:

1. **Answer Metrics:**
   - Exact Match (EM): Normalized string match
   - F1: Token-level precision/recall

2. **Supporting Fact Metrics:**
   - SP EM: All supporting facts exactly correct
   - SP F1: F1 over supporting facts

3. **Joint Metrics:**
   - Joint EM: Both answer and supporting facts correct
   - Joint F1: Product of answer F1 and SP F1

4. **Normalization:**
   - Lowercase
   - Remove articles (a, an, the)
   - Remove punctuation
   - Remove extra whitespace
