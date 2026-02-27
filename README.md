# Agentic RAG Benchmark

Benchmarking AgenticRAG systems and its viability in the face of long context optimized recursive methods like Recursive RAG and Recursive LMs.

**Target:** ACL/NAACL 2026 publication

## Current Results

### Vanilla RAG Baseline (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Cost |
|-----------|-------------|----------|------|
| **Dense** | **45.0%** | **59.5%** | $0.79 |
| Hybrid    | 44.1%       | 58.6%    | $0.76 |
| BM25      | 38.2%       | 51.5%    | $0.77 |

*Model: gpt-4o-mini | Dense retrieval uses text-embedding-3-small*

### ReAct RAG (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **46.0%** | **59.9%** | 7,611 | $9.18 | 4.05 | 2.66 |
| Dense     | 45.7%       | 59.3%    | 5,950        | $9.66 | 4.07 | 2.69 |
| BM25      | 38.8%       | 50.8%    | 3,923        | $11.16 | 4.56 | 3.38 |

*Model: gpt-4o-mini | max_iterations=7 | concurrency=3*

### Planner RAG (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|------|---------------|----------|--------------|---------------------|
| **Dense** | **33.7%** | **44.9%** | 4,629 | $4.03 | 8.13 | 2.33 |
| Hybrid    | 33.5%       | 44.5%    | 6,857        | $3.98 | 8.14 | 2.34 |
| BM25      | 27.7%       | 37.7%    | 14,618       | $4.09 | 8.43 | 2.30 |

*Model: gpt-4o-mini | max_iterations=5 | max_branching_factor=2 | concurrency=2*

**By Question Type (Dense Retriever, Planner RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 33.9% | 47.7% |
| Comparison | 1,487 | 32.8% | 33.9% |

### Self-RAG (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **40.6%** | **55.0%** | 1,893 | $2.08 | 10.75 | 0.84 |
| Dense     | 40.6%       | 54.9%    | 4,497        | $2.13 | 10.77 | 0.84 |
| BM25      | 37.0%       | 50.4%    | 9,666        | $2.15 | 11.02 | 0.84 |

*Model: gpt-4o-mini | num_candidates=3 | concurrency=2*

**By Question Type (Hybrid Retriever, Self-RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 36.0% | 51.5% |
| Comparison | 1,487 | 59.0% | 68.7% |

### Recursive LM (HotpotQA Validation Subset)

Results from development subset (HotpotQA validation, `subset=100`, gpt-4o-mini):

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **BM25** | **52.0%** | 63.1% | 3,033 | $0.042 | 3.6 | 2.8 |
| Hybrid    | 51.0%       | **67.0%** | 3,265 | $0.038 | 3.1 | 2.4 |
| Dense     | 49.0%       | 65.2%    | 3,537        | $0.038 | 3.1 | 2.4 |

*Model: gpt-4o-mini | max_depth=3 | memoization=true | concurrency=5*

RLM implementation is complete and unit-tested; full validation run pending.

**By Question Type (BM25 Retriever, Recursive LM):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 79 | 46.8% | 58.9% |
| Comparison | 21 | 71.4% | 78.8% |

### Cross-Architecture Comparison (Best Retriever per Architecture)

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost |
|--------------|------|----------------|-------------|----------|---------------|------|
| Vanilla RAG  | Baseline | Dense    | 45.0%       | 59.5%    | 1.0           | $0.79 |
| **ReAct RAG** | **Agentic** | **Hybrid** | **46.0%** | **59.9%** | **4.05** | **$9.18** |
| Self-RAG     | Agentic  | Hybrid   | 40.6%       | 55.0%    | 10.75         | $2.08 |
| Planner RAG  | Agentic  | Dense    | 33.7%       | 44.9%    | 8.13          | $4.03 |
| Recursive LM | RLM | BM25 | 52.0%* | 63.1%* | 3.6 | $0.042* |

*\* Subset results (100 questions) — not directly comparable to full validation runs (7,405 questions). Full run pending.*

**Key Findings:**
- ReAct RAG with Hybrid retrieval achieves the best overall performance on full validation (46.0% EM, 59.9% F1)
- ReAct provides +1.0% EM and +0.4% F1 improvement over Vanilla RAG's best, but at ~12x the cost
- Planner RAG significantly underperforms all other architectures (33.7% EM with Dense), despite using 8+ LLM calls per question
- Planner RAG's tree-based planning approach appears to over-decompose questions, leading to higher error accumulation across sub-answers
- Self-RAG underperforms both Vanilla RAG (-4.4% EM) and ReAct RAG (-5.4% EM) despite using ~11 LLM calls per question
- Self-RAG's self-reflection mechanism often skips retrieval (avg 0.84 retrieval calls), which may hurt multi-hop performance where evidence gathering is critical
- Self-RAG is significantly cheaper than ReAct ($2.08 vs $9.18) but more expensive than Vanilla ($0.79), offering neither the best accuracy nor the best cost-efficiency
- Recursive LM shows promising subset results (52.0% EM) with very low cost (~$0.0004/question) and moderate LLM usage (3.6 calls), but full validation is needed to confirm
- BM25 consistently underperforms Dense/Hybrid across all architectures; Dense and Hybrid are closely matched
- ReAct uses 4-5 LLM calls and 2.7-3.4 retrievals per question on average
- BM25 with ReAct requires more iterations (4.56 LLM calls) than dense/hybrid (4.05-4.07), suggesting weaker initial retrieval drives more search attempts

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Unix/Mac)
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Validate Pipeline

```bash
python scripts/test_pipeline.py
```

### 4. Run Baseline Experiment

```bash
# Run with Dense retriever (recommended - best performance)
python scripts/run_experiment.py --config configs/vanilla_dense_full.yaml

# Or run a quick test with 100 questions
python scripts/run_experiment.py --config configs/vanilla_dense.yaml
```

### 5. Analyze Results

```bash
# View summary of a specific run
python scripts/analyze_results.py --results results/<run_id> --breakdown

# Compare multiple runs
python scripts/analyze_results.py --results results --compare
```

## Project Structure

```
agentic_rag_benchmark/
├── src/
│   ├── core/           # Base abstractions
│   ├── architectures/  # RAG implementations
│   ├── retrieval/      # BM25, Dense, Hybrid
│   ├── data/           # Dataset loaders
│   ├── evaluation/     # Metrics
│   └── utils/          # Cache, logging
├── configs/            # YAML configurations
├── prompts/            # Prompt templates
├── scripts/            # Experiment runners
└── tests/              # Unit tests
```

## Implemented Architectures

| Architecture | Type | Status |
|--------------|------|--------|
| Vanilla RAG | Baseline | ✅ Complete |
| ReAct RAG | Agentic | ✅ Complete |
| Self-RAG | Agentic | ✅ Complete |
| Planner RAG | Agentic | ✅ Complete |
| IRCoT | Recursive | 🔲 Planned |
| REAP | Recursive | 🔲 Planned |
| Recursive LM | RLM | ✅ Implemented (full results pending) |

## Datasets

- **HotpotQA** (implemented) - Multi-hop QA with bridge/comparison questions
- **MuSiQue** (planned) - Multi-hop with explicit decomposition
- **2WikiMultiHopQA** (planned) - Wikipedia-based reasoning

## License

MIT License - see [LICENSE](LICENSE) for details.
