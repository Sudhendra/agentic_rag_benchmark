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

### Vanilla RAG Baseline (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Cost |
|-----------|-------------|----------|------|
| **Dense** | **12.6%** | **24.3%** | $0.27 |
| Hybrid    | 12.0%       | 23.5%    | $0.27 |
| BM25      | 6.9%        | 16.3%    | $0.26 |

*Model: gpt-4o-mini | Dense retrieval uses text-embedding-3-small*

**By Question Type (Dense Retriever, Vanilla RAG on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 17.1% | 29.6% |
| Compositional | ~300 | 7.7% | 18.6% |

### ReAct RAG (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **19.7%** | **27.8%** | 9,374 | $5.24 | 5.57 | 4.67 |
| Hybrid    | 19.3%       | 28.1%    | 8,876        | $5.15 | 5.60 | 4.69 |
| BM25      | 11.9%       | 18.0%    | 11,703       | $5.74 | 6.09 | 5.49 |

*Model: gpt-4o-mini | max_iterations=7 | concurrency=3*

**By Question Type (Dense Retriever, ReAct RAG on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 23.6% | 33.9% |
| Compositional | ~300 | 15.5% | 21.2% |

### Planner RAG (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **15.8%** | **25.9%** | 6,227 | $2.08 | 10.15 | 2.77 |
| Hybrid    | 14.6%       | 24.3%    | 12,642       | $2.03 | 10.20 | 2.75 |
| BM25      | 7.5%        | 15.1%    | 11,760       | $2.07 | 10.81 | 2.59 |

*Model: gpt-4o-mini | max_iterations=5 | max_branching_factor=2 | concurrency=2*

**By Question Type (Dense Retriever, Planner RAG on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 21.1% | 33.0% |
| Compositional | ~300 | 10.1% | 18.2% |

### Self-RAG (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **11.6%** | **23.2%** | 4,290 | $1.00 | 13.79 | 0.95 |
| Hybrid    | 10.8%       | 22.3%    | 977          | $0.95 | 13.57 | 0.95 |
| BM25      | 6.8%        | 16.3%    | 6,630        | $0.98 | 14.10 | 0.95 |

*Model: gpt-4o-mini | num_candidates=3 | concurrency=2*

**By Question Type (Dense Retriever, Self-RAG on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 15.8% | 28.0% |
| Compositional | ~300 | 7.1% | 17.9% |

### Recursive LM (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **16.7%** | **28.9%** | 8,772 | $2.10 | 7.41 | 5.32 |
| Hybrid    | 16.5%       | 28.6%    | 13,235       | $2.14 | 7.83 | 5.59 |
| BM25      | 9.8%        | 19.8%    | 13,753       | $3.09 | 12.29 | 8.51 |

*Model: gpt-4o-mini | max_depth=3 | memoization=true | concurrency=2*

**By Question Type (Dense Retriever, Recursive LM on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 21.3% | 34.9% |
| Compositional | ~300 | 11.7% | 22.4% |

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

### Recursive LM (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **46.3%** | **60.2%** | 7,078 | $3.19 | 3.60 | 2.74 |
| Dense     | 46.1%       | 60.1%    | 4,951        | $3.34 | 3.61 | 2.75 |
| BM25      | 40.1%       | 52.6%    | 9,576        | $5.01 | 5.98 | 4.29 |

*Model: gpt-4o-mini | max_depth=3 | memoization=true | concurrency=2*

**By Question Type (Hybrid Retriever, Recursive LM):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 40.7% | 55.9% |
| Comparison | 1,487 | 67.8% | 77.0% |

### IRCoT (HotpotQA Subset Smoke Validation - 5 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| BM25      | 40.0%       | 64.4%    | 3,127        | $0.0028 | 4.8        | 4.8 |

*Model: gpt-4o-mini | max_steps=4 | subset_size=5 | smoke validation only, not directly comparable to full-validation runs*

**By Question Type (BM25 Retriever, IRCoT subset):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 3 | 66.7% | 88.9% |
| Comparison | 2 | 0.0% | 27.8% |

### MuSiQue Cross-Architecture Comparison (Best Retriever per Architecture)

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost |
|--------------|------|----------------|-------------|----------|---------------|------|
| Vanilla RAG  | Baseline | Dense    | 12.6%       | 24.3%    | 1.0           | $0.27 |
| **ReAct RAG** | **Agentic** | **Dense** | **19.7%** | **27.8%** | **5.57** | **$5.24** |
| Recursive LM | RLM | Dense | 16.7% | 28.9% | 7.41 | $2.10 |
| Planner RAG  | Agentic  | Dense    | 15.8%       | 25.9%    | 10.15         | $2.08 |
| Self-RAG     | Agentic  | Dense   | 11.6%       | 23.2%    | 13.79         | $1.00 |

**Key MuSiQue Findings:**
- **ReAct RAG leads on MuSiQue** with 19.7% EM, outperforming all other architectures
- Recursive LM is second best at 16.7% EM, followed by Planner RAG (15.8%), Vanilla RAG (12.6%), and Self-RAG (11.6%)
- Unlike HotpotQA, ReAct significantly outperforms Vanilla RAG on MuSiQue (+7.1% EM)
- Self-RAG underperforms Vanilla RAG on MuSiQue (-1.0% EM), consistent with HotpotQA pattern
- All architectures except Self-RAG outperform the Vanilla RAG baseline on MuSiQue

### Cross-Architecture Comparison (HotpotQA Best Retriever per Architecture)

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost |
|--------------|------|----------------|-------------|----------|---------------|------|
| Vanilla RAG  | Baseline | Dense    | 45.0%       | 59.5%    | 1.0           | $0.79 |
| **ReAct RAG** | **Agentic** | **Hybrid** | **46.0%** | **59.9%** | **4.05** | **$9.18** |
| Recursive LM | RLM | Hybrid | 46.3% | 60.2% | 3.60 | $3.19 |
| Self-RAG     | Agentic  | Hybrid   | 40.6%       | 55.0%    | 10.75         | $2.08 |
| Planner RAG  | Agentic  | Dense    | 33.7%       | 44.9%    | 8.13          | $4.03 |

**Key Findings:**
- **ReAct RAG and Recursive LM are tied for best EM** (46.0% vs 46.3%), but Recursive LM achieves this with ~4x lower cost ($3.19 vs $9.18)
- ReAct RAG with Hybrid retrieval achieves the best overall F1 (59.9%), narrowly edging Recursive LM (60.2%)
- Recursive LM offers the best cost-efficiency among agentic architectures: $3.19 for 46.3% EM vs $9.18 for 46.0% EM with ReAct
- Recursive LM uses only 3.6 LLM calls per question on average (Hybrid/Dense), far fewer than Self-RAG (10.75) and Planner RAG (8.13)
- Planner RAG significantly underperforms all other architectures (33.7% EM with Dense), despite using 8+ LLM calls per question
- Planner RAG's tree-based planning approach appears to over-decompose questions, leading to higher error accumulation across sub-answers
- Self-RAG underperforms both Vanilla RAG (-4.4% EM) and ReAct RAG (-5.4% EM) despite using ~11 LLM calls per question
- Self-RAG's self-reflection mechanism often skips retrieval (avg 0.84 retrieval calls), which may hurt multi-hop performance where evidence gathering is critical
- BM25 consistently underperforms Dense/Hybrid across all architectures; Dense and Hybrid are closely matched
- Bridge questions remain challenging across all architectures (40-45% EM), while Comparison questions are easier (52-68% EM)

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
| IRCoT | Recursive | ✅ Implemented |
| REAP | Recursive | ✅ Implemented |
| Recursive LM | RLM | ✅ Complete |

## Datasets

- **HotpotQA** (implemented) - Multi-hop QA with bridge/comparison questions
- **MuSiQue** (implemented) - Multi-hop with explicit decomposition
- **2WikiMultiHopQA** (implemented) - Wikipedia-based reasoning

## License

MIT License - see [LICENSE](LICENSE) for details.
