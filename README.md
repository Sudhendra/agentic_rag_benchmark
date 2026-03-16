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

*Model: gpt-4o-mini | Dense retrieval uses text-embedding-3-small | Run ID: bfc8f293*

### Vanilla RAG Baseline (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Cost |
|-----------|-------------|----------|------|
| **Dense** | **12.6%** | **24.3%** | $0.27 |
| Hybrid    | 12.0%       | 23.5%    | $0.27 |
| BM25      | 6.9%        | 16.3%    | $0.26 |

*Model: gpt-4o-mini | Dense retrieval uses text-embedding-3-small | Run ID: e1c01e60*

### Vanilla RAG Baseline (2WikiMultiHopQA Full Validation - 12,576 questions)

| Retriever | Exact Match | F1 Score | Cost |
|-----------|-------------|----------|------|
| **BM25** | **26.0%** | **30.9%** | $1.16 |

*Model: gpt-4o-mini | Run ID: fb0e11a5*

**Note:** 2WikiMultiHopQA full experiments for all architectures pending.

---

### ReAct RAG (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **46.0%** | **59.9%** | 7,611 | $9.18 | 4.05 | 2.66 |
| Dense     | 45.7%       | 59.3%    | 5,950        | $9.66 | 4.07 | 2.69 |
| BM25      | 38.8%       | 50.8%    | 3,923        | $11.16 | 4.56 | 3.38 |

*Model: gpt-4o-mini | max_iterations=7 | concurrency=3 | Run IDs: 25cc3f6b (Hybrid), 47103104 (Dense), 0e7932b0 (BM25)*

### ReAct RAG (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **19.7%** | **27.8%** | 9,374 | $5.24 | 5.57 | 4.67 |
| Hybrid    | 19.3%       | 28.1%    | 8,876        | $5.15 | 5.60 | 4.69 |
| BM25      | 11.9%       | 18.0%    | 11,703       | $5.74 | 6.09 | 5.49 |

*Model: gpt-4o-mini | max_iterations=7 | concurrency=3 | Run IDs: 8d4736d4 (Dense), f34150a9 (Hybrid), 85e4a6b9 (BM25)*

---

### Planner RAG (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **33.7%** | **44.9%** | 4,629 | $4.03 | 8.13 | 2.33 |
| Hybrid    | 33.5%       | 44.5%    | 6,857        | $3.98 | 8.14 | 2.34 |
| BM25      | 27.7%       | 37.7%    | 14,618       | $4.09 | 8.43 | 2.30 |

*Model: gpt-4o-mini | max_iterations=5 | max_branching_factor=2 | concurrency=2 | Run IDs: b4284f7f (Dense), dedaa9b2 (Hybrid), 19114c8b (BM25)*

### Planner RAG (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **15.8%** | **25.9%** | 6,227 | $2.08 | 10.15 | 2.77 |
| Hybrid    | 14.6%       | 24.3%    | 12,642       | $2.03 | 10.20 | 2.75 |
| BM25      | 7.5%        | 15.1%    | 11,760       | $2.07 | 10.81 | 2.59 |

*Model: gpt-4o-mini | max_iterations=5 | max_branching_factor=2 | concurrency=2 | Run IDs: 66516cec (Dense), dc3a1e2e (Hybrid), 6cc7eba1 (BM25)*

---

### Self-RAG (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **40.6%** | **55.0%** | 1,893 | $2.08 | 10.75 | 0.84 |
| Dense     | 40.6%       | 54.9%    | 4,497        | $2.13 | 10.77 | 0.84 |
| BM25      | 37.0%       | 50.4%    | 9,666        | $2.15 | 11.02 | 0.84 |

*Model: gpt-4o-mini | num_candidates=3 | concurrency=2 | Run IDs: 7272b4eb (Hybrid), 72dc70f2 (Dense), e8d57330 (BM25)*

### Self-RAG (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **11.6%** | **23.2%** | 4,290 | $1.00 | 13.79 | 0.95 |
| Hybrid    | 10.8%       | 22.3%    | 977          | $0.95 | 13.57 | 0.95 |
| BM25      | 6.8%        | 16.3%    | 6,630        | $0.98 | 14.10 | 0.95 |

*Model: gpt-4o-mini | num_candidates=3 | concurrency=2 | Run IDs: 1c0a526f (Dense), b6cd2de9 (Hybrid), b666a9d4 (BM25)*

---

### Recursive LM (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **46.3%** | **60.2%** | 7,078 | $3.19 | 3.60 | 2.74 |
| Dense     | 46.1%       | 60.1%    | 4,951        | $3.34 | 3.61 | 2.75 |
| BM25      | 40.1%       | 52.6%    | 9,576        | $5.01 | 5.98 | 4.29 |

*Model: gpt-4o-mini | max_depth=3 | memoization=true | concurrency=2 | Run IDs: 09581743 (Hybrid), a381842d (Dense), 9b4f7587 (BM25)*

### Recursive LM (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **16.7%** | **28.9%** | 8,772 | $2.10 | 7.41 | 5.32 |
| Hybrid    | 16.5%       | 28.6%    | 13,235       | $2.14 | 7.83 | 5.59 |
| BM25      | 9.8%        | 19.8%    | 13,753       | $3.09 | 12.29 | 8.51 |

*Model: gpt-4o-mini | max_depth=3 | memoization=true | concurrency=2 | Run IDs: f01337dd (Dense), 99900bcd (Hybrid), 32f5bafa (BM25)*

---

### IRCoT (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **42.9%** | **59.9%** | 9,874 | $3.81 | 4.16 | 3.58 |
| Dense     | 42.0%       | 59.2%    | 3,278        | $3.77 | 4.17 | 3.59 |
| BM25      | 38.5%       | 54.7%    | 6,463        | $3.97 | 4.20 | 3.64 |

*Model: gpt-4o-mini | max_steps=4 | max_context_tokens=3000 | concurrency=3 | Run IDs: 4d923d09 (Hybrid), 3e4b5fc8 (Dense), 1c4afb94 (BM25)*

### IRCoT (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **21.7%** | **35.9%** | 6,615 | $1.71 | 4.64 | 4.33 |
| Hybrid    | 20.8%       | 34.6%    | 8,389        | $1.73 | 4.64 | 4.32 |
| BM25      | 16.8%       | 28.1%    | 5,824        | $1.68 | 4.61 | 4.28 |

*Model: gpt-4o-mini | max_steps=4 | max_context_tokens=3000 | Run IDs: 51077083 (Dense), 202cc61a (Hybrid), d9578a21 (BM25)*

---

### REAP (HotpotQA Full Validation - 7,405 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **28.1%** | **41.7%** | 4,613 | $6.49 | 6.24 | 1.97 |
| Hybrid    | 27.3%       | 41.1%    | 3,777        | $6.54 | 6.40 | 2.06 |
| BM25      | 24.6%       | 36.6%    | 14,713       | $7.41 | 7.93 | 3.05 |

*Model: gpt-4o-mini | max_iterations=5 | max_active_requirements=2 | concurrency=2 | Run IDs: c4da1615 (Dense), f667025a (Hybrid), 0eb19318 (BM25)*

### REAP (MuSiQue Full Validation - 2,417 questions)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **7.0%** | **15.7%** | 4,751 | $2.14 | 6.93 | 2.29 |
| Dense     | 7.0%       | 15.6%    | 5,230        | $2.17 | 6.91 | 2.27 |
| BM25      | 4.5%       | 11.6%    | 12,098       | $2.23 | 7.70 | 2.74 |

*Model: gpt-4o-mini | max_iterations=5 | max_active_requirements=2 | Run IDs: 842f7d8b (Hybrid), 37b58fa1 (Dense), 26863c88 (BM25)*

---

### Cross-Architecture Comparison (HotpotQA Best Retriever per Architecture)

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost |
|--------------|------|----------------|-------------|----------|---------------|------|
| Vanilla RAG  | Baseline | Dense    | 45.0%       | 59.5%    | 1.0           | $0.79 |
| **Recursive LM** | **RLM** | **Hybrid** | **46.3%** | **60.2%** | **3.60** | **$3.19** |
| ReAct RAG | Agentic | Hybrid | 46.0% | 59.9% | 4.05 | $9.18 |
| IRCoT | Recursive | Hybrid | 42.9% | 59.9% | 4.16 | $3.81 |
| Self-RAG     | Agentic  | Hybrid   | 40.6%       | 55.0%    | 10.75         | $2.08 |
| Planner RAG  | Agentic  | Dense    | 33.7%       | 44.9%    | 8.13          | $4.03 |
| REAP | Recursive | Dense | 28.1% | 41.7% | 6.24 | $6.49 |

**Key Findings:**
- **Recursive LM leads on HotpotQA** with 46.3% EM and 60.2% F1, closely followed by ReAct RAG (46.0% EM)
- Recursive LM is the most cost-efficient: $3.19 for 46.3% EM vs ReAct's $9.18 for 46.0% EM
- IRCoT achieves 42.9% EM with similar cost to RLM ($3.81), making it a competitive recursive alternative
- Planner RAG significantly underperforms all other architectures (33.7% EM), despite using 8+ LLM calls
- Self-RAG underperforms Vanilla RAG (-4.4% EM) due to low retrieval usage (avg 0.84 calls)
- REAP is the worst architecture on HotpotQA (28.1% EM)

---

### Cross-Architecture Comparison (MuSiQue Best Retriever per Architecture)

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost |
|--------------|------|----------------|-------------|----------|---------------|------|
| Vanilla RAG  | Baseline | Dense    | 12.6%       | 24.3%    | 1.0           | $0.27 |
| **IRCoT** | **Recursive** | **Dense** | **21.7%** | **35.9%** | **4.64** | **$1.71** |
| ReAct RAG | Agentic | Dense | 19.7% | 27.8% | 5.57 | $5.24 |
| Recursive LM | RLM | Dense | 16.7% | 28.9% | 7.41 | $2.10 |
| Planner RAG  | Agentic  | Dense    | 15.8%       | 25.9%    | 10.15         | $2.08 |
| Self-RAG     | Agentic  | Dense   | 11.6%       | 23.2%    | 13.79         | $1.00 |
| REAP | Recursive | Hybrid | 7.0% | 15.7% | 6.93 | $2.14 |

**Key Findings:**
- **IRCoT leads on MuSiQue** with 21.7% EM, outperforming all other architectures
- ReAct RAG is second best at 19.7% EM, followed by Recursive LM (16.7% EM)
- IRCoT is also the most cost-efficient on MuSiQue ($1.71 for 21.7% EM)
- REAP significantly underperforms on MuSiQue (7.0% EM) - worst by far
- All architectures except REAP outperform the Vanilla RAG baseline on MuSiQue

---

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
- **2WikiMultiHopQA** (partially implemented) - Baseline complete, all architectures pending

## Total Experiment Cost

| Dataset | Questions | Cost |
|---------|-----------|------|
| HotpotQA | 7,405 | $97.27 |
| MuSiQue | 2,417 | $45.02 |
| 2WikiMultiHopQA | 12,576 | $1.16 |
| **Total** | **22,398** | **$143.46** |

## License

MIT License - see [LICENSE](LICENSE) for details.
