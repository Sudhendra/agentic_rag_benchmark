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

### ReAct RAG (HotpotQA Full Validation)

Results pending. Run the ReAct configs in `configs/react_*_full.yaml` to populate this table.

### Self-RAG (HotpotQA Full Validation)

Results pending. Run the Self-RAG configs in `configs/self_rag_*_full.yaml` to populate this table.

| Retriever | Exact Match | F1 Score | Cost | LLM Calls/Q |
|-----------|-------------|----------|------|-------------|
| BM25      | pending     | pending  | pending | ~7       |
| Dense     | pending     | pending  | pending | ~7       |
| Hybrid    | pending     | pending  | pending | ~7       |

*Model: gpt-4o-mini | 3 candidates per question | ~$0.014/question*

### Planner RAG (HotpotQA Full Validation)

Results pending. Full validation is intentionally deferred until after merge to `dev`; run the Planner configs in `configs/planner_*_full.yaml` to populate this table.

Development subset check (HotpotQA validation, `subset=20`, gpt-4o-mini):

| Retriever | Exact Match | F1 Score | Cost | LLM Calls/Q |
|-----------|-------------|----------|------|-------------|
| BM25      | 40.0%       | 51.7%    | $0.0104 | 9.20 |
| Dense     | 50.0%       | 61.4%    | $0.0100 | 8.70 |
| Hybrid    | 45.0%       | 56.7%    | $0.0103 | 9.15 |

Planner implementation is complete and unit-tested; post-MVP optimization backlog:

- Add semantic/embedding-based sibling diversification beyond lexical similarity pruning.
- Add sentence-level supporting fact traceability for richer analysis.
- Explore parallel sibling-node solving to reduce end-to-end latency.

### By Question Type (Dense Retriever)

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 39.6% | 55.4% |
| Comparison | 1,487 | 66.3% | 75.9% |

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
| ReAct RAG | Agentic | ✅ Implemented (results pending) |
| Self-RAG | Agentic | ✅ Implemented (results pending) |
| Planner RAG | Agentic | ✅ Implemented (results pending) |
| IRCoT | Recursive | 🔲 Planned |
| REAP | Recursive | 🔲 Planned |
| Recursive LM | RLM | 🔲 Planned |

## Datasets

- **HotpotQA** (implemented) - Multi-hop QA with bridge/comparison questions
- **MuSiQue** (planned) - Multi-hop with explicit decomposition
- **2WikiMultiHopQA** (planned) - Wikipedia-based reasoning

## License

MIT License - see [LICENSE](LICENSE) for details.
