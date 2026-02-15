# Next Steps: Agentic RAG Benchmark

**Date:** February 15, 2026  
**Status:** Phase 2 In Progress - ReAct RAG Results Available, Self-RAG Pending  
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
| ReAct RAG | `src/architectures/agentic/react_rag.py` | ✅ Complete |
| Self-RAG | `src/architectures/agentic/self_rag.py` | ✅ Implemented (results pending) |
| Architecture Factory | `src/architectures/factory.py` | ✅ Complete |
| HotpotQA Loader | `src/data/hotpotqa.py` | ✅ Complete |
| Metrics (EM, F1) | `src/evaluation/metrics.py` | ✅ Complete |
| Evaluator | `src/evaluation/evaluator.py` | ✅ Complete (with progress logging) |
| SQLite Cache | `src/utils/cache.py` | ✅ Complete |
| Config Loader | `src/utils/config.py` | ✅ Complete |
| Results Saver | `src/utils/results.py` | ✅ Complete |
| Logging | `src/utils/logging.py` | ✅ Complete |
| Experiment Runner | `scripts/run_experiment.py` | ✅ Complete |
| MLflow Integration | `scripts/run_experiment.py` | ✅ Complete |
| Configs | All vanilla, react, self_rag configs (100-sample + full) | ✅ Complete |
| Unit Tests | `tests/test_*.py` (127 tests) | ✅ Complete |
| Analysis Script | `scripts/analyze_results.py` | ✅ Complete |

---

## Benchmark Results

### Vanilla RAG - Full Validation Set (7,405 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost |
|-----------|-------------|----------|--------------|------|
| **Dense** | **45.0%** | **59.5%** | 1,411 | $0.79 |
| Hybrid    | 44.1%       | 58.6%    | 2,127        | $0.76 |
| BM25      | 38.2%       | 51.5%    | 2,321        | $0.77 |

**Breakdown by Question Type (Dense, Vanilla RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 39.6% | 55.4% |
| Comparison | 1,487 | 66.3% | 75.9% |

**Recommendation:** Dense retrieval as the default for Vanilla RAG baseline.

---

### ReAct RAG - Full Validation Set (7,405 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **46.0%** | **59.9%** | 7,611 | $9.18 | 4.05 | 2.66 |
| Dense     | 45.7%       | 59.3%    | 5,950        | $9.66 | 4.07 | 2.69 |
| BM25      | 38.8%       | 50.8%    | 3,923        | $11.16 | 4.56 | 3.38 |

*Configuration: max_iterations=7, top_k=5, concurrency=3*

**Breakdown by Question Type (Hybrid, ReAct RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 44.5% | 59.3% |
| Comparison | 1,487 | 52.2% | 62.3% |

**Breakdown by Question Type (Dense, ReAct RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 44.2% | 58.6% |
| Comparison | 1,487 | 51.8% | 62.0% |

**Breakdown by Question Type (BM25, ReAct RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 37.9% | 50.5% |
| Comparison | 1,487 | 42.5% | 52.1% |

**Key Findings:**
- Hybrid retrieval narrowly edges out Dense (+0.3% EM, +0.6% F1) as best retriever for ReAct
- BM25 lags significantly behind Dense/Hybrid (-7% EM, -9% F1)
- BM25 requires more iterations (4.56 LLM calls, 3.38 retrievals) than Dense/Hybrid (~4.05 LLM calls, ~2.67 retrievals), suggesting weaker initial retrieval drives more search attempts
- ReAct narrows the Bridge vs Comparison gap: only +7.7% EM difference (vs +27% for Vanilla RAG)
- Comparison questions are less dominant with ReAct (52.2% EM) vs Vanilla RAG (66.3% EM), indicating the iterative approach may over-refine already-answerable comparison questions

**Recommendation:** Hybrid retrieval as default for ReAct RAG.

---

### Vanilla RAG vs ReAct RAG Comparison

| Retriever | Vanilla EM | ReAct EM | Delta EM | Vanilla F1 | ReAct F1 | Delta F1 | Cost Ratio |
|-----------|-----------|----------|----------|-----------|----------|----------|------------|
| BM25      | 38.2%     | 38.8%    | +0.6%    | 51.5%     | 50.8%    | -0.7%    | 14.5x |
| Dense     | 45.0%     | 45.7%    | +0.7%    | 59.5%     | 59.3%    | -0.2%    | 12.2x |
| Hybrid    | 44.1%     | 46.0%    | +1.9%    | 58.6%     | 59.9%    | +1.3%    | 12.0x |

**Key Observations:**
- ReAct provides marginal EM improvements (+0.6% to +1.9%) over Vanilla RAG
- F1 improvements are negligible or slightly negative for BM25/Dense, slightly positive for Hybrid
- Cost increases dramatically: 12-15x more expensive per run
- Hybrid retrieval benefits most from ReAct's iterative approach
- The iterative retrieval helps most when initial retrieval quality is moderate (Hybrid), but provides diminishing returns with already-strong retrieval (Dense)
- For BM25, ReAct improves EM but actually hurts F1, likely due to the model accumulating noisy context from weak BM25 results across iterations

**Question Type Comparison (Best Retriever per Architecture):**

| Type | Vanilla EM (Dense) | ReAct EM (Hybrid) | Delta |
|------|-------------------|-------------------|-------|
| Bridge | 39.6% | 44.5% | +4.9% |
| Comparison | 66.3% | 52.2% | -14.1% |

- ReAct substantially improves Bridge questions (+4.9% EM) which require multi-hop reasoning
- ReAct substantially degrades Comparison questions (-14.1% EM), possibly because iterative retrieval adds noise to questions that are straightforward with good context

---

## Remaining Tasks

### Priority 1: Self-RAG Full Evaluation (High Impact)

Run Self-RAG with all three retrievers on the full validation set:

```bash
# 1. Self-RAG + BM25
python scripts/run_experiment.py --config configs/self_rag_bm25_full.yaml

# 2. Self-RAG + Dense
python scripts/run_experiment.py --config configs/self_rag_dense_full.yaml

# 3. Self-RAG + Hybrid
python scripts/run_experiment.py --config configs/self_rag_hybrid_full.yaml
```

After each run, analyze:
```bash
python scripts/analyze_results.py --results results/<run_id> --breakdown --errors
```

### Priority 2: Anthropic Client Implementation

**File:** `src/core/llm_client.py`

Implement `AnthropicClient` to enable running all architectures with Claude models for cross-model comparison.

### Priority 3: Cross-Architecture Comparison

After all 9 full runs complete (3 Vanilla + 3 ReAct + 3 Self-RAG):
```bash
python scripts/analyze_results.py --results results --compare
```

Update documentation with master comparison table.

### Priority 4: Additional Datasets

- MuSiQue - Multi-hop with explicit decomposition
- 2WikiMultiHopQA - Wikipedia-based reasoning

### Priority 5: Remaining Architectures

- Planner RAG (Agentic)
- IRCoT (Recursive)
- REAP (Recursive)
- Recursive LM (RLM)

---

## Completed Results Summary

| Run ID | Architecture | Retriever | Questions | EM | F1 | Cost |
|--------|-------------|-----------|-----------|-----|-----|------|
| `74d7b162` | vanilla_rag | bm25 | 7,405 | 38.2% | 51.5% | $0.77 |
| `bfc8f293` | vanilla_rag | dense | 7,405 | 45.0% | 59.5% | $0.79 |
| `b054e24b` | vanilla_rag | hybrid | 7,405 | 44.1% | 58.6% | $0.76 |
| `0e7932b0` | react_rag | bm25 | 7,405 | 38.8% | 50.8% | $11.16 |
| `47103104` | react_rag | dense | 7,405 | 45.7% | 59.3% | $9.66 |
| `25cc3f6b` | react_rag | hybrid | 7,405 | 46.0% | 59.9% | $9.18 |

**Total cost so far:** ~$32.32 (Vanilla: $2.32, ReAct: $30.00)

---

## Infrastructure Improvements Made

1. **ReAct prompt rewrite** - Added few-shot examples to `prompts/react.txt` for proper tool use
2. **Retry hardening** - Extended `llm_client.py` retry to handle 403/PermissionDenied, connection errors, 500s; 5 attempts with longer backoff
3. **Progress logging** - Added real-time progress output to `evaluator.py` (count, %, rate, ETA every 50 questions)
4. **Concurrency tuning** - Reduced `max_concurrency` to 3 for ReAct configs to avoid rate limits

---

## Questions for Team Discussion

1. Is the marginal ReAct improvement (+1% EM) worth the 12x cost increase? How to frame this in the paper?
2. Should we add retrieval quality metrics (recall@k, precision@k) as an additional analysis?
3. Do we need fullwiki setting for HotpotQA, or is distractor sufficient for the paper?
4. Should we add a cost budget limit to the experiment runner?
