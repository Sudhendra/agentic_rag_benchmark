# Next Steps: Agentic RAG Benchmark

**Date:** February 16, 2026  
**Status:** Phase 2 Complete - Vanilla RAG, ReAct RAG, Self-RAG Results Available  
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
| Self-RAG | `src/architectures/agentic/self_rag.py` | ✅ Complete |
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

### Self-RAG - Full Validation Set (7,405 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **40.6%** | **55.0%** | 1,893 | $2.08 | 10.75 | 0.84 |
| Dense     | 40.6%       | 54.9%    | 4,497        | $2.13 | 10.77 | 0.84 |
| BM25      | 37.0%       | 50.4%    | 9,666        | $2.15 | 11.02 | 0.84 |

*Configuration: num_candidates=3, top_k=5, concurrency=2*

**Breakdown by Question Type (Hybrid, Self-RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 36.0% | 51.5% |
| Comparison | 1,487 | 59.0% | 68.7% |

**Breakdown by Question Type (Dense, Self-RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 35.9% | 51.3% |
| Comparison | 1,487 | 59.7% | 69.1% |

**Breakdown by Question Type (BM25, Self-RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 32.1% | 46.5% |
| Comparison | 1,487 | 56.4% | 66.1% |

**Key Findings:**
- Hybrid and Dense are nearly identical (40.6% EM both); BM25 lags by -3.6% EM
- Self-RAG uses ~11 LLM calls per question but only ~0.84 retrieval calls, meaning the self-reflection mechanism frequently skips retrieval
- Low retrieval usage likely hurts multi-hop performance where gathering evidence from multiple documents is critical
- Self-RAG is the fastest architecture with Hybrid retrieval (1,893ms), benefiting heavily from cached LLM calls
- Comparison questions (59.0% EM) are significantly easier than Bridge questions (36.0% EM), consistent with other architectures

**Recommendation:** Hybrid retrieval as default for Self-RAG (best F1, lowest latency, lowest cost).

---

### Vanilla RAG vs ReAct RAG vs Self-RAG Comparison

**Best Retriever per Architecture:**

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost |
|--------------|------|----------------|-------------|----------|---------------|------|
| Vanilla RAG  | Baseline | Dense    | 45.0%       | 59.5%    | 1.0           | $0.79 |
| **ReAct RAG** | **Agentic** | **Hybrid** | **46.0%** | **59.9%** | **4.05** | **$9.18** |
| Self-RAG     | Agentic  | Hybrid   | 40.6%       | 55.0%    | 10.75         | $2.08 |

**Per-Retriever Delta (vs Vanilla RAG Baseline):**

| Retriever | ReAct EM Delta | ReAct F1 Delta | Self-RAG EM Delta | Self-RAG F1 Delta |
|-----------|---------------|---------------|-------------------|-------------------|
| BM25      | +0.6%         | -0.7%         | -1.2%             | -1.1%             |
| Dense     | +0.7%         | -0.2%         | -4.4%             | -4.6%             |
| Hybrid    | +1.9%         | +1.3%         | -3.5%             | -3.6%             |

**Question Type Comparison (Best Retriever per Architecture):**

| Type | Vanilla EM (Dense) | ReAct EM (Hybrid) | Self-RAG EM (Hybrid) |
|------|-------------------|-------------------|----------------------|
| Bridge | 39.6% | 44.5% | 36.0% |
| Comparison | 66.3% | 52.2% | 59.0% |

**Key Observations:**
- Self-RAG is the weakest of the three architectures on accuracy, underperforming even single-pass Vanilla RAG
- Self-RAG's self-reflection approach (generate-then-critique) does not compensate for its low retrieval utilization (~0.84 calls vs ReAct's ~2.7 calls)
- ReAct remains the only architecture to improve over Vanilla RAG, and only with Hybrid retrieval (+1.9% EM)
- Self-RAG's cost ($2.08) falls between Vanilla ($0.79) and ReAct ($9.18), but its accuracy does not justify even this moderate cost increase
- On Bridge questions (multi-hop), Self-RAG is the worst performer (36.0% EM), suggesting that self-reflection without iterative retrieval is insufficient for evidence gathering
- On Comparison questions, Self-RAG (59.0%) recovers closer to Vanilla (66.3%) than ReAct (52.2%), indicating the reflection mechanism helps with straightforward comparisons but not multi-hop reasoning

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

### Priority 1: Anthropic Client Implementation

**File:** `src/core/llm_client.py`

Implement `AnthropicClient` to enable running all architectures with Claude models for cross-model comparison.

### Priority 2: Cross-Architecture Comparison

After adding Anthropic model runs for cross-model comparison:
```bash
python scripts/analyze_results.py --results results --compare
```

Note: 9 full runs complete (3 Vanilla + 3 ReAct + 3 Self-RAG) with gpt-4o-mini. Cross-architecture comparison tables have been added above. Next comparison milestone is cross-model (OpenAI vs Anthropic).

### Priority 3: Additional Datasets

- MuSiQue - Multi-hop with explicit decomposition
- 2WikiMultiHopQA - Wikipedia-based reasoning

### Priority 4: Remaining Architectures

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
| `e8d57330` | self_rag | bm25 | 7,405 | 37.0% | 50.4% | $2.15 |
| `72dc70f2` | self_rag | dense | 7,405 | 40.6% | 54.9% | $2.13 |
| `7272b4eb` | self_rag | hybrid | 7,405 | 40.6% | 55.0% | $2.08 |

**Total cost so far:** ~$38.68 (Vanilla: $2.32, ReAct: $30.00, Self-RAG: $6.36)

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
5. Self-RAG underperforms Vanilla RAG despite ~11 LLM calls per question. Is the low retrieval rate (0.84 calls) the primary cause? Should we experiment with forcing retrieval (disabling the "no retrieval needed" reflection token)?
6. How should we frame the Self-RAG results in the paper -- as evidence that self-reflection without sufficient retrieval is counterproductive for multi-hop QA?
