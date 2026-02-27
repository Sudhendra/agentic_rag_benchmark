# Next Steps: Agentic RAG Benchmark

**Date:** February 24, 2026  
**Status:** Phase 2+ Complete - Vanilla RAG, ReAct RAG, Self-RAG Full Results; Planner RAG, Recursive LM Subset Results Available  
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
| Recursive LM | `src/architectures/rlm/recursive_lm.py` | ✅ Complete |
| RLM Configs | `configs/rlm*.yaml` (BM25, Dense, Hybrid) | ✅ Complete |
| RLM Prompts | `prompts/rlm.txt`, `prompts/rlm_combine.txt` | ✅ Complete |
| RLM Tests | `tests/test_recursive_lm.py` (16 tests) | ✅ Complete |

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

### Planner RAG - Full Validation Set (7,405 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **33.7%** | **44.9%** | 4,629 | $4.03 | 8.13 | 2.33 |
| Hybrid    | 33.5%       | 44.5%    | 6,857        | $3.98 | 8.14 | 2.34 |
| BM25      | 27.7%       | 37.7%    | 14,618       | $4.09 | 8.43 | 2.30 |

*Configuration: max_iterations=5, max_branching_factor=2, top_k=5, concurrency=2*

**Breakdown by Question Type (Dense, Planner RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 33.9% | 47.7% |
| Comparison | 1,487 | 32.8% | 33.9% |

**Breakdown by Question Type (Hybrid, Planner RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 34.3% | 47.9% |
| Comparison | 1,487 | 30.5% | 31.3% |

**Breakdown by Question Type (BM25, Planner RAG):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 28.1% | 40.4% |
| Comparison | 1,487 | 26.1% | 26.8% |

**Key Findings:**
- Planner RAG significantly underperforms all other architectures on full validation (best: 33.7% EM with Dense)
- Dense and Hybrid are nearly tied (33.7% vs 33.5% EM); BM25 lags significantly (27.7% EM)
- Planner RAG uses ~8 LLM calls and ~2.3 retrieval calls per question
- Despite higher retrieval usage than Self-RAG, performance is worse, suggesting tree-based planning introduces error accumulation
- Bridge and Comparison questions perform nearly identically (33.9% vs 32.8% EM with Dense), unlike other architectures where Comparison >> Bridge

**Recommendation:** Do not use Planner RAG for HotpotQA. The tree-based planning approach appears to over-decompose questions, leading to worse accuracy than even single-pass retrieval.

---

### Recursive LM - Validation Subset (100 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **BM25** | **52.0%** | 63.1% | 3,033 | $0.042 | 3.6 | 2.8 |
| Hybrid    | 51.0%       | **67.0%** | 3,265 | $0.038 | 3.1 | 2.4 |
| Dense     | 49.0%       | 65.2%    | 3,537        | $0.038 | 3.1 | 2.4 |

*Configuration: max_depth=3, memoization=true, top_k=5, concurrency=5*

**Breakdown by Question Type (BM25, Recursive LM):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 79 | 46.8% | 58.9% |
| Comparison | 21 | 71.4% | 78.8% |

**Breakdown by Question Type (Hybrid, Recursive LM):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 79 | 48.1% | 64.5% |
| Comparison | 21 | 61.9% | 76.5% |

**Key Findings:**
- RLM shows strong subset results: 52.0% EM (BM25), 67.0% F1 (Hybrid)
- Extremely cost-efficient: ~$0.0004/question vs $0.0012 (ReAct) and $0.0003 (Vanilla)
- Low LLM overhead: 3.1-3.6 calls per question, indicating many questions answered directly without decomposition
- BM25 surprisingly leads on EM (52.0%) while Hybrid leads on F1 (67.0%); retrievers closely matched
- Comparison questions significantly easier (71.4% EM) than Bridge (46.8% EM), consistent with other architectures
- Critical prompt engineering finding: v1 prompts without explicit "short extractive" instruction yielded 9-14% EM; adding formatting guidance boosted to 49-52% EM (3-5x improvement, zero architecture changes)

**Recommendation:** Full validation needed to confirm subset results. BM25 for best EM, Hybrid for best F1.

---

### Vanilla RAG vs ReAct RAG vs Self-RAG vs Recursive LM Comparison

**Best Retriever per Architecture:**

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost |
|--------------|------|----------------|-------------|----------|---------------|------|
| Vanilla RAG  | Baseline | Dense    | 45.0%       | 59.5%    | 1.0           | $0.79 |
| **ReAct RAG** | **Agentic** | **Hybrid** | **46.0%** | **59.9%** | **4.05** | **$9.18** |
| Self-RAG     | Agentic  | Hybrid   | 40.6%       | 55.0%    | 10.75         | $2.08 |
| Planner RAG  | Agentic  | Dense    | 33.7%       | 44.9%    | 8.13          | $4.03 |
| Recursive LM | RLM | BM25 | 52.0%* | 63.1%* | 3.6 | $0.042* |

*\* Subset results (100 questions) — not directly comparable to full validation runs (7,405 questions).*

**Per-Retriever Delta (vs Vanilla RAG Baseline):**

| Retriever | ReAct EM Delta | ReAct F1 Delta | Self-RAG EM Delta | Self-RAG F1 Delta | Planner EM Delta | Planner F1 Delta |
|-----------|---------------|---------------|-------------------|-------------------|------------------|-------------------|
| BM25      | +0.6%         | -0.7%         | -1.2%             | -1.1%             | -10.5%           | -13.8%            |
| Dense     | +0.7%         | -0.2%         | -4.4%             | -4.6%             | -11.3%           | -14.6%            |
| Hybrid    | +1.9%         | +1.3%         | -3.5%             | -3.6%             | -10.6%           | -14.1%            |

**Question Type Comparison (Best Retriever per Architecture):**

| Type | Vanilla EM (Dense) | ReAct EM (Hybrid) | Self-RAG EM (Hybrid) | Planner EM (Dense) |
|------|-------------------|-------------------|----------------------|-------------------|
| Bridge | 39.6% | 44.5% | 36.0% | 33.9% |
| Comparison | 66.3% | 52.2% | 59.0% | 32.8% |

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

Note: 12 full runs complete (3 Vanilla + 3 ReAct + 3 Self-RAG + 3 Planner RAG) with gpt-4o-mini. RLM subset results available (100 questions). Full validation run for RLM pending. Next comparison milestone is cross-model (OpenAI vs Anthropic).

### Priority 3: Additional Datasets

- MuSiQue - Multi-hop with explicit decomposition
- 2WikiMultiHopQA - Wikipedia-based reasoning

### Priority 4: Remaining Architectures

- ~~Planner RAG (Agentic)~~ ✅ Implemented
- IRCoT (Recursive)
- REAP (Recursive)
- ~~Recursive LM (RLM)~~ ✅ Implemented

### Priority 5: Full Validation Runs

Run full validation (7,405 questions) for Recursive LM:
```bash
# Create full configs and run
python scripts/run_experiment.py --config configs/rlm_bm25_full.yaml
python scripts/run_experiment.py --config configs/rlm_dense_full.yaml
python scripts/run_experiment.py --config configs/rlm_hybrid_full.yaml
```

Note: Planner RAG full validation runs complete. RLM full validation pending.

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
| `19114c8b` | planner_rag | bm25 | 7,405 | 27.7% | 37.7% | $4.09 |
| `b4284f7f` | planner_rag | dense | 7,405 | 33.7% | 44.9% | $4.03 |
| `dedaa9b2` | planner_rag | hybrid | 7,405 | 33.5% | 44.5% | $3.98 |
| `4c86dc85` | recursive_lm | bm25 | 100 | 52.0% | 63.1% | $0.042 |
| `9bb46d75` | recursive_lm | dense | 100 | 49.0% | 65.2% | $0.038 |
| `f8cac4cf` | recursive_lm | hybrid | 100 | 51.0% | 67.0% | $0.038 |

**Total cost so far:** ~$50.99 (Vanilla: $2.32, ReAct: $30.00, Self-RAG: $6.36, Planner: $12.10, RLM: $0.12)

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
7. RLM subset results (52% EM) look strong, but subset-to-full comparisons are unreliable. Should we prioritize the full RLM validation run next, or focus on implementing IRCoT/REAP first to complete the Recursive paradigm column?
8. RLM prompt sensitivity was extreme (9% -> 52% EM from wording changes alone). Should we include a prompt sensitivity analysis section in the paper?
9. Planner RAG severely underperforms all architectures (33.7% EM vs 45% Vanilla, 46% ReAct). Is this an implementation issue, or does tree-based planning fundamentally not work for HotpotQA-style multi-hop questions? Should we debug further or deprioritize Planner RAG for the paper?
