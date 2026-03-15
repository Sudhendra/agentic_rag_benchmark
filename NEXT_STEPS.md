# Next Steps: Agentic RAG Benchmark

**Date:** March 3, 2026  
**Status:** Phase 3 Complete - All Architectures (Vanilla, ReAct, Planner, Self-RAG, Recursive LM, IRCoT, REAP) Full HotpotQA and MuSiQue Results Available; 2WikiMultiHopQA Pending
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

### Recursive LM - Full Validation Set (7,405 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Hybrid** | **46.3%** | **60.2%** | 7,078 | $3.19 | 3.60 | 2.74 |
| Dense     | 46.1%       | 60.1%    | 4,951        | $3.34 | 3.61 | 2.75 |
| BM25      | 40.1%       | 52.6%    | 9,576        | $5.01 | 5.98 | 4.29 |

*Configuration: max_depth=3, memoization=true, top_k=5, concurrency=2*

**Breakdown by Question Type (Hybrid, Recursive LM):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 40.7% | 55.9% |
| Comparison | 1,487 | 67.8% | 77.0% |

**Breakdown by Question Type (Dense, Recursive LM):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 40.7% | 55.9% |
| Comparison | 1,487 | 67.8% | 77.0% |

**Breakdown by Question Type (BM25, Recursive LM):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | 5,918 | 35.4% | 49.2% |
| Comparison | 1,487 | 58.6% | 66.3% |

**Key Findings:**
- RLM with Hybrid retrieval achieves 46.3% EM and 60.2% F1 on full validation, tying ReAct for best EM while being 4x cheaper ($3.19 vs $9.18)
- RLM is the most cost-efficient agentic architecture: $3.19 for 46.3% EM vs ReAct's $9.18 for 46.0% EM
- RLM uses only 3.6 LLM calls per question on average (Hybrid/Dense), far fewer than Self-RAG (10.75) and Planner RAG (8.13)
- BM25 with RLM requires significantly more LLM calls (5.98) and retrievals (4.29) than Dense/Hybrid (~3.6 LLM, ~2.7 retrieval), similar pattern to other architectures
- Comparison questions (67.8% EM) are significantly easier than Bridge questions (40.7% EM), consistent with other architectures
- Critical prompt engineering finding: v1 prompts without explicit "short extractive" instruction yielded 9-14% EM; adding formatting guidance boosted to 46% EM (5x improvement)

**Recommendation:** RLM with Hybrid retrieval is recommended for best cost-efficiency among agentic architectures.

---

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
- Dense retrieval is the best retriever for all architectures on MuSiQue

---

### Vanilla RAG vs ReAct RAG vs Self-RAG vs Recursive LM Comparison (HotpotQA)

**Best Retriever per Architecture:**

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost |
|--------------|------|----------------|-------------|----------|---------------|------|
| Vanilla RAG  | Baseline | Dense    | 45.0%       | 59.5%    | 1.0           | $0.79 |
| **ReAct RAG** | **Agentic** | **Hybrid** | **46.0%** | **59.9%** | **4.05** | **$9.18** |
| Recursive LM | RLM | Hybrid | 46.3% | 60.2% | 3.60 | $3.19 |
| Self-RAG     | Agentic  | Hybrid   | 40.6%       | 55.0%    | 10.75         | $2.08 |
| Planner RAG  | Agentic  | Dense    | 33.7%       | 44.9%    | 8.13          | $4.03 |

*\* Subset results (100 questions) — not directly comparable to full validation runs (7,405 questions).*

**Per-Retriever Delta (vs Vanilla RAG Baseline):**

| Retriever | ReAct EM Delta | ReAct F1 Delta | Self-RAG EM Delta | Self-RAG F1 Delta | Planner EM Delta | Planner F1 Delta |
|-----------|---------------|---------------|-------------------|-------------------|------------------|-------------------|
| BM25      | +0.6%         | -0.7%         | -1.2%             | -1.1%             | -10.5%           | -13.8%            |
| Dense     | +0.7%         | -0.2%         | -4.4%             | -4.6%             | -11.3%           | -14.6%            |
| Hybrid    | +1.9%         | +1.3%         | -3.5%             | -3.6%             | -10.6%           | -14.1%            |

**Question Type Comparison (Best Retriever per Architecture):**

| Type | Vanilla EM (Dense) | ReAct EM (Hybrid) | RLM EM (Hybrid) | Self-RAG EM (Hybrid) | Planner EM (Dense) |
|------|-------------------|-------------------|-----------------|----------------------|-------------------|
| Bridge | 39.6% | 44.5% | 40.7% | 36.0% | 33.9% |
| Comparison | 66.3% | 52.2% | 67.8% | 59.0% | 32.8% |

**Key Observations:**
- ReAct RAG and Recursive LM are tied for best EM (46.0% vs 46.3%), but RLM achieves this at ~4x lower cost ($3.19 vs $9.18)
- RLM achieves the best F1 (60.2%) among all architectures, edging out ReAct (59.9%)
- RLM is the most cost-efficient agentic architecture: $3.19 for 46.3% EM vs ReAct's $9.18 for 46.0% EM
- Self-RAG underperforms Vanilla RAG (-4.4% EM) despite using ~11 LLM calls per question
- Planner RAG is the worst architecture (33.7% EM), significantly underperforming Vanilla RAG (-11.3% EM)
- On Bridge questions (multi-hop), ReAct leads (44.5% EM), followed by RLM (40.7%), then Vanilla (39.6%)
- On Comparison questions, RLM leads (67.8% EM), nearly matching Vanilla (66.3%), while ReAct degrades (52.2%)

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

### Vanilla RAG - MuSiQue Full Validation Set (2,417 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost |
|-----------|-------------|----------|--------------|------|
| **Dense** | **12.6%** | **24.3%** | 1,089 | $0.27 |
| Hybrid    | 12.0%       | 23.5%    | 1,484        | $0.27 |
| BM25      | 6.9%        | 16.3%    | 1,251        | $0.26 |

**Breakdown by Question Type (Dense, Vanilla RAG on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 17.1% | 29.6% |
| Compositional | ~300 | 7.7% | 18.6% |

**Key Findings:**
- MuSiQue is significantly harder than HotpotQA: 12.6% EM vs 45.0% EM for Vanilla RAG
- Dense retrieval remains the best retriever for MuSiQue (+5.7% EM over BM25)
- Compositional questions (7.7% EM) are much harder than Bridge questions (17.1% EM)
- The explicit decomposition in MuSiQue appears to require more sophisticated reasoning than single-pass retrieval can handle

---

### ReAct RAG - MuSiQue Full Validation Set (2,417 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **19.7%** | **27.8%** | 9,374 | $5.24 | 5.57 | 4.67 |
| Hybrid    | 19.3%       | 28.1%    | 8,876        | $5.15 | 5.60 | 4.69 |
| BM25      | 11.9%       | 18.0%    | 11,703       | $5.74 | 6.09 | 5.49 |

*Configuration: max_iterations=7, top_k=5, concurrency=3*

**Breakdown by Question Type (Dense, ReAct RAG on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 23.6% | 33.9% |
| Compositional | ~300 | 15.5% | 21.2% |

**Key Findings:**
- ReAct RAG significantly improves over Vanilla RAG on MuSiQue: +7.1% EM (19.7% vs 12.6%)
- ReAct provides larger gains on MuSiQue than HotpotQA (+7.1% vs +0.7%), showing iterative retrieval helps more on harder datasets
- BM25 with ReAct requires more iterations (6.09 LLM calls, 5.49 retrievals) vs Dense/Hybrid (~5.6 LLM, ~4.7 retrievals)
- Bridge questions (23.6% EM) remain harder than Compositional (15.5% EM), similar to Vanilla RAG pattern
- Cost is ~19x higher than Vanilla RAG ($5.24 vs $0.27) but provides substantial accuracy gains

---

### Planner RAG - MuSiQue Full Validation Set (2,417 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **15.8%** | **25.9%** | 6,227 | $2.08 | 10.15 | 2.77 |
| Hybrid    | 14.6%       | 24.3%    | 12,642       | $2.03 | 10.20 | 2.75 |
| BM25      | 7.5%        | 15.1%    | 11,760       | $2.07 | 10.81 | 2.59 |

*Configuration: max_iterations=5, max_branching_factor=2, top_k=5, concurrency=2*

**Breakdown by Question Type (Dense, Planner RAG on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 21.1% | 33.0% |
| Compositional | ~300 | 10.1% | 18.2% |

**Key Findings:**
- Planner RAG outperforms Vanilla RAG on MuSiQue: +3.2% EM (15.8% vs 12.6%), unlike HotpotQA where it underperformed
- Planner RAG underperforms ReAct RAG on MuSiQue: -3.9% EM (15.8% vs 19.7%)
- Dense is the best retriever for Planner RAG on MuSiQue (+8.3% EM over BM25)
- Planner RAG uses ~10 LLM calls per question but only ~2.8 retrieval calls, suggesting decomposition doesn't drive more retrieval
- Bridge questions (21.1% EM) are harder than Compositional (10.1% EM), consistent with other architectures

---

### Self-RAG - MuSiQue Full Validation Set (2,417 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **11.6%** | **23.2%** | 4,290 | $1.00 | 13.79 | 0.95 |
| Hybrid    | 10.8%       | 22.3%    | 977          | $0.95 | 13.57 | 0.95 |
| BM25      | 6.8%        | 16.3%    | 6,630        | $0.98 | 14.10 | 0.95 |

*Configuration: num_candidates=3, top_k=5, concurrency=2*

**Breakdown by Question Type (Dense, Self-RAG on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 15.8% | 28.0% |
| Compositional | ~300 | 7.1% | 17.9% |

**Key Findings:**
- Self-RAG underperforms Vanilla RAG on MuSiQue: -1.0% EM (11.6% vs 12.6%), similar to HotpotQA pattern
- Self-RAG underperforms ReAct RAG on MuSiQue: -8.1% EM (11.6% vs 19.7%)
- Self-RAG uses ~14 LLM calls per question but only ~0.95 retrieval calls, meaning self-reflection frequently skips retrieval
- Low retrieval usage severely hurts multi-hop performance on MuSiQue
- Bridge questions (15.8% EM) are harder than Compositional (7.1% EM), consistent with other architectures

---

### Recursive LM - MuSiQue Full Validation Set (2,417 questions, gpt-4o-mini)

| Retriever | Exact Match | F1 Score | Latency (ms) | Cost | Avg LLM Calls | Avg Retrieval Calls |
|-----------|-------------|----------|--------------|------|---------------|---------------------|
| **Dense** | **16.7%** | **28.9%** | 8,772 | $2.10 | 7.41 | 5.32 |
| Hybrid    | 16.5%       | 28.6%    | 13,235       | $2.14 | 7.83 | 5.59 |
| BM25      | 9.8%        | 19.8%    | 13,753       | $3.09 | 12.29 | 8.51 |

*Configuration: max_depth=3, memoization=true, top_k=5, concurrency=2*

**Breakdown by Question Type (Dense, Recursive LM on MuSiQue):**

| Type | Count | Exact Match | F1 |
|------|-------|-------------|-----|
| Bridge | ~2,100 | 21.3% | 34.9% |
| Compositional | ~300 | 11.7% | 22.4% |

**Key Findings:**
- ReAct RAG leads on MuSiQue with 19.7% EM, followed by RLM at 16.7% EM
- RLM outperforms Planner RAG (15.8%) and Self-RAG (11.6%) on MuSiQue
- RLM uses ~7.4 LLM calls with ~5.3 retrieval calls, more balanced than Self-RAG
- BM25 with RLM requires significantly more iterations (12.29 LLM calls, 8.51 retrievals) than Dense/Hybrid

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

Note: All full runs complete (Vanilla, ReAct, Self-RAG, Planner RAG, RLM, IRCoT) with gpt-4o-mini on HotpotQA and MuSiQue. REAP pending. Next comparison milestone is cross-model (OpenAI vs Anthropic).

### Priority 3: Additional Datasets

- ~~MuSiQue~~ ✅ Vanilla RAG, ReAct RAG, Planner RAG, Self-RAG, Recursive LM results available
- 2WikiMultiHopQA - All architectures pending

### Priority 4: Remaining Architectures

- ~~Planner RAG (Agentic)~~ ✅ Implemented
- ~~IRCoT (Recursive)~~ ✅ Implemented (full validation complete)
- ~~REAP (Recursive)~~ ✅ Implemented (subset smoke run complete; full validation configs added)
- ~~Recursive LM (RLM)~~ ✅ Implemented

### Priority 5: Full Validation Runs

Note: All full validation runs complete (Vanilla, ReAct, Self-RAG, Planner RAG, RLM, IRCoT, REAP) on HotpotQA and MuSiQue. 2WikiMultiHopQA pending.

---

## Completed Results Summary

### HotpotQA (7,405 questions)

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
| `09581743` | recursive_lm | hybrid | 7,405 | 46.3% | 60.2% | $3.19 |
| `a381842d` | recursive_lm | dense | 7,405 | 46.1% | 60.1% | $3.34 |
| `9b4f7587` | recursive_lm | bm25 | 7,405 | 40.1% | 52.6% | $5.01 |
| `1c4afb94` | ircot_rag | bm25 | 7,405 | 38.5% | 54.7% | $3.97 |
| `3e4b5fc8` | ircot_rag | dense | 7,405 | 42.0% | 59.2% | $3.77 |
| `4d923d09` | ircot_rag | hybrid | 7,405 | 42.9% | 59.9% | $3.81 |
| `0eb19318` | reap_rag | bm25 | 7,405 | 24.6% | 36.6% | $7.41 |
| `c4da1615` | reap_rag | dense | 7,405 | 28.1% | 41.7% | $6.49 |
| `f667025a` | reap_rag | hybrid | 7,405 | 27.3% | 41.1% | $6.54 |

**HotpotQA Total cost:** ~$94.52

### MuSiQue (2,417 questions)

| Run ID | Architecture | Retriever | Questions | EM | F1 | Cost |
|--------|-------------|-----------|-----------|-----|-----|------|
| `eecfa8d0` | vanilla_rag | bm25 | 2,417 | 6.9% | 16.3% | $0.26 |
| `e1c01e60` | vanilla_rag | dense | 2,417 | 12.6% | 24.3% | $0.27 |
| `db4e2728` | vanilla_rag | hybrid | 2,417 | 12.0% | 23.5% | $0.27 |
| `85e4a6b9` | react_rag | bm25 | 2,417 | 11.9% | 18.0% | $5.74 |
| `8d4736d4` | react_rag | dense | 2,417 | 19.7% | 27.8% | $5.24 |
| `f34150a9` | react_rag | hybrid | 2,417 | 19.3% | 28.1% | $5.15 |
| `6cc7eba1` | planner_rag | bm25 | 2,417 | 7.5% | 15.1% | $2.07 |
| `66516cec` | planner_rag | dense | 2,417 | 15.8% | 25.9% | $2.08 |
| `dc3a1e2e` | planner_rag | hybrid | 2,417 | 14.6% | 24.3% | $2.03 |
| `b666a9d4` | self_rag | bm25 | 2,417 | 6.8% | 16.3% | $0.98 |
| `1c0a526f` | self_rag | dense | 2,417 | 11.6% | 23.2% | $1.00 |
| `b6cd2de9` | self_rag | hybrid | 2,417 | 10.8% | 22.3% | $0.95 |
| `32f5bafa` | recursive_lm | bm25 | 2,417 | 9.8% | 19.8% | $3.09 |
| `f01337dd` | recursive_lm | dense | 2,417 | 16.7% | 28.9% | $2.10 |
| `99900bcd` | recursive_lm | hybrid | 2,417 | 16.5% | 28.6% | $2.14 |

**MuSiQue Total cost:** ~$34.37

---

**Total cost so far:** ~$128.89

---

## Infrastructure Improvements Made

1. **ReAct prompt rewrite** - Added few-shot examples to `prompts/react.txt` for proper tool use
2. **Retry hardening** - Extended `llm_client.py` retry to handle 403/PermissionDenied, connection errors, 500s; 5 attempts with longer backoff
3. **Progress logging** - Added real-time progress output to `evaluator.py` (count, %, rate, ETA every 50 questions)
4. **Concurrency tuning** - Reduced `max_concurrency` to 3 for ReAct configs to avoid rate limits
5. **Embedding caching** - Added disk-based embedding cache in `dense.py` for faster subsequent runs
6. **BadRequestError retry** - Added `openai.BadRequestError` to retry exceptions in both LLM client and Dense retriever

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
