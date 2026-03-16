# Next Steps: Agentic RAG Benchmark

**Date:** March 16, 2026  
**Status:** Phase 3 Complete - All Architectures (Vanilla, ReAct, Planner, Self-RAG, Recursive LM, IRCoT, REAP) Full HotpotQA and MuSiQue Results Available; 2WikiMultiHopQA Baseline Complete, Architectures Pending  
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
| MuSiQue Loader | `src/data/musique.py` | ✅ Complete |
| 2WikiMultiHopQA Loader | `src/data/wiki2hop.py` | ✅ Complete |
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
| Planner RAG | `src/architectures/agentic/planner_rag.py` | ✅ Complete |
| IRCoT | `src/architectures/recursive/ircot.py` | ✅ Complete |
| REAP | `src/architectures/recursive/reap.py` | ✅ Complete |
| RLM Prompts | `prompts/rlm.txt`, `prompts/rlm_combine.txt` | ✅ Complete |
| RLM Tests | `tests/test_recursive_lm.py` (16 tests) | ✅ Complete |

---

## Benchmark Results

### HotpotQA Cross-Architecture Summary (7,405 questions, gpt-4o-mini)

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost | Run ID |
|--------------|------|----------------|-------------|----------|---------------|------|--------|
| Vanilla RAG  | Baseline | Dense    | 45.0%       | 59.5%    | 1.0           | $0.79 | bfc8f293 |
| **Recursive LM** | **RLM** | **Hybrid** | **46.3%** | **60.2%** | **3.60** | **$3.19** | **09581743** |
| ReAct RAG | Agentic | Hybrid | 46.0% | 59.9% | 4.05 | $9.18 | 25cc3f6b |
| IRCoT | Recursive | Hybrid | 42.9% | 59.9% | 4.16 | $3.81 | 4d923d09 |
| Self-RAG     | Agentic  | Hybrid   | 40.6%       | 55.0%    | 10.75         | $2.08 | 7272b4eb |
| Planner RAG  | Agentic  | Dense    | 33.7%       | 44.9%    | 8.13          | $4.03 | b4284f7f |
| REAP | Recursive | Dense | 28.1% | 41.7% | 6.24 | $6.49 | c4da1615 |

### MuSiQue Cross-Architecture Summary (2,417 questions, gpt-4o-mini)

| Architecture | Type | Best Retriever | Exact Match | F1 Score | Avg LLM Calls | Cost | Run ID |
|--------------|------|----------------|-------------|----------|---------------|------|--------|
| Vanilla RAG  | Baseline | Dense    | 12.6%       | 24.3%    | 1.0           | $0.27 | e1c01e60 |
| **IRCoT** | **Recursive** | **Dense** | **21.7%** | **35.9%** | **4.64** | **$1.71** | **51077083** |
| ReAct RAG | Agentic | Dense | 19.7% | 27.8% | 5.57 | $5.24 | 8d4736d4 |
| Recursive LM | RLM | Dense | 16.7% | 28.9% | 7.41 | $2.10 | f01337dd |
| Planner RAG  | Agentic  | Dense    | 15.8%       | 25.9%    | 10.15         | $2.08 | 66516cec |
| Self-RAG     | Agentic  | Dense   | 11.6%       | 23.2%    | 13.79         | $1.00 | 1c0a526f |
| REAP | Recursive | Hybrid | 7.0% | 15.7% | 6.93 | $2.14 | 842f7d8b |

### 2WikiMultiHopQA (12,576 questions)

| Architecture | Retriever | Exact Match | F1 Score | Cost | Status |
|--------------|-----------|-------------|----------|------|--------|
| Vanilla RAG | BM25 | 26.0% | 30.9% | $1.16 | ✅ Complete (fb0e11a5) |
| ReAct RAG | - | - | - | - | ❌ Pending |
| Self-RAG | - | - | - | - | ❌ Pending |
| Planner RAG | - | - | - | - | ❌ Pending |
| Recursive LM | - | - | - | - | ❌ Pending |
| IRCoT | - | - | - | - | ❌ Pending |
| REAP | - | - | - | - | ❌ Pending |

---

## Key Findings

### HotpotQA Findings

1. **Recursive LM leads on HotpotQA** with 46.3% EM and 60.2% F1
2. **ReAct RAG is a close second** at 46.0% EM but costs 3x more ($9.18 vs $3.19)
3. **IRCoT is a strong middle ground**: 42.9% EM at $3.81 (similar cost to RLM)
4. **Self-RAG underperforms Vanilla RAG** (-4.4% EM) despite ~11 LLM calls
5. **Planner RAG severely underperforms** all architectures (33.7% EM)
6. **REAP is the worst architecture** on HotpotQA (28.1% EM)

### MuSiQue Findings

1. **IRCoT leads on MuSiQue** with 21.7% EM, outperforming all other architectures
2. **ReAct RAG is second best** at 19.7% EM
3. **Recursive LM achieves 16.7% EM** with good cost-efficiency ($2.10)
4. **REAP significantly underperforms** on MuSiQue (7.0% EM) - worst by far
5. **All architectures except REAP outperform Vanilla RAG baseline** on MuSiQue

### Cross-Dataset Observations

- **HotpotQA vs MuSiQue difficulty**: All architectures perform significantly worse on MuSiQue (max 46% vs 22% EM)
- **IRCoT generalizes well**: Best on MuSiQue, competitive on HotpotQA
- **REAP struggles**: Consistently worst or near-worst on both datasets

---

## Remaining Tasks

### Priority 1: 2WikiMultiHopQA Full Experiments

Run all architectures on 2WikiMultiHopQA to complete the benchmark:
- Vanilla RAG (BM25, Dense, Hybrid) ✅ Complete
- ReAct RAG (BM25, Dense, Hybrid) ❌ Pending
- Self-RAG (BM25, Dense, Hybrid) ❌ Pending
- Planner RAG (BM25, Dense, Hybrid) ❌ Pending
- Recursive LM (BM25, Dense, Hybrid) ❌ Pending
- IRCoT (BM25, Dense, Hybrid) ❌ Pending
- REAP (BM25, Dense, Hybrid) ❌ Pending

**Estimated cost:** ~$150-200 for full 2WikiMultiHopQA (12,576 questions × 7 architectures × 3 retrievers)

### Priority 2: Anthropic Client Implementation

**File:** `src/core/llm_client.py`

Implement `AnthropicClient` to enable running all architectures with Claude models for cross-model comparison.

### Priority 3: Cross-Model Comparison

After adding Anthropic model runs:
```bash
python scripts/analyze_results.py --results results --compare
```

---

## Completed Results Summary

### HotpotQA (7,405 questions)

| Run ID | Architecture | Retriever | Questions | EM | F1 | Cost |
|--------|-------------|-----------|-----------|-----|-----|------|
| bfc8f293 | vanilla_rag | dense | 7,405 | 45.0% | 59.5% | $0.79 |
| b054e24b | vanilla_rag | hybrid | 7,405 | 44.1% | 58.6% | $0.76 |
| 74d7b162 | vanilla_rag | bm25 | 7,405 | 38.2% | 51.5% | $0.77 |
| 25cc3f6b | react_rag | hybrid | 7,405 | 46.0% | 59.9% | $9.18 |
| 47103104 | react_rag | dense | 7,405 | 45.7% | 59.3% | $9.66 |
| 0e7932b0 | react_rag | bm25 | 7,405 | 38.8% | 50.8% | $11.16 |
| 7272b4eb | self_rag | hybrid | 7,405 | 40.6% | 55.0% | $2.08 |
| 72dc70f2 | self_rag | dense | 7,405 | 40.6% | 54.9% | $2.13 |
| e8d57330 | self_rag | bm25 | 7,405 | 37.0% | 50.4% | $2.15 |
| b4284f7f | planner_rag | dense | 7,405 | 33.7% | 44.9% | $4.03 |
| dedaa9b2 | planner_rag | hybrid | 7,405 | 33.5% | 44.5% | $3.98 |
| 19114c8b | planner_rag | bm25 | 7,405 | 27.7% | 37.7% | $4.09 |
| 09581743 | recursive_lm | hybrid | 7,405 | 46.3% | 60.2% | $3.19 |
| a381842d | recursive_lm | dense | 7,405 | 46.1% | 60.1% | $3.34 |
| 9b4f7587 | recursive_lm | bm25 | 7,405 | 40.1% | 52.6% | $5.01 |
| 4d923d09 | ircot_rag | hybrid | 7,405 | 42.9% | 59.9% | $3.81 |
| 3e4b5fc8 | ircot_rag | dense | 7,405 | 42.0% | 59.2% | $3.77 |
| 1c4afb94 | ircot_rag | bm25 | 7,405 | 38.5% | 54.7% | $3.97 |
| c4da1615 | reap_rag | dense | 7,405 | 28.1% | 41.7% | $6.49 |
| f667025a | reap_rag | hybrid | 7,405 | 27.3% | 41.1% | $6.54 |
| 0eb19318 | reap_rag | bm25 | 7,405 | 24.6% | 36.6% | $7.41 |

**HotpotQA Total cost:** $97.27

### MuSiQue (2,417 questions)

| Run ID | Architecture | Retriever | Questions | EM | F1 | Cost |
|--------|-------------|-----------|-----------|-----|-----|------|
| e1c01e60 | vanilla_rag | dense | 2,417 | 12.6% | 24.3% | $0.27 |
| db4e2728 | vanilla_rag | hybrid | 2,417 | 12.0% | 23.5% | $0.27 |
| eecfa8d0 | vanilla_rag | bm25 | 2,417 | 6.9% | 16.3% | $0.26 |
| 8d4736d4 | react_rag | dense | 2,417 | 19.7% | 27.8% | $5.24 |
| f34150a9 | react_rag | hybrid | 2,417 | 19.3% | 28.1% | $5.15 |
| 85e4a6b9 | react_rag | bm25 | 2,417 | 11.9% | 18.0% | $5.74 |
| 66516cec | planner_rag | dense | 2,417 | 15.8% | 25.9% | $2.08 |
| dc3a1e2e | planner_rag | hybrid | 2,417 | 14.6% | 24.3% | $2.03 |
| 6cc7eba1 | planner_rag | bm25 | 2,417 | 7.5% | 15.1% | $2.07 |
| 1c0a526f | self_rag | dense | 2,417 | 11.6% | 23.2% | $1.00 |
| b6cd2de9 | self_rag | hybrid | 2,417 | 10.8% | 22.3% | $0.95 |
| b666a9d4 | self_rag | bm25 | 2,417 | 6.8% | 16.3% | $0.98 |
| f01337dd | recursive_lm | dense | 2,417 | 16.7% | 28.9% | $2.10 |
| 99900bcd | recursive_lm | hybrid | 2,417 | 16.5% | 28.6% | $2.14 |
| 32f5bafa | recursive_lm | bm25 | 2,417 | 9.8% | 19.8% | $3.09 |
| 51077083 | ircot_rag | dense | 2,417 | 21.7% | 35.9% | $1.71 |
| 202cc61a | ircot_rag | hybrid | 2,417 | 20.8% | 34.6% | $1.73 |
| d9578a21 | ircot_rag | bm25 | 2,417 | 16.8% | 28.1% | $1.68 |
| 842f7d8b | reap_rag | hybrid | 2,417 | 7.0% | 15.7% | $2.14 |
| 37b58fa1 | reap_rag | dense | 2,417 | 7.0% | 15.6% | $2.17 |
| 26863c88 | reap_rag | bm25 | 2,417 | 4.5% | 11.6% | $2.23 |

**MuSiQue Total cost:** $45.02

### 2WikiMultiHopQA (12,576 questions)

| Run ID | Architecture | Retriever | Questions | EM | F1 | Cost |
|--------|-------------|-----------|-----------|-----|-----|------|
| fb0e11a5 | vanilla_rag | bm25 | 12,576 | 26.0% | 30.9% | $1.16 |

**2WikiMultiHopQA Total cost:** $1.16

---

## Total Experiment Cost

| Dataset | Questions | Runs | Cost |
|---------|-----------|------|------|
| HotpotQA | 7,405 | 21 | $97.27 |
| MuSiQue | 2,417 | 21 | $45.02 |
| 2WikiMultiHopQA | 12,576 | 1 | $1.16 |
| **Total** | **22,398** | **43** | **$143.46** |

---

## Infrastructure Improvements Made

1. **ReAct prompt rewrite** - Added few-shot examples to `prompts/react.txt` for proper tool use
2. **Retry hardening** - Extended `llm_client.py` retry to handle 403/PermissionDenied, connection errors, 500s; 5 attempts with longer backoff
3. **Progress logging** - Added real-time progress output to `evaluator.py` (count, %, rate, ETA every 50 questions)
4. **Concurrency tuning** - Reduced `max_concurrency` to 3 for ReAct configs to avoid rate limits
5. **Embedding caching** - Added disk-based embedding cache in `dense.py` for faster subsequent runs
6. **BadRequestError retry** - Added `openai.BadRequestError` to retry exceptions in both LLM client and Dense retriever
7. **2Wiki dataset fix** - Changed from deprecated `xanhho/2WikiMultihopQA` to `framolfese/2WikiMultihopQA`

---

## Questions for Team Discussion

1. Is the marginal ReAct improvement (+1% EM over RLM) worth the 3x cost increase? How to frame this in the paper?
2. Should we add retrieval quality metrics (recall@k, precision@k) as an additional analysis?
3. Do we need fullwiki setting for HotpotQA, or is distractor sufficient for the paper?
4. Should we add a cost budget limit to the experiment runner?
5. Self-RAG underperforms Vanilla RAG despite ~11 LLM calls per question. Is the low retrieval rate (0.84 calls) the primary cause? Should we experiment with forcing retrieval?
6. How should we frame the Self-RAG results in the paper -- as evidence that self-reflection without sufficient retrieval is counterproductive for multi-hop QA?
7. RLM prompt sensitivity was extreme (9% -> 52% EM from wording changes alone). Should we include a prompt sensitivity analysis section in the paper?
8. Planner RAG severely underperforms all architectures (33.7% EM vs 45% Vanilla, 46% ReAct). Is this an implementation issue, or does tree-based planning fundamentally not work for HotpotQA-style multi-hop questions?
9. REAP consistently performs worst across both datasets. Should we investigate implementation issues or deprioritize for the paper?
10. Should we run 2WikiMultiHopQA experiments for all architectures, or focus on paper writing first?

(End of file - total 328 lines)
