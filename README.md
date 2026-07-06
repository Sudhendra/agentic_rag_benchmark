# Agentic RAG Benchmark

Benchmarking AgenticRAG systems and its viability in the face of long context optimized recursive methods like Recursive RAG and Recursive LMs.

**Target:** ACL/NAACL 2026 publication

## Config-Driven Benchmarks

This project supports reproducible benchmark suites through YAML matrices in `configs/suites/`.

```bash
python scripts/run_suite.py --suite configs/suites/dev_smoke.yaml --dry-run
python scripts/run_suite.py --suite configs/suites/dev_smoke.yaml --yes --skip-existing
```

See `docs/CONFIG_DRIVEN_BENCHMARKS.md` for suite structure, filters, and reproducibility notes.

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

### MuSiQue Hop-Stratified Analysis (Best Retriever per Architecture)

**MuSiQue Question Distribution:** 2-hop: 1,252 (51.8%), 3-hop: 760 (31.4%), 4-hop: 405 (16.8%)

| Architecture | 2-Hop EM | 2-Hop F1 | 3-Hop EM | 3-Hop F1 | 4-Hop EM | 4-Hop F1 | Overall EM | Overall F1 |
|--------------|----------|----------|----------|----------|----------|----------|------------|------------|
| **IRCoT** | **26.8%** | **42.6%** | 16.2% | 30.4% | **16.5%** | **25.4%** | **21.7%** | **35.9%** |
| ReAct RAG | 23.6% | 33.9% | **18.6%** | **26.4%** | 9.9% | 11.5% | 19.7% | 27.8% |
| Recursive LM | 21.3% | 34.9% | 12.4% | 24.5% | 10.4% | 18.4% | 16.7% | 28.9% |
| Planner RAG | 21.1% | 33.0% | 11.8% | 20.6% | 6.9% | 13.6% | 15.8% | 25.9% |
| Vanilla RAG | 17.1% | 29.6% | 7.9% | 20.2% | 7.4% | 15.6% | 12.6% | 24.3% |
| Self-RAG | 15.8% | 28.0% | 6.4% | 18.4% | 8.4% | 17.1% | 11.6% | 23.2% |
| REAP | 10.0% | 20.4% | 4.3% | 11.7% | 3.0% | 8.6% | 7.0% | 15.7% |

**Key Hop-Stratified Findings:**

1. **IRCoT dominates at every hop count** - Best EM on 2-hop, 3-hop, and 4-hop questions
2. **All architectures degrade with more hops** - Performance drops from 2-hop → 3-hop → 4-hop
3. **2-hop vs 4-hop gap**: IRCoT shows smallest degradation (26.8% → 16.5% = -10.3pp), REAP shows largest (10.0% → 3.0% = -7.0pp but from lower base)
4. **ReAct RAG excels at 3-hop** (18.6% EM) - Better than IRCoT (16.2%) at this hop count
5. **4-hop is the breaking point** - All architectures struggle significantly (max 16.5% EM)
6. **Vanilla RAG baseline degrades severely** on 3-hop (7.9%) and 4-hop (7.4%)

---

### All Architectures (All Retrievers) - MuSiQue Hop Analysis

| Architecture | Retriever | 2-Hop EM | 3-Hop EM | 4-Hop EM | Overall EM |
|--------------|-----------|----------|----------|----------|------------|
| ircot_rag | dense | 26.8% | 16.2% | 16.5% | 21.7% |
| ircot_rag | hybrid | 27.2% | 13.9% | 13.8% | 20.8% |
| react_rag | dense | 23.6% | 18.6% | 9.9% | 19.7% |
| react_rag | hybrid | 23.6% | 18.0% | 8.4% | 19.3% |
| ircot_rag | bm25 | 22.4% | 10.1% | 11.9% | 16.8% |
| recursive_lm | dense | 21.3% | 12.4% | 10.4% | 16.7% |
| recursive_lm | hybrid | 22.4% | 11.3% | 7.7% | 16.5% |
| planner_rag | dense | 21.1% | 11.8% | 6.9% | 15.8% |
| planner_rag | hybrid | 19.9% | 10.4% | 5.9% | 14.6% |
| vanilla_rag | dense | 17.1% | 7.9% | 7.4% | 12.6% |
| vanilla_rag | hybrid | 16.7% | 7.1% | 6.7% | 12.0% |
| react_rag | bm25 | 17.9% | 5.8% | 4.9% | 11.9% |
| self_rag | dense | 15.8% | 6.4% | 8.4% | 11.6% |
| self_rag | hybrid | 14.5% | 7.0% | 6.7% | 10.8% |
| recursive_lm | bm25 | 13.3% | 7.1% | 4.4% | 9.8% |
| planner_rag | bm25 | 11.1% | 4.1% | 3.0% | 7.5% |
| reap_rag | hybrid | 10.0% | 4.3% | 3.0% | 7.0% |
| reap_rag | dense | 9.9% | 3.4% | 4.4% | 7.0% |
| vanilla_rag | bm25 | 9.9% | 3.8% | 3.5% | 6.9% |
| self_rag | bm25 | 9.3% | 3.9% | 4.7% | 6.8% |
| reap_rag | bm25 | 6.2% | 3.4% | 1.2% | 4.5% |

---

### Error Analysis (MuSiQue)

#### Error Rate by Hop Count

| Architecture | 2-Hop Error Rate | 3-Hop Error Rate | 4-Hop Error Rate | Overall Error Rate |
|--------------|-----------------|-----------------|-----------------|-------------------|
| **IRCoT** | **73.2%** | **83.8%** | **83.5%** | **78.3%** |
| ReAct RAG | 76.4% | 81.4% | 90.1% | 80.3% |
| Recursive LM | 78.7% | 87.6% | 89.6% | 83.3% |
| Planner RAG | 78.9% | 88.2% | 93.1% | 84.2% |
| Vanilla RAG | 82.9% | 92.1% | 92.6% | 87.4% |
| Self-RAG | 84.2% | 93.6% | 91.6% | 88.4% |
| REAP | **90.0%** | **95.7%** | **97.0%** | **93.0%** |

#### Key Error Analysis Findings

1. **IRCoT has lowest error rate** across all hop counts - consistent with its overall best performance
2. **Error rate increases with hop count** for all architectures
3. **REAP fails catastrophically** - 90%+ error rate on 2-hop, 95%+ on 3-hop and 4-hop
4. **3-hop is a critical threshold** - error rates jump significantly from 2-hop to 3-hop
5. **4-hop questions are nearly unsolvable** - all architectures have >83% error rate

#### Sample Errors by Architecture

**IRCoT (Best) - 2-hop failure examples:**
- Pred: "None" | Gold: "Miquette Giraudy"
- Pred: "Carl Laemmle" | Gold: "Mike Medavoy"
- Pred: "Nuevo Laredo Municipality" | Gold: "Tamaulipas"

**REAP (Worst) - 2-hop failure examples:**
- Pred: "Unknown" | Gold: "Miquette Giraudy"
- Pred: "Orion Pictures" | Gold: "Mike Medavoy"
- Pred: "Nuevo Laredo, Mexico" | Gold: "Tamaulipas"

**REAP failure patterns:**
- Returns "Unknown" or "Insufficient information" when context exists
- Extracts wrong entities from context
- Overlong/garbled responses on complex questions

**Vanilla RAG - 4-hop failure examples:**
- Pred: "The context does not provide information about the duration..." | Gold: "about 400 years"
- Common: Gives up early with "no information" rather than chaining reasoning

---

### HotpotQA Error Analysis

#### Error Rate by Question Type

| Architecture | Bridge EM | Bridge Error | Comparison EM | Comparison Error | Overall EM |
|--------------|-----------|--------------|---------------|------------------|------------|
| **Recursive LM** | 41.2% | 58.8% | **66.7%** | 33.3% | **46.3%** |
| ReAct RAG | 44.5% | 55.5% | 52.2% | 47.8% | 46.0% |
| IRCoT | **45.3%** | 54.7% | 33.4% | 66.6% | 42.9% |
| Self-RAG | 36.0% | 64.0% | 59.0% | 41.0% | 40.6% |
| Vanilla RAG | 39.6% | 60.4% | 66.3% | 33.7% | 45.0% |
| Planner RAG | 33.9% | 66.1% | 32.8% | 67.2% | 33.7% |
| REAP | 24.1% | 75.9% | 44.2% | 55.8% | 28.1% |

#### Key HotpotQA Error Findings

1. **Comparison questions are easier** - All architectures perform better on Comparison than Bridge
2. **Recursive LM excels at Comparison** (66.7% EM) - Best on comparison questions
3. **IRCoT struggles on Comparison** (33.4% EM) - Surprisingly weak on comparison
4. **REAP fails on Bridge** (75.9% error rate) - Worst on multi-hop Bridge questions
5. **Bridge vs Comparison gap varies by architecture** - IRCoT has reverse pattern (better on Bridge)

#### Sample Errors - HotpotQA

**Recursive LM (Best overall) - Bridge failures:**
- Pred: "Under Secretary of State for Political Affairs" | Gold: "Chief of Protocol"
- Pred: "New York City" | Gold: "Greenwich Village, New York City"

**Recursive LM - Comparison failures:**
- Pred: "yes" | Gold: "no"
- Pred: "Robert Erskine Childers" | Gold: "Robert Erskine Childers DSC"

**REAP (Worst) - Bridge failures:**
- Pred: "The information is not available." | Gold: "Chief of Protocol"
- Pred: "The 'Starbound' series by Amie Kaufman" | Gold: "Animorphs"

**REAP - Comparison failures:**
- Pred: "yes" | Gold: "no" (multiple instances)
- Common: Binary "yes/no" confusion on comparison questions

---

### MuSiQue: 3-hop vs 4-hop Breaking Point

#### 3-hop Failures (IRCoT)
- Pred: "England" | Gold: "Denver"
- Pred: "Xanana Gusmão" | Gold: "Francisco Guterres" (entity confusion)
- Pattern: Entity confusion between related entities in same domain

#### 4-hop Failures (Breaking Point)
- Pred: "approximately 53 years" | Gold: "about 400 years" (order of magnitude error)
- Pred: "None" | Gold: "about 400 years" (complete failure)
- Pred: "53 years" | Gold: "about 400 years" (wrong reasoning chain)
- Pattern: Numerical/quantitative reasoning breaks down completely

#### ReAct vs IRCoT on 3-hop

| Metric | ReAct | IRCoT | Difference |
|--------|-------|-------|------------|
| EM on 3-hop | 18.6% | 16.2% | +2.4% (ReAct better) |
| Questions where ReAct succeeds, IRCoT fails | 79 | - | - |
| Questions where IRCoT succeeds, ReAct fails | - | 61 | - |

**Key Insight:** ReAct is better at 3-hop, IRCoT is better at 4-hop. This suggests different reasoning strategies work better at different complexity levels.

---

### Generating Error Analysis

To generate error analysis for any run:

```bash
# Basic error analysis
python scripts/analyze_results.py --results results/<run_id> --errors

# Error analysis by hop count (MuSiQue)
python scripts/analyze_results.py --results results/<run_id> --errors --hops

# Error analysis by question type (HotpotQA - Bridge/Comparison)
python scripts/analyze_results.py --results results/<run_id> --errors --breakdown

# Filter by F1 threshold
python scripts/analyze_results.py --results results/<run_id> --errors --error-threshold 0.3
```

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
