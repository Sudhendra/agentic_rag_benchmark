# Finding 2: Vanilla RAG Dominates the Cost-Performance Pareto Frontier

**Filed:** June 17, 2026  
**Severity:** 🔴 Critical (core paper narrative)  
**Tags:** `cost-efficiency` `pareto` `baseline`

---

## The Finding

Vanilla RAG (single retrieve-then-read) achieves **45.0% EM / 59.5% F1** at **$0.79 total cost** on HotpotQA. No architecture with higher cost delivers statistically better F1.

This establishes Vanilla as the **Pareto-optimal point**: no other architecture improves F1 without massively increasing cost.

## The Data

| Architecture | EM | F1 | F1 95% CI | Cost | Cost Ratio |
|-------------|-----|-----|-----------|------|------------|
| **Vanilla RAG** | **45.0%** | **59.5%** | [0.584, 0.605] | **$0.79** | **1.0x** |
| RLM | 46.3% | 60.2% | [0.592, 0.612] | $3.19 | 4.0x |
| ReAct | 46.0% | 59.9% | [0.589, 0.609] | $9.18 | 11.6x |
| IRCoT | 42.9% | 59.9% | [0.588, 0.610] | $3.81 | 4.8x |
| Self-RAG | 40.6% | 55.0% | — | $2.08 | 2.6x |
| Planner | 33.7% | 44.9% | — | $4.03 | 5.1x |
| REAP | 28.1% | 41.7% | — | $6.49 | 8.2x |

**All runs:** gpt-4o-mini, HotpotQA distractor, 7,405 questions.

## Analysis

### The Confidence Interval Problem
- Vanilla F1 CI: [0.584, 0.605]
- ReAct F1 CI: [0.589, 0.609]
- RLM F1 CI: [0.592, 0.612]

All three intervals **overlap substantially**. There is no statistically significant accuracy difference between Vanilla, ReAct, and RLM at this model scale. The only real difference is cost.

### The Cost Multiplier
- Vanilla: $0.79
- RLM: $3.19 (4x cost, +1.3% EM — not significant)
- ReAct: $9.18 (12x cost, +1.0% EM — not significant)
- REAP: $6.49 (8x cost, -16.9% EM — significantly worse)

### The Efficiency Metric
Define `efficiency = F1 / cost`. Vanilla scores 75.3 F1/$, beating the next best (Self-RAG at 26.4 F1/$) by **2.9x**.

## Why This Matters

This is the paper's central finding: **For small/cheap LLMs (gpt-4o-mini class), complex agentic loops waste tokens without improving accuracy.** The retrieval-and-read paradigm is Pareto-optimal.

The practical implication: practitioners should invest in **better retrieval** (dense, hybrid) rather than agentic loops when using affordable models.

## Caveats
- This is on HotpotQA (distractor setting). Results may differ on full-wiki or MuSiQue.
- Results may differ on stronger models (gpt-4o, Claude 3.5 Sonnet).
- RLM with the optimized v4 prompt (Finding 1) may shift the frontier.
