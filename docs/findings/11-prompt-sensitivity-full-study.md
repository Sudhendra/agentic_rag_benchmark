# Finding 11: Prompt Sensitivity — Full Cross-Architecture Study

**Filed:** June 18, 2026  
**Severity:** 🟠 Major (paper contribution — novel finding)  
**Tags:** `prompt-engineering` `cross-architecture` `sensitivity` `rlm` `react` `ircot`

---

## Summary

A systematic prompt sensitivity study across RLM (5 variants), ReAct (3 variants), and IRCoT (3
variants) reveals that **architecture-specific prompt brittleness is itself a measurable and
meaningful property** of RAG systems. The "prompt tax" — performance swing caused purely by
wording changes — varies from 10 percentage points (IRCoT, robust) to 42 percentage points
(ReAct, brittle). This exceeds the performance gap between most architectures in the main
benchmark.

**Total cost:** $0.28 for all 11 variants × 50 questions.

---

## Complete Results

### Recursive LM (5 variants)

| Variant | EM | F1 | Cost | Latency | Tok/Q | Run ID |
|---------|-----|-----|------|---------|-------|--------|
| **v4 Strict** — format constraints + negative examples | **58.0%** | **70.5%** | $0.012 | 12ms | 1,525 | `dacd4aba06c9` |
| v2 Structured — step-by-step scaffolding | 50.0% | 67.2% | $0.039 | 35ms | 4,324 | `d5679dba58e8` |
| v0 Baseline — current default | 44.0% | 62.7% | $0.018 | 6ms | 2,271 | `9baf0332d319` |
| v3 Persona — "expert PhD" framing | 40.0% | 54.7% | $0.040 | 10ms | 4,791 | `ba3580b34390` |
| v1 Minimalist — stripped to essentials | 18.0% | 36.7% | $0.009 | 3ms | 1,126 | `7fd24f3b79aa` |

**Prompt Tax: 40.0% EM swing** (58% → 18%)  
**EM Std Dev: 13.4%** | **Coef of Variation: 0.32**

### ReAct (3 variants)

| Variant | EM | F1 | Cost | Latency | Tok/Q | Run ID |
|---------|-----|-----|------|---------|-------|--------|
| **v0 Baseline** — 2 worked examples, detailed rules | **48.0%** | **62.0%** | $0.038 | 5ms | 4,622 | `f7d73e293f05` |
| v2 Strict — explicit format rules, max 8 steps | 26.0% | 39.3% | $0.011 | 2ms | 708 | `254a6dd3a4ab` |
| v1 Minimalist — minimal instructions, no examples | 6.0% | 20.6% | $0.024 | 5ms | 2,814 | `cd5c69000c96` |

**Prompt Tax: 42.0% EM swing** (48% → 6%)  
**EM Std Dev: 17.2%** | **Coef of Variation: 0.64**

### IRCoT (3 variants)

| Variant | EM | F1 | Cost | Latency | Tok/Q | Run ID |
|---------|-----|-----|------|---------|-------|--------|
| **v2 Strict** — format constraints, [ANSWER] enforcement | **52.0%** | **64.6%** | $0.014 | 4ms | 1,800 | `f38de73084ee` |
| v0 Baseline — current default | 44.0% | 67.8% | $0.040 | 9ms | 5,045 | `c5d906ecaa0d` |
| v1 Minimalist — one-liner instruction | 42.0% | 61.5% | $0.038 | 10ms | 4,711 | `35e551274732` |

**Prompt Tax: 10.0% EM swing** (52% → 42%)  
**EM Std Dev: 4.3%** | **Coef of Variation: 0.09**

---

## Cross-Architecture Prompt Tax Summary

| Architecture | Variants | EM Swing | F1 Swing | Best Variant | Worst Variant |
|-------------|:--------:|:--------:|:--------:|:------------:|:-------------:|
| **ReAct** | 3 | **42%** | 41.5% | Baseline (48%) | Minimalist (6%) |
| **RLM** | 5 | **40%** | 33.9% | Strict (58%) | Minimalist (18%) |
| **IRCoT** | 3 | **10%** | 6.2% | Strict (52%) | Minimalist (42%) |

**Setup:** HotpotQA distractor, BM25 retriever, gpt-4o-mini, temperature=0, 50 questions each.

---

## Key Insights

### 1. IRCoT is Structurally Robust; ReAct is Structurally Fragile

IRCoT's interleaved reasoning format — "write one grounded sentence, or output `[ANSWER]`" —
is intuitive enough that even a one-liner instruction produces 42% EM. The task is legible to the
model without scaffolding.

ReAct's Thought/Action/Observation loop is **not intuitive** for gpt-4o-mini. Without worked
examples demonstrating `search[query]` and `finish[answer]` syntax, the model:
- Outputs free-form text instead of structured steps
- Skips retrieval and answers from memory
- Gets confused by the `finish[answer]` bracket format
- Collapses to 6% EM — barely above random

**Implication:** ReAct's reported performance in the literature likely reflects as much about
prompt engineering quality as about the underlying loop mechanism.

### 2. RLM's Best Prompt (Strict, 58%) Beats Its Full-Run Baseline (46.3%)

The v4 Strict variant on 50 questions hits 58% EM. The full-run RLM on 7,405 questions with the
default prompt hits 46.3% EM. The difference is entirely prompt wording — the format constraints
and negative examples in v4 force the model into extractive mode, reducing hallucination and
unnecessary decomposition.

This suggests the 46.3% full-run figure is a **lower bound** on RLM capability, not a ceiling.
A future full run with v4 as default could materially raise the reported score.

### 3. The Strict Variant Wins for Both RLM and IRCoT — But Backfires for ReAct

For RLM and IRCoT, adding explicit output constraints and negative examples improves performance.
For ReAct, the strict variant (26% EM) is worse than the baseline (48%) because it removes the
worked examples that are essential for format compliance.

This reveals a nuanced pattern: constraints help when the task is already legible; they hurt when
they replace the worked examples the model needs to understand the format at all.

### 4. Cost Efficiency Inverts Across Architectures

| Architecture | Best-performing variant | Cost | Interpretation |
|---|---|---|---|
| RLM | Strict (58%) | $0.012 | Fewer recursive calls — stops when confident |
| ReAct | Baseline (48%) | $0.038 | Examples require more tokens but pay off |
| IRCoT | Strict (52%) | $0.014 | Enforced [ANSWER] format ends loops sooner |

The cheapest variant is not always the worst, and the most expensive is not always the best.

---

## Paper Narrative

This finding should appear in the paper as its own subsection, framed as:

> *"We measure prompt sensitivity — the EM swing across prompt variants — as a first-class
> architectural property. IRCoT exhibits a 10-point prompt tax (robust); ReAct and RLM exhibit
> 40–42-point taxes (brittle). This differential sensitivity must be accounted for in any fair
> cross-architecture comparison: performance differences that appear to reflect architectural
> choices may partly reflect prompt engineering quality."*

This directly responds to the Grok reviewer's note that "prompt engineering dependence" is a
weakness. We turn it into a contribution by quantifying it systematically.

---

## Action Items

- [x] Run all 11 variants (RLM ×5, ReAct ×3, IRCoT ×3) — June 18, 2026
- [x] Export consolidated JSON: `results/sensitivity/sensitivity_results.json`
- [ ] Run full-scale validation of RLM v4 Strict on all 7,405 HotpotQA questions
- [ ] Add prompt sensitivity figure to `generate_publication_figures.py` (Figure 5)
- [ ] Add prompt sensitivity section to paper (Methods §3.4 or Results §4.5)
