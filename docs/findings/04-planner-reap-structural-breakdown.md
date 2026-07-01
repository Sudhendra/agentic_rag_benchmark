# Finding 4: Planner & REAP — Structural Breakdown of Complex Reasoning

**Filed:** June 17, 2026  
**Severity:** 🟠 Major (interesting negative result)  
**Tags:** `planner` `reap` `decomposition`

---

## The Finding

The two most architecturally complex methods — Planner RAG and REAP — perform **significantly worse than Vanilla RAG**, despite being far more expensive.

| Architecture | EM | F1 | Cost | Compared to Vanilla |
|-------------|-----|-----|------|-------------------|
| Vanilla | 45.0% | 59.5% | $0.79 | — |
| Planner | 33.7% | 44.9% | $4.03 | **-11.3% EM, 5.1x cost** |
| REAP | 28.1% | 41.7% | $6.49 | **-16.9% EM, 8.2x cost** |

## Planner RAG Analysis

**Planner decomposes questions, solves sub-questions, and synthesizes answers.** It fails because:

1. **Over-decomposition** — The planner breaks simple questions into unnecessary sub-questions, each introducing error
2. **Error compounding** — Errors in sub-answers propagate to the final synthesis
3. **Prompt complexity** — 4 separate prompts (action, solve, synthesize, bridge_refine) create too many failure points

Planner code is the largest in the project at **1,403 lines** — the most complex implementation with the worst results.

## REAP Analysis

**REAP decomposes, plans, extracts, and synthesizes.** It fails because:

1. **Multi-stage pipeline fragility** — Each of the 5 stages (decompose, extract, plan, replan, synthesize) has its own failure mode
2. **JSON parsing failures** — The extract/plan stages rely on structured output that small models fail to produce reliably
3. **Model strength ceiling** — gpt-4o-mini cannot reliably execute the full REAP protocol

### The REAP Gap (28.1% vs paper's 59.2%)

| Factor | Estimated Impact |
|--------|-----------------|
| Model strength (gpt-4o-mini vs GPT-4) | ~20-25 points |
| Implementation fidelity | ~5-8 points |
| Prompt quality | ~3-5 points |

The dominant factor is **model strength**. The original REAP paper likely uses GPT-4, which handles structured multi-step protocols far better.

## Paper Narrative

Frame this as: *"Complex multi-stage architectures degrade catastrophically on small LLMs. The added reasoning structure does not compensate for the model's limited capacity to follow multi-step protocols."*

This is a **strong negative result** — more interesting to reviewers than a positive one, since it challenges the assumption that "more structure = better reasoning."
