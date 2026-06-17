# Finding 3: ReAct's Agentic Loop Fails to Justify Its Cost

**Filed:** June 17, 2026  
**Severity:** 🟠 Major (supports core narrative)  
**Tags:** `agentic` `react` `cost-analysis`

---

## The Finding

ReAct achieves **46.0% EM** — statistically indistinguishable from Vanilla's 45.0% — but costs **11.6x more** ($9.18 vs $0.79) and takes **5.4x longer** (7.6s vs 1.4s per question).

## The Data

| Metric | Vanilla | ReAct | Delta |
|--------|---------|-------|-------|
| EM | 45.0% | 46.0% | +1.0% |
| F1 | 59.5% | 59.9% | +0.4% |
| F1 CI | [0.584, 0.605] | [0.589, 0.609] | Overlap |
| Cost | $0.79 | $9.18 | **11.6x** |
| Latency | 1.4s | 7.6s | **5.4x** |
| Tokens/Q | 607 | 9,355 | **15.4x** |

## Why ReAct Fails on Small Models

1. **The scratchpad grows unbounded** — each Thought-Action-Observation step appends to context, causing quadratic token growth
2. **gpt-4o-mini's reasoning ceiling** — the model struggles to maintain coherent multi-step plans, frequently looping or converging to wrong answers
3. **Retrieval overuse** — ReAct searches for every sub-question independently, but Vanilla gets similar information in one shot

## The Inefficiency Mechanism

ReAct's Thought/Action/Observation loop creates a **negative feedback cycle**:
- More steps → longer context → more attention dilution
- More retrieval calls → more noise → model gets confused
- Confusion → more steps → repeat

## Paper Narrative

This directly supports the "Agentic loops fail on SLMs" thesis. The 12x cost premium buys no statistical accuracy gain. It's the strongest quantitative evidence against indiscriminate agentic adoption.
