# Finding 5: RLM — The Best Cost-Performance Compromise

**Filed:** June 17, 2026  
**Severity:** 🟢 Positive (core result)  
**Tags:** `rlm` `recursive` `cost-efficiency`

---

## The Finding

Recursive LM achieves the **best absolute performance** (46.3% EM, 60.2% F1) at **4x Vanilla's cost** — the most efficient among non-Vanilla architectures. With the optimized v4 prompt (Finding 1), it reaches **58.0% EM**.

## The Data (default prompt)

| Metric | Value | vs Vanilla | vs ReAct |
|--------|-------|-----------|----------|
| EM | 46.3% | +1.3% | +0.3% |
| F1 | 60.2% | +0.7% | +0.3% |
| Cost | $3.19 | 4.0x | **0.35x** |
| Latency | 7.1s | 5.1x | 0.9x |

## Why RLM Wins

1. **Programmatic recursion** beats unstructured agentic loops — the LLM only decides "direct or decompose," while the recursion logic is handled by code
2. **Memoization** prevents redundant computation — repeated sub-questions are cached
3. **Shallower reasoning** — RLM typically decomposes 1-2 levels, while ReAct can loop 5-10+ times

## The v4 Boost

With the optimized strict prompt (Finding 1), RLM reaches **58.0% EM** — a **25% relative improvement** over the already-best default prompt. The cost drops to $0.012 per question (for 50 questions in the sensitivity study, with LLM cache hits).

## Paper Narrative

RLM represents the **sweet spot**: programmatic control over recursion (no runaway loops) plus retrieval at each step. It proves that "guided reasoning" beats "free-form reasoning" at this model scale.
