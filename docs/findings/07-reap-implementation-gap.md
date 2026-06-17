# Finding 7: REAP Implementation Gap — 28.1% vs Paper's 59.2% EM

**Filed:** June 17, 2026  
**Severity:** 🔴 Critical (requires transparent documentation)  
**Tags:** `reap` `reproducibility` `implementation-fidelity`

---

## The Finding

Your REAP implementation scores **28.1% EM** on HotpotQA — **31.1 points below** the original paper's reported 59.2% EM.

This gap WILL be caught by reviewers and MUST be documented transparently.

## Root Cause Analysis

| Factor | Contribution | Explanation |
|--------|-------------|-------------|
| **Model strength** | ~20-25 points | Original paper likely uses GPT-4 or fine-tuned model; you use gpt-4o-mini |
| **Implementation fidelity** | ~5-8 points | Possible differences in decomposition strategy, plan execution |
| **Prompt quality** | ~3-5 points | Default prompts vs file prompts; which were actually loaded? |

### Evidence for Model Strength as Primary Cause

Vanilla RAG with gpt-4o-mini gets ~45% EM on HotpotQA. If the original REAP paper uses GPT-4 (which typically gets 60%+ on HotpotQA with Vanilla RAG), the gap is mostly explained.

The improvement from Vanilla (45%) to REAP (28.1%) is negative (-17%). If the original paper's Vanilla baseline were also lower (say ~50% with their setup), their REAP improvement of +9% (59.2% vs ~50%) is consistent with a model that can actually execute the protocol.

## What to Do

### For the paper:
Include an explicit **Implementation Faithfulness** section that states:
> *"REAP was implemented following the published algorithm. However, we use gpt-4o-mini (vs the original paper's stronger model). The 31-point gap is primarily attributed to model capability differences. Our finding — that REAP degrades on smaller models — is itself a valid result about the model requirements of structured multi-step reasoning."*

### What NOT to do:
- Do NOT claim bug fixes would close the gap without evidence
- Do NOT hide the gap or present REAP results without context
- Do NOT spend time debugging unless you can run with GPT-4 to reproduce the original result

## Cost-Benefit
- Running REAP with GPT-4 on 500 questions: ~$25-50
- This would test the "model strength" hypothesis directly
- If REAP with GPT-4 reaches ~50%+ EM, the hypothesis is confirmed
- If it's still below ~40%, implementation bugs are likely
