# Finding 10: Prompt Sensitivity — ReAct & IRCoT

**Filed:** June 17, 2026  
**Tags:** `prompt-engineering` `cross-architecture` `sensitivity`

---

## The Finding

Prompt sensitivity varies dramatically by architecture. ReAct has **42-point swings** (examples are critical), IRCoT has **10-point swings** (format-robust). This means one-size-fits-all prompt standardization is impossible — each architecture needs a tailored approach.

## ReAct Results

| Variant | EM | F1 | Cost | Tokens/Q |
|---------|-----|-----|------|----------|
| **v0 Baseline** — 2 worked examples, detailed rules | **48.0%** | 62.0% | $0.038 | ~9,000 |
| v2 Strict — format rules, max 8 steps, no examples | 26.0% | 39.3% | $0.011 | ~3,500 |
| v1 Minimalist — stripped to essentials | 6.0% | 20.6% | $0.024 | ~5,500 |

### Why ReAct collapses without examples

ReAct's Thought/Action/Observation format is **not intuitive** for gpt-4o-mini. Without the worked examples showing "search first, then use finish[]", the model:
- Outputs free-form text instead of structured steps
- Skips retrieval entirely and answers from memory
- Gets confused about the finish[answer] bracket syntax

The 6% EM on minimalist is catastrophic — barely better than random.

## IRCoT Results

| Variant | EM | F1 | Cost | Tokens/Q |
|---------|-----|-----|------|----------|
| **v2 Strict** — format constraints, answer examples | **52.0%** | 64.6% | $0.014 | ~1,800 |
| v0 Baseline — current prompt | 44.0% | 67.8% | $0.040 | ~4,000 |
| v1 Minimalist — stripped | 42.0% | 61.5% | $0.038 | ~3,800 |

### Why IRCoT is robust

IRCoT's "write one reasoning sentence or [ANSWER]" format is **natural** for LLMs. Even with minimal instructions, the model understands the task. The strict variant helps slightly by enforcing the [ANSWER] format more aggressively.

## Cross-Architecture Sensitivity Comparison

| Architecture | Swing | Best | Worst | Sensitivity Pattern |
|-------------|:-----:|:----:|:-----:|-------------------|
| **RLM** | 40 pts | 58% (Strict) | 18% (Minimalist) | Constraints help; verbosity helps |
| **ReAct** | 42 pts | 48% (Baseline) | 6% (Minimalist) | **Examples are essential**; constraints hurt |
| **IRCoT** | 10 pts | 52% (Strict) | 42% (Minimalist) | **Robust**; format barely matters |

## Methodological Implication

This finding is critical for the paper's methodology section:

> *"Prompt sensitivity is architecture-specific. RLM and ReAct show 40+ point swings from prompt changes, while IRCoT is within 10 points. Fair cross-architecture comparison requires architecture-specific prompt optimization — naive 'same format for all' approaches confound architectural differences with prompt quality differences."*

This transforms a weakness ("our prompts may not be equally optimized") into a finding ("architecture-specific prompt sensitivity is itself a measurable property of RAG systems").
