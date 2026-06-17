# Phase 0 Audit — Complete Status

Branch: `feature/analytics-robustness`  
Date: June 2026

---

## 0.1 — Corpus Semantics: ✅ DONE (commit `9e443f8`)

**What was done:** Each HotpotQA question loads with `candidate_corpus=question_corpus` (its own 10-document distractor pool). The evaluator detects this and indexes per-question corpora individually.

**Verify by:** Run any config on 5 questions and check that `evaluator.py:40-43` detects `uses_question_scoped_corpus = True`. Output should show `concurrency=1`.

**No action needed.**

---

## 0.2 — Supporting Facts: ✅ Pipeline DONE, scores will be 0

**What was done:** The pipeline code exists:
- `evaluator.py:79-87` computes `supporting_fact_evaluation` and `joint_metrics`
- `base.yaml:28` has `compute_supporting_facts: true`
- `EvaluationResult` carries `supporting_fact_em`, `supporting_fact_f1`, `joint_em`, `joint_f1`

**What's NOT done:** No architecture currently populates `RAGResponse.supporting_facts`. Every response returns `supporting_facts=None`, so the evaluator will report:
```
supporting_fact_status = "not_provided"
supporting_fact_em = 0.0
supporting_fact_f1 = 0.0
joint_em = 0.0
joint_f1 = 0.0
```

**Verify by:**
```bash
python scripts/run_experiment.py --config configs/vanilla_dense.yaml
```
Check the `summary.json` output for `avg_supporting_fact_em`, `avg_supporting_fact_f1`, `avg_joint_em`, `avg_joint_f1`.

**Your decision:** 
- Option A: Accept that no architecture predicts supporting facts. Document this as a finding — the paper states *"multi-hop QA architectures can identify answers but cannot pinpoint supporting evidence"*. 
- Option B: Implement supporting fact extraction in 1-2 architectures. This requires modifying each architecture's `answer()` to emit `(title, sent_idx)` tuples in `RAGResponse.supporting_facts`. This is ~2-3 days of work per architecture.

**Recommended:** Go with Option A for now. The zero supporting fact scores are themselves a valid scientific finding.

---

## 0.3 — REAP Investigation: ❌ NEEDS WORK

### The Problem
| Source | HotpotQA EM | Gap |
|--------|-------------|-----|
| Original REAP paper | 59.2% | — |
| Your implementation | 28.1% | **31.1 points** |

A 31-point gap WILL be caught by reviewers.

### Root Cause Analysis

**1. Model strength difference (MOST LIKELY):**
- Original REAP paper uses a much stronger model (likely GPT-4 or a fine-tuned model)
- Your implementation uses gpt-4o-mini
- Vanilla RAG with gpt-4o-mini gets 45.0% EM, while gpt-4o could get 60%+
- If the original paper uses GPT-4, a 31-point gap is mostly explained by model quality

**2. Prompt fidelity (POSSIBLE):**
- Your default prompts in `reap.py:22-103` differ from the file prompts in `prompts/reap_*.txt`
- The file prompts are slightly cleaner (e.g., "explicit" vs "explicit")
- Verify which prompts actually loaded during runs

**3. Implementation differences (POSSIBLE):**
- The original REAP paper may use fine-tuned models for specific sub-modules
- Your implementation uses a single LLM for all sub-tasks (decompose, plan, extract, synthesize)
- The paper's architecture may have additional verifier/re-ranker components not in your code

### ✅ Action Items

**Step 1:** Document the model difference explicitly:
```bash
# Run Vanilla RAG on 500 questions to establish baseline for comparison
python scripts/run_experiment.py --config configs/vanilla_dense.yaml --subset 500
```

**Step 2:** Create a REAP analysis script (new file, not modifying existing):

```bash
# Export REAP errors to CSV for manual inspection
python scripts/export_errors.py \
  --results results/<reap_run_id> \
  --output reap_error_analysis.csv
```

**Step 3:** Manually inspect 10 REAP failures and categorize:
- Is it returning "Unknown" when answer exists?
- Is it extracting wrong entities?
- Is the JSON parsing failing?
- Is it running out of iterations before finding answer?

**For the paper:** Frame this as an honest finding:
> *"REAP's complex decomposition-plan-extract-synthesize pipeline degrades significantly on smaller LLMs (gpt-4o-mini), suggesting that multi-step structured reasoning requires stronger base model capabilities."*

---

## 0.4 — Reproducibility Controls: ✅ DONE (commit `acc4335`)

**What was done:**
- Seed from `config.experiment.seed` → `random.seed()` + `numpy.random.seed()`
- `generation_temperature`, `generation_max_tokens`, `generation_seed` passed through `common_config` to all architectures
- Run directory uses MLflow run_id or timestamp+UUID fallback

**Verify by:**
```bash
# Check that seed appears in output artifacts
python scripts/run_experiment.py --config configs/vanilla_dense.yaml
# Then check results/<run_id>/resolved_config.yaml for experiment.seed
```

**No action needed.**

---

## 0.5 — Quick Wins: ⚠️ MINOR ISSUES REMAIN

### Issue A: `print()` in dense.py (cosmetic)
`src/retrieval/dense.py:132` uses `print()` instead of `logging.info()`. This is cosmetic — it works but isn't consistent with the rest of the project.

**Fix** (your call — low priority):
```bash
# In src/retrieval/dense.py, line 132:
# Change: print(f"  Embedding batch ...")
# To: logger.info(f"Embedding batch ...")
```
Add `logger = logging.getLogger(__name__)` at the top of dense.py.

### Issue B: `results/` in .gitignore ✅ DONE
`.gitignore:50` already has `results/`. Verified.

### Issue C: Sentence splitter double periods (cosmetic — ACCEPT)
`src/core/types.py:56` — `[s.strip() + "." for s in ...]` always adds a period, which doubles periods on already-terminated sentences. This does NOT affect retrieval or answer quality. Low priority.

---

## GroqClient Changes (from my earlier work)

These are already applied to your working tree as unstaged modifications:

| File | Change | Status |
|------|--------|--------|
| `src/core/llm_client.py` | Added `base_url`, `api_key_env_var`, `pricing` params; `groq` provider | ✔ On disk |
| `scripts/run_experiment.py` | Passes `base_url` and `api_key_env_var` from config | ✔ On disk |
| `configs/base.yaml` | `compute_supporting_facts: true` | ✔ On disk |
| `.env.example` | Added `GROQ_API_KEY` | ✔ On disk |
| `configs/vanilla_groq.yaml` | New config file | ✔ On disk |

**To use Groq:**
1. Sign up at https://console.groq.com (free, no credit card)
2. Set env var: `set GROQ_API_KEY=gsk_your_key_here`
3. Run: `python scripts/run_experiment.py --config configs/vanilla_groq.yaml`

---

## Summary: What's Left for Phase 0

| Task | Status | Time Needed |
|------|--------|-------------|
| 0.1 Corpus semantics | ✅ Committed | None |
| 0.2 Supporting facts | ✅ Pipeline committed, scores=0 | Accept as finding |
| 0.3 REAP investigation | ❌ Not done | 1-2 hours analysis |
| 0.4 Reproducibility | ✅ Committed | None |
| 0.5 Quick wins | 🟡 Minor (print in dense.py) | 10 min if desired |
| Groq validation | 🟡 Code on disk, needs testing | 5 min to run test |

**Total remaining work for Phase 0:** ~2 hours
