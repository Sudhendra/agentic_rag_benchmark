# Agentic RAG Benchmark Publication Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the current benchmark prototype into a methodologically defensible, reproducible, publication-ready research artifact, with a realistic path to ACL/EMNLP quality and a stretch path to ICML.

**Architecture:** The work should proceed in three layers. First, fix benchmark validity issues that can invalidate claims regardless of model quality. Second, harden reproducibility and complete the full experiment matrix so every reported number is auditable. Third, add robustness analysis, retrieval diagnostics, and a paper narrative that elevates the contribution from an engineering comparison to a scientific study of reasoning paradigms under cost and retrieval constraints.

**Tech Stack:** Python 3.11+, pytest, HuggingFace datasets, OpenAI API, optional Anthropic API, YAML configs, MLflow, JSON/JSONL artifacts.

---

## Success Criteria

1. Every benchmark claim in `README.md` is backed by a checked-in or externally released artifact set.
2. HotpotQA and 2Wiki runs report answer EM/F1 plus supporting-fact and joint metrics.
3. Retrieval behavior is correct, tested, and aligned with dataset semantics.
4. The full architecture x retriever x dataset matrix is complete, or the paper scope is explicitly narrowed.
5. The paper can defend not just aggregate wins, but when and why each paradigm succeeds or fails.

---

### Task 1: Fix Corpus Semantics And Retrieval Validity

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `src/evaluation/evaluator.py`
- Modify: `src/data/hotpotqa.py`
- Modify: `src/data/musique.py`
- Modify: `src/data/wiki2hop.py`
- Test: `tests/test_evaluator.py`
- Test: `tests/test_run_experiment_config.py`
- Create: `tests/test_dataset_corpus_semantics.py`

**Step 1: Write the failing tests**

Add tests that assert distractor-style datasets are evaluated against the intended corpus scope per question, not an unrelated flattened global corpus. Add at least one test that constructs two questions with overlapping titles and verifies retrieval for question A cannot see question B's distractors unless the benchmark protocol explicitly allows it.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_corpus_semantics.py tests/test_evaluator.py -v`

Expected: FAIL because the current runner passes one global corpus into `Evaluator.evaluate()` and then into `rag.answer()`.

**Step 3: Write minimal implementation**

Refactor data loading and evaluation so each question can carry or resolve its own candidate corpus when required by dataset semantics. Keep the public benchmark surface small: one explicit mechanism for per-question corpus resolution is better than multiple special cases.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_corpus_semantics.py tests/test_evaluator.py tests/test_run_experiment_config.py -v`

Expected: PASS.

**Step 5: Commit**

`git add scripts/run_experiment.py src/evaluation/evaluator.py src/data/hotpotqa.py src/data/musique.py src/data/wiki2hop.py tests/test_dataset_corpus_semantics.py tests/test_evaluator.py tests/test_run_experiment_config.py && git commit -m "fix: align benchmark corpus handling with dataset semantics"`

### Task 2: Fix Retriever Correctness Bugs

**Files:**
- Modify: `src/core/retriever.py`
- Modify: `src/retrieval/dense.py`
- Test: `tests/test_retriever_factory.py`
- Create: `tests/test_retriever_indexing.py`
- Create: `tests/test_dense_retriever.py`

**Step 1: Write the failing tests**

Add tests for:
- re-indexing when corpus identity changes but corpus length stays constant
- cached dense embeddings being normalized exactly like fresh embeddings
- consistent top-k ordering between fresh and cached embedding paths

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever_indexing.py tests/test_dense_retriever.py -v`

Expected: FAIL on current `_ensure_indexed()` logic and cached embedding normalization behavior.

**Step 3: Write minimal implementation**

Make corpus invalidation explicit and deterministic. Normalize embeddings before saving to cache or normalize on load before use. Avoid introducing multiple cache representations.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever_indexing.py tests/test_dense_retriever.py tests/test_retriever_factory.py -v`

Expected: PASS.

**Step 5: Commit**

`git add src/core/retriever.py src/retrieval/dense.py tests/test_retriever_indexing.py tests/test_dense_retriever.py tests/test_retriever_factory.py && git commit -m "fix: harden retriever indexing and dense cache consistency"`

### Task 3: Complete Multi-Hop Evaluation Metrics

**Files:**
- Modify: `src/evaluation/evaluator.py`
- Modify: `src/evaluation/metrics.py`
- Modify: `src/utils/results.py`
- Modify: `configs/base.yaml`
- Test: `tests/test_evaluator.py`
- Test: `tests/test_metrics.py`
- Create: `tests/test_supporting_fact_evaluation.py`

**Step 1: Write the failing tests**

Add evaluator tests that construct questions with supporting facts and verify:
- supporting-fact EM/F1 are computed when enabled
- joint EM/F1 are populated
- saved predictions include these fields consistently

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_supporting_fact_evaluation.py tests/test_evaluator.py tests/test_metrics.py -v`

Expected: FAIL because the evaluator currently hardcodes supporting-fact metrics to `None`.

**Step 3: Write minimal implementation**

Honor `evaluation.compute_supporting_facts`. If an architecture does not emit supporting facts yet, record that explicitly and document the limitation instead of silently returning `None` everywhere. For datasets with gold supporting facts, compute the answer-only and joint metrics in one place.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_supporting_fact_evaluation.py tests/test_evaluator.py tests/test_metrics.py -v`

Expected: PASS.

**Step 5: Commit**

`git add src/evaluation/evaluator.py src/evaluation/metrics.py src/utils/results.py configs/base.yaml tests/test_supporting_fact_evaluation.py tests/test_evaluator.py tests/test_metrics.py && git commit -m "feat: add supporting fact and joint multi-hop evaluation"`

### Task 4: Wire Reproducibility Controls End-To-End

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `configs/base.yaml`
- Modify: `src/utils/results.py`
- Modify: `src/architectures/vanilla_rag.py`
- Modify: `src/architectures/agentic/react_rag.py`
- Modify: `src/architectures/agentic/self_rag.py`
- Modify: `src/architectures/agentic/planner_rag.py`
- Modify: `src/architectures/recursive/ircot.py`
- Modify: `src/architectures/recursive/reap.py`
- Modify: `src/architectures/rlm/recursive_lm.py`
- Test: `tests/test_run_experiment_config.py`
- Create: `tests/test_generation_config_propagation.py`

**Step 1: Write the failing tests**

Add tests that verify `temperature`, `max_tokens`, seed metadata, and unique run directory behavior are propagated from config into execution and saved artifacts.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_experiment_config.py tests/test_generation_config_propagation.py -v`

Expected: FAIL because temperature/seed are not fully wired and manual run IDs can collide.

**Step 3: Write minimal implementation**

Propagate generation parameters through the runner into every architecture callsite. Record the effective seed in artifacts even if the provider is only approximately deterministic. Replace `results/manual` fallback with a timestamp or UUID-based run directory.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_experiment_config.py tests/test_generation_config_propagation.py -v`

Expected: PASS.

**Step 5: Commit**

`git add scripts/run_experiment.py configs/base.yaml src/utils/results.py src/architectures/vanilla_rag.py src/architectures/agentic/react_rag.py src/architectures/agentic/self_rag.py src/architectures/agentic/planner_rag.py src/architectures/recursive/ircot.py src/architectures/recursive/reap.py src/architectures/rlm/recursive_lm.py tests/test_run_experiment_config.py tests/test_generation_config_propagation.py && git commit -m "fix: make experiment configuration reproducible end to end"`

### Task 5: Expand Validation Tests For Datasets And Retrieval

**Files:**
- Create: `tests/test_musique_loader.py`
- Create: `tests/test_wiki2hop_loader.py`
- Create: `tests/test_bm25_retriever.py`
- Create: `tests/test_hybrid_retriever.py`
- Modify: `tests/test_hotpotqa.py`

**Step 1: Write the tests**

Add loader tests matching the quality bar already present for HotpotQA. Add retriever tests for ranking sanity, top-k truncation, empty query behavior, and deterministic output under mock embeddings.

**Step 2: Run tests to verify baseline behavior**

Run: `pytest tests/test_hotpotqa.py tests/test_musique_loader.py tests/test_wiki2hop_loader.py tests/test_bm25_retriever.py tests/test_hybrid_retriever.py -v`

Expected: FAIL until new coverage is implemented and adjusted to actual retriever behavior.

**Step 3: Write minimal implementation**

Only patch production code if tests reveal real bugs. Do not refactor the loaders or retrievers just to satisfy test aesthetics.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hotpotqa.py tests/test_musique_loader.py tests/test_wiki2hop_loader.py tests/test_bm25_retriever.py tests/test_hybrid_retriever.py -v`

Expected: PASS.

**Step 5: Commit**

`git add tests/test_hotpotqa.py tests/test_musique_loader.py tests/test_wiki2hop_loader.py tests/test_bm25_retriever.py tests/test_hybrid_retriever.py && git commit -m "test: expand dataset and retriever correctness coverage"`

### Task 6: Add Retrieval Diagnostics And Statistical Analysis

**Files:**
- Modify: `src/core/types.py`
- Modify: `src/evaluation/evaluator.py`
- Modify: `src/utils/results.py`
- Modify: `scripts/analyze_results.py`
- Create: `tests/test_analysis_metrics.py`
- Create: `docs/plans/2026-04-03-analysis-protocol.md`

**Step 1: Write the failing tests**

Add analysis tests for retrieval-oriented metrics and aggregate statistics helpers, including confidence intervals or bootstrap summaries on stored predictions.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis_metrics.py -v`

Expected: FAIL because retrieval diagnostics and significance tooling do not yet exist.

**Step 3: Write minimal implementation**

Track the minimum additional fields needed for publication-quality analysis, such as evidence recall proxies, retrieval hit rates, and confidence interval summaries. Keep raw artifact format backward-compatible if possible.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis_metrics.py tests/test_results.py -v`

Expected: PASS.

**Step 5: Commit**

`git add src/core/types.py src/evaluation/evaluator.py src/utils/results.py scripts/analyze_results.py tests/test_analysis_metrics.py docs/plans/2026-04-03-analysis-protocol.md && git commit -m "feat: add retrieval diagnostics and statistical analysis utilities"`

### Task 7: Finish Benchmark Matrix Coverage

**Files:**
- Modify: `src/core/llm_client.py`
- Modify: `configs/*.yaml`
- Modify: `README.md`
- Modify: `NEXT_STEPS.md`
- Test: `tests/test_llm_client.py`
- Create: `tests/test_anthropic_client.py`
- Create: `docs/plans/2026-04-03-run-manifest.md`

**Step 1: Write the failing tests**

Add tests for `AnthropicClient` creation, response normalization, and cache behavior. Add a run-manifest document listing every required experiment for the paper.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_client.py tests/test_anthropic_client.py -v`

Expected: FAIL because Anthropic is unimplemented.

**Step 3: Write minimal implementation**

Implement `AnthropicClient` only to the level needed for parity with `OpenAIClient`: generation, cost tracking, caching, and configuration. Then add configs for at least one Claude model. If time or budget blocks this, explicitly downgrade the paper claim to a single-provider study and remove the optional-provider framing from docs.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_client.py tests/test_anthropic_client.py -v`

Expected: PASS.

**Step 5: Commit**

`git add src/core/llm_client.py configs README.md NEXT_STEPS.md tests/test_llm_client.py tests/test_anthropic_client.py docs/plans/2026-04-03-run-manifest.md && git commit -m "feat: add second-provider benchmarking support"`

### Task 8: Run And Archive The Full Experiment Matrix

**Files:**
- Create: `results/<run_id>/summary.json`
- Create: `results/<run_id>/predictions.jsonl`
- Create: `results/<run_id>/resolved_config.yaml`
- Modify: `README.md`
- Modify: `NEXT_STEPS.md`
- Create: `docs/plans/2026-04-03-release-artifacts.md`

**Step 1: Create the run manifest**

List the required runs explicitly:
- HotpotQA: 7 architectures x 3 retrievers
- MuSiQue: 7 architectures x 3 retrievers
- 2Wiki: 7 architectures x 3 retrievers
- optional second-provider subset matrix if budget-limited

**Step 2: Execute the runs in a controlled order**

Run a small smoke subset first for each config, then full runs. Save every resolved config and artifact bundle. Do not update the README tables until all numbers are traceable to artifacts.

**Step 3: Verify artifact completeness**

Run: `python scripts/analyze_results.py --results results/ --compare`

Expected: every row in the paper summary is reconstructible from stored artifacts.

**Step 4: Update summary docs**

Update `README.md` and `NEXT_STEPS.md` only after all run IDs and metrics are verified.

**Step 5: Commit**

`git add results README.md NEXT_STEPS.md docs/plans/2026-04-03-release-artifacts.md && git commit -m "docs: archive reproducible benchmark artifacts"`

### Task 9: Add Robustness Studies Required For A Strong Paper

**Files:**
- Create: `docs/plans/2026-04-03-robustness-study.md`
- Modify: `scripts/analyze_results.py`
- Modify: `README.md`

**Step 1: Define the minimum robustness suite**

Include:
- prompt sensitivity for Recursive LM and at least one agentic method
- variance across repeated runs where nondeterminism remains
- budget-constrained comparison curves
- hop-stratified and question-type stratified comparisons across datasets

**Step 2: Run the study**

Use a reduced but representative subset if full-matrix repetition is too expensive, but document exactly why and how it was sampled.

**Step 3: Analyze and summarize**

Use `scripts/analyze_results.py` or a companion analysis notebook/script to produce publication figures and uncertainty estimates.

**Step 4: Update the public narrative**

Revise claims so they depend on stable findings, not one-off leaderboard numbers.

**Step 5: Commit**

`git add docs/plans/2026-04-03-robustness-study.md scripts/analyze_results.py README.md && git commit -m "docs: add robustness analysis for publication claims"`

### Task 10: Write The Paper To Match The Evidence

**Files:**
- Create: `docs/plans/2026-04-03-paper-outline.md`
- Modify: `README.md`
- Modify: `NEXT_STEPS.md`

**Step 1: Write the outline**

Use this structure:
- Introduction: why paradigm-level comparison matters under API-only constraints
- Benchmark Protocol: datasets, retrieval semantics, prompt fidelity, evaluation metrics
- Methods: exact implementation plus deviations from source papers
- Results: aggregate, cost, latency, retrieval, robustness
- Analysis: hop depth, failure modes, prompt sensitivity, cost-performance frontier
- Limitations: implementation approximations, provider scope, dataset constraints

**Step 2: Write the claim discipline section**

For each architecture, include a short faithfulness note: what is faithful, what is approximated, what was changed for API-only operation.

**Step 3: Decide the title and venue framing**

Primary framing for a strong NLP paper:
- comparative study of reasoning paradigms for multi-hop QA under API-only constraints

Stretch framing for ICML:
- cost-robustness tradeoffs and benchmark methodology for modern retrieval-reasoning systems

**Step 4: Update project docs**

Ensure `README.md` no longer overstates unsupported conclusions.

**Step 5: Commit**

`git add docs/plans/2026-04-03-paper-outline.md README.md NEXT_STEPS.md && git commit -m "docs: align paper narrative with benchmark evidence"`

---

## Publication Strategy

### Realistic venue ladder

1. **Primary realistic target:** ACL / EMNLP / NAACL
2. **Stretch target:** ICML only if Tasks 1-9 are completed and the final paper emphasizes scientific insight, robustness, and methodology rather than raw benchmarking alone
3. **Fallback if scope slips:** Workshop paper plus benchmark release

### ICML bar to clear

The project should only be submitted to ICML if all of the following are true:
- benchmark semantics are methodologically defensible
- claims are backed by auditable artifacts
- results include robustness and significance analysis
- the paper contributes a broader scientific insight than "architecture A beats architecture B"

### Recommended contribution framing

Do not frame the paper as a simple leaderboard.

Frame it as a study of:
- when agentic, recursive, and recursive-LM paradigms win
- how much they cost to win
- how retrieval quality and prompt sensitivity affect the conclusions
- how to benchmark these systems fairly under API-only constraints

---

## Recommended Execution Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7
8. Task 8
9. Task 9
10. Task 10

Do not run the expensive full matrix before Tasks 1-4 are complete.

---

Plan complete and saved to `docs/plans/2026-04-03-icml-publication-roadmap.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
