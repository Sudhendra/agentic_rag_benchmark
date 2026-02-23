# Planner RAG Must-Fix Pre-Full-Run Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make PlannerRAG paper-aligned, lightweight, and reliable before running full-dataset benchmarks.

**Architecture:** Keep the current inference-only planner loop (gate -> plan actions -> node solve -> synthesize), then tighten correctness and reduce per-question overhead. Align inference behavior with TreePS-RAG principles that transfer to inference (bounded tree growth, diversity-preserving pruning, robust multi-step reasoning) and with RAGShaper principles (robustness to distractors, explicit recovery from bad intermediate outputs). Do not add RL training machinery.

**Tech Stack:** Python 3.11+, pytest/pytest-asyncio, existing PlannerRAG implementation, YAML configs, prompt templates.

---

## Paper-Derived Must-Fix Constraints (Applied to Inference)

1. **Bounded but diverse search:** keep strict max depth/branching while reducing redundant sibling expansions (TreePS-RAG similarity-pruning spirit).
2. **Step quality matters:** avoid brittle one-shot post-processing; support bounded refinement attempts for bridge/compositional questions.
3. **Lightweight execution:** reduce unnecessary per-node calls and repeated retrieval/solve work.
4. **Robustness under noise:** preserve recovery behavior when planner/synthesis outputs generic or invalid answers.
5. **No full-dataset run in this phase:** only unit tests + small subset smoke checks.

---

### Task 1: Add failing tests for must-fix correctness and performance behavior

**Files:**
- Create: `tests/test_planner_paper_alignment.py`
- Modify: `tests/test_planner_rag.py`
- Test: `tests/test_planner_paper_alignment.py`

**Step 1: Write failing tests for bridge refine attempts + disable gates**

```python
@pytest.mark.asyncio
async def test_bridge_refine_respects_max_attempts(mock_llm, mock_retriever, sample_question, sample_corpus):
    # synth answer invalid -> refine invalid -> refine valid on 2nd attempt
    # expect exactly 2 refine prompts when max_attempts=2
    ...


@pytest.mark.asyncio
async def test_bridge_refine_disabled_skips_refine_call(mock_llm, mock_retriever, sample_question, sample_corpus):
    # invalid synth answer, but bridge_refine_enabled=False
    # expect zero refine prompts
    ...
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_planner_paper_alignment.py -k "refine" -v`
Expected: FAIL because current code performs at most one refine attempt and does not cover all gates.

**Step 3: Write failing tests for lightweight solve path + duplicate solve avoidance**

```python
@pytest.mark.asyncio
async def test_single_call_solver_json_path_reduces_llm_calls(...):
    # solver emits {"answer": "Paris", "confidence": 0.93}
    # expect only one solver-side LLM call for that node
    ...


@pytest.mark.asyncio
async def test_repeated_select_same_node_reuses_cached_solve(...):
    # planner selects same node twice
    # expect retriever called once for identical query/top_k
    ...
```

**Step 4: Run tests to verify failure**

Run: `pytest tests/test_planner_paper_alignment.py -k "single_call_solver or cached_solve" -v`
Expected: FAIL because current implementation always does separate confidence call and does no per-question memoization.

**Step 5: Commit tests**

```bash
git add tests/test_planner_paper_alignment.py tests/test_planner_rag.py
git commit -m "test: add planner must-fix pre-full-run coverage"
```

---

### Task 2: Implement bounded multi-attempt bridge refinement + solved-child counting fix

**Files:**
- Modify: `src/architectures/agentic/planner_rag.py`
- Test: `tests/test_planner_paper_alignment.py`

**Step 1: Write/confirm failing tests for attempts and solved-child semantics**

```python
@pytest.mark.asyncio
async def test_child_solved_counter_counts_only_solved_children(...):
    # child remains active (low confidence) -> should not count as solved child
    ...
```

**Step 2: Run targeted failure**

Run: `pytest tests/test_planner_paper_alignment.py -k "child_solved_counter or refine" -v`
Expected: FAIL on current counting and refinement behavior.

**Step 3: Write minimal implementation**

```python
max_attempts = self.config["bridge_refine_max_attempts"]
for attempt in range(max_attempts):
    if not needs_refine(answer_text):
        break
    answer_text, t, c, raw = await self._refine_bridge_answer(...)
    ...

if resolved_node_id != "root" and solve_result["status"] == "solved":
    child_solved_count += 1
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_planner_paper_alignment.py -k "refine or child_solved_counter" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/architectures/agentic/planner_rag.py tests/test_planner_paper_alignment.py
git commit -m "fix: honor planner refine budget and solved-child semantics"
```

---

### Task 3: Implement lightweight single-call solver+confidence with fallback

**Files:**
- Modify: `src/architectures/agentic/planner_rag.py`
- Modify: `prompts/planner_solve.txt`
- Test: `tests/test_planner_paper_alignment.py`

**Step 1: Write failing tests for combined solver response**

```python
@pytest.mark.asyncio
async def test_solver_combined_json_avoids_second_confidence_call(...):
    # solver returns strict JSON with answer/confidence
    # expect no extra confidence prompt call
    ...


@pytest.mark.asyncio
async def test_solver_combined_json_fallbacks_to_confidence_prompt_when_invalid(...):
    # malformed solver output should trigger old confidence fallback
    ...
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_planner_paper_alignment.py -k "combined_json" -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
# in solve prompt ask for strict JSON: {"answer": "...", "confidence": 0.0-1.0}
parsed = self._try_parse_json(answer_text)
if isinstance(parsed, dict) and "answer" in parsed and "confidence" in parsed:
    node_answer = str(parsed["answer"]).strip()
    confidence = self._parse_confidence(str(parsed["confidence"]))
    num_llm_calls = 1
else:
    # fallback: existing confidence prompt path
    ...
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_planner_paper_alignment.py -k "combined_json" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/architectures/agentic/planner_rag.py prompts/planner_solve.txt tests/test_planner_paper_alignment.py
git commit -m "perf: reduce planner solve overhead via combined answer confidence"
```

---

### Task 4: Add per-question memoization for retrieval and solved node outputs

**Files:**
- Modify: `src/architectures/agentic/planner_rag.py`
- Test: `tests/test_planner_paper_alignment.py`

**Step 1: Write failing cache tests**

```python
@pytest.mark.asyncio
async def test_identical_query_uses_retrieval_cache(...):
    # same query/top_k retrieved once per question
    ...


@pytest.mark.asyncio
async def test_reselect_solved_node_reuses_node_solution(...):
    # avoid second solve path when node already solved with evidence
    ...
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_planner_paper_alignment.py -k "retrieval_cache or reuses_node_solution" -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
retrieval_cache: dict[tuple[str, int], RetrievalResult] = {}
node_solve_cache: dict[str, dict[str, Any]] = {}

cache_key = (query, top_k)
if cache_key in retrieval_cache:
    retrieval_result = retrieval_cache[cache_key]
else:
    retrieval_result = await self.retriever.retrieve(...)
    retrieval_cache[cache_key] = retrieval_result

if node.node_id in node_solve_cache and node.status == "solved":
    return node_solve_cache[node.node_id]
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_planner_paper_alignment.py -k "retrieval_cache or reuses_node_solution" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/architectures/agentic/planner_rag.py tests/test_planner_paper_alignment.py
git commit -m "perf: cache planner retrieval and solved-node outputs per question"
```

---

### Task 5: Add rollout diversity pruning (inference-safe, lightweight)

**Files:**
- Modify: `src/architectures/agentic/planner_rag.py`
- Modify: `configs/planner.yaml`
- Modify: `configs/planner_dense.yaml`
- Modify: `configs/planner_hybrid.yaml`
- Modify: `configs/planner_bm25_full.yaml`
- Modify: `configs/planner_dense_full.yaml`
- Modify: `configs/planner_hybrid_full.yaml`
- Test: `tests/test_planner_paper_alignment.py`

**Step 1: Write failing tests for duplicate/near-duplicate sub-question pruning**

```python
def test_rollout_prunes_duplicate_subquestions():
    # ["Who founded X?", "who founded x", "What is X's founder?"]
    # expect deduplicated/pruned list before child creation
    ...
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_planner_paper_alignment.py -k "rollout_prunes" -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
# config defaults
"rollout_similarity_threshold": (float, False, 0.85)

# dedupe before node creation
unique_candidates = self._prune_similar_sub_questions(candidates, threshold=...)
for sub_question in unique_candidates[:max_children]:
    ...
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_planner_paper_alignment.py -k "rollout_prunes" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/architectures/agentic/planner_rag.py configs/planner*.yaml tests/test_planner_paper_alignment.py
git commit -m "perf: prune redundant planner rollout children for diverse search"
```

---

### Task 6: Config/docs consistency and pre-full-run verification

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `tests/test_run_experiment_config.py`
- Modify: `README.md`
- Modify: `docs/TECHNICAL_SPEC.md`

**Step 1: Write failing test for planner max_context_tokens default precedence**

```python
def test_planner_uses_planner_default_context_tokens_when_not_set():
    # ensure planner keeps schema default (4000) unless planner config overrides it
    ...
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_run_experiment_config.py -k planner -v`
Expected: FAIL if `llm.max_tokens` leaks into planner default behavior.

**Step 3: Write minimal implementation + docs update**

```python
elif architecture_name == "planner_rag":
    architecture_config = {
        "top_k": retrieval_config.get("top_k", 5),
        **config.get("planner", {}),
    }
```

Update docs to explicitly state: no full-dataset planner benchmark is run in this phase; full run is deferred until post-merge.

**Step 4: Run verification (no full dataset)**

Run: `pytest tests/test_planner_rag.py tests/test_planner_config.py tests/test_planner_prompt.py tests/test_planner_paper_alignment.py tests/test_run_experiment_config.py -v`
Expected: PASS.

Run (optional smoke only, not full):
- `python scripts/run_experiment.py --config configs/planner.yaml --subset 20`
- `python scripts/run_experiment.py --config configs/planner_dense.yaml --subset 20`
- `python scripts/run_experiment.py --config configs/planner_hybrid.yaml --subset 20`

Expected: successful completion and no planner parser/runtime regressions.

**Step 5: Commit**

```bash
git add scripts/run_experiment.py tests/test_run_experiment_config.py README.md docs/TECHNICAL_SPEC.md
git commit -m "docs: align planner config semantics and pre-full-run status"
```

---

## Execution Discipline

- Use `@superpowers/test-driven-development` for each task.
- If a test fails unexpectedly, switch to `@superpowers/systematic-debugging` before patching.
- Before claiming completion, use `@superpowers/verification-before-completion`.
- Do not run full HotpotQA validation (`subset_size: null`) in this plan.

## Acceptance Criteria (Must Pass Before Full Run)

1. Planner targeted suite passes (including new paper-alignment tests).
2. LLM calls per question on subset smoke runs do not regress vs current README subset baseline (`~8.7-9.2`), with expected improvement on at least one retriever config.
3. No regressions in bridge/comparison handling behaviors already tested.
4. Docs accurately reflect that full benchmark remains deferred until post-merge.
