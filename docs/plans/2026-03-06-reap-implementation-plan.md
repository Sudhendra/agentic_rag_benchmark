# REAP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a faithful subset-ready REAP recursive architecture with explicit plan state, fact extraction, plan repair, tuning hooks, and full benchmark wiring for later validation.

**Architecture:** REAP will be implemented as a recursive architecture that performs (1) upfront decomposition into plan requirements, (2) iterative planner/fact-extractor rounds where each active requirement is retrieved and converted into evidence-backed facts labeled as `DIRECT_ANSWER`, `PARTIAL_CLUE`, or `FAILED_EXTRACT`, (3) plan updates or replanning based on those fulfillment states, and (4) final answer synthesis over collected facts. This keeps REAP distinct from IRCoT’s reasoning-step retrieval loop and Recursive LM’s recursive sub-question solver while remaining simple enough for API-only subset experimentation.

**Tech Stack:** Python 3.11+, asyncio, existing `BaseRAG` / `RAGResponse` interfaces, existing retrievers, YAML configs, prompt templates, pytest + pytest-asyncio, OpenAI/Anthropic clients via the repo’s `BaseLLMClient`.

---

### Task 1: Add failing REAP architecture and helper tests

**Files:**
- Create: `tests/test_reap_rag.py`
- Test: `tests/test_architecture_factory.py`
- Test: `tests/test_run_experiment_config.py`

**Step 1: Write the failing test**

Add focused tests for the public contract and the core REAP helper behavior.

```python
def test_get_name():
    rag = REAPRAG(AsyncMock(model="test-model"), AsyncMock(), {})
    assert rag.get_name() == "reap_rag"
```

```python
def test_config_defaults():
    rag = REAPRAG(AsyncMock(model="test-model"), AsyncMock(), {})
    assert rag.config["max_iterations"] == 5
    assert rag.config["top_k"] == 3
    assert rag.config["max_active_requirements"] == 2
```

```python
def test_extract_json_block_parses_llm_output():
    payload = REAPRAG._extract_json_block('analysis... {"next_step": "SYNTHESIZE_ANSWER"}')
    assert payload["next_step"] == "SYNTHESIZE_ANSWER"
```

```python
def test_mark_requirements_resolved_only_for_direct_answers():
    plan = [{"requirement_id": "r1", "question": "Who...", "status": "pending"}]
    facts = [{"fulfills_requirement_id": "r1", "fulfillment_level": "DIRECT_ANSWER"}]
    resolved = REAPRAG._mark_resolved_requirements(plan, facts)
    assert resolved[0]["status"] == "resolved"
```

Factory/runner failing coverage:

```python
def test_create_architecture_reap_merges_top_level_config(mock_llm, mock_retriever):
    rag = create_architecture("reap_rag", mock_llm, mock_retriever, {"top_k": 7, "reap": {"max_iterations": 4}})
    assert rag.config["top_k"] == 7
    assert rag.config["max_iterations"] == 4
```

```python
def test_build_rag_uses_reap_nested_config(monkeypatch):
    ...
    assert architecture_config["max_iterations"] == 4
    assert architecture_config["max_context_tokens"] == 3000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_reap_rag.py tests/test_architecture_factory.py tests/test_run_experiment_config.py -q`

Expected: FAIL with missing imports / unknown architecture errors because REAP is not implemented yet.

**Step 3: Write minimal implementation**

Add only enough code to make one happy-path helper test pass:
- create `src/architectures/recursive/reap.py`
- define `REAPRAG`
- implement `get_name`, `get_type`, `get_config_schema`
- implement minimal JSON extraction / requirement status helpers

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_reap_rag.py::test_extract_json_block_parses_llm_output -q`

Expected: PASS

**Step 5: Commit**

```bash
git add src/architectures/recursive/reap.py tests/test_reap_rag.py tests/test_architecture_factory.py tests/test_run_experiment_config.py
git commit -m "test: add REAP architecture scaffolding coverage"
```

### Task 2: Implement the decomposition -> plan -> fact extraction -> synthesis loop

**Files:**
- Create: `src/architectures/recursive/reap.py`
- Create: `prompts/reap_decompose.txt`
- Create: `prompts/reap_extract.txt`
- Create: `prompts/reap_plan.txt`
- Create: `prompts/reap_replan.txt`
- Create: `prompts/reap_synthesize.txt`
- Test: `tests/test_reap_rag.py`

**Step 1: Write the failing test**

Add tests that lock in REAP’s identity as planner/fact-extractor rather than IRCoT-style reasoning.

```python
@pytest.mark.asyncio
async def test_reap_handles_direct_fact_extraction_flow(mock_llm, mock_retriever, bridge_question, sample_corpus):
    mock_llm.generate.side_effect = [
        ('{"user_goal": "...", "requirements": [{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": []}, {"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": ["r1"]}]}', 20, 0.001),
        ('{"next_step": "EXECUTE", "updated_plan": [...], "next_actions": [{"requirement_id": "r1", "question": "What is the capital of France?"}]}', 20, 0.001),
        ('{"reasoned_facts": [{"reasoning": "Paris is directly stated.", "direct_evidence": "Paris is the capital of France.", "statement": "The capital of France is Paris.", "fulfills_requirement_id": "r1", "fulfillment_level": "DIRECT_ANSWER"}]}', 20, 0.001),
        ('{"next_step": "EXECUTE", "updated_plan": [...], "next_actions": [{"requirement_id": "r2", "question": "Who is the mayor of Paris?"}]}', 20, 0.001),
        ('{"reasoned_facts": [{"reasoning": "Anne Hidalgo is directly stated.", "direct_evidence": "Anne Hidalgo is the mayor of Paris.", "statement": "The mayor of Paris is Anne Hidalgo.", "fulfills_requirement_id": "r2", "fulfillment_level": "DIRECT_ANSWER"}]}', 20, 0.001),
        ('Anne Hidalgo', 10, 0.001),
    ]
    ...
    assert response.answer == "Anne Hidalgo"
    assert [step.action for step in response.reasoning_chain] == ["decompose", "plan", "extract", "plan", "extract", "synthesize"]
```

```python
@pytest.mark.asyncio
async def test_reap_uses_replanner_after_partial_clue(...):
    ...
    assert "replan" in [step.action for step in response.reasoning_chain]
```

```python
def test_extract_reasoned_facts_requires_fulfillment_levels():
    payload = REAPRAG._parse_reasoned_facts({"reasoned_facts": [...]})
    assert payload[0]["fulfillment_level"] == "PARTIAL_CLUE"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_reap_rag.py -q`

Expected: FAIL because the REAP loop, fact parsing, and prompts do not exist yet.

**Step 3: Write minimal implementation**

Implement in `src/architectures/recursive/reap.py`:
- explicit plan state: list of requirement dicts with `requirement_id`, `question`, `depends_on`, `status`
- fact state: list of fact dicts with `reasoning`, `direct_evidence`, `statement`, `fulfills_requirement_id`, `fulfillment_level`
- decomposition LLM call
- planner vs replanner split
- retrieval per active executable requirement
- fact extraction over retrieved docs + known facts
- requirement resolution only for `DIRECT_ANSWER`
- final synthesis over facts
- reasoning trace with actions `decompose`, `plan`, `replan`, `extract`, `synthesize`
- robust JSON extraction from LLM output fences / prose wrappers

Keep v1 intentionally simple:
- no deep branching tree explosion
- execute up to `max_active_requirements` actions per planner round
- no recursive `answer()` calls
- keep evidence snippets short and use existing `_build_context()` for token control

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_reap_rag.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add src/architectures/recursive/reap.py prompts/reap_*.txt tests/test_reap_rag.py
git commit -m "feat: implement REAP planning and fact extraction loop"
```

### Task 3: Wire REAP into factory, configs, and runner

**Files:**
- Modify: `src/architectures/factory.py`
- Modify: `src/architectures/recursive/__init__.py`
- Modify: `scripts/run_experiment.py`
- Create: `configs/reap.yaml`
- Create: `configs/reap_dense.yaml`
- Create: `configs/reap_hybrid.yaml`
- Create: `configs/reap_bm25_full.yaml`
- Create: `configs/reap_dense_full.yaml`
- Create: `configs/reap_hybrid_full.yaml`
- Create: `tests/test_reap_config.py`
- Test: `tests/test_architecture_factory.py`
- Test: `tests/test_run_experiment_config.py`

**Step 1: Write the failing test**

Add wiring/config tests:

```python
def test_reap_yaml_config_loads():
    config = load_config(Path("configs/reap.yaml"))
    assert config["architecture"]["name"] == "reap_rag"
    assert config["reap"]["max_iterations"] == 5
```

```python
def test_reap_subset_config_uses_small_dev_subset():
    config = load_config(Path("configs/reap.yaml"))
    assert config["data"]["subset_size"] == 100
```

```python
def test_build_rag_uses_reap_nested_config(monkeypatch):
    ...
    assert architecture_config["max_iterations"] == 5
    assert architecture_config["top_k"] == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_reap_config.py tests/test_architecture_factory.py tests/test_run_experiment_config.py -q`

Expected: FAIL because REAP factory/runner/config wiring is absent.

**Step 3: Write minimal implementation**

Update wiring so:
- architecture name is `reap_rag`
- nested config key is `reap`
- runner passes retrieval + context settings into REAP like IRCoT/RLM
- `configs/reap.yaml` defaults to BM25 + subset 100
- retriever variants inherit from `reap.yaml`
- full configs use `subset_size: null`

Suggested dev config:

```yaml
inherits: base.yaml

experiment:
  name: "reap"

architecture:
  name: "reap_rag"

reap:
  max_iterations: 5
  max_active_requirements: 2
  max_context_tokens: 3000
  max_docs: 8

retrieval:
  method: "bm25"
  top_k: 3

data:
  subset_size: 100
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_reap_config.py tests/test_architecture_factory.py tests/test_run_experiment_config.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add src/architectures/factory.py src/architectures/recursive/__init__.py scripts/run_experiment.py configs/reap*.yaml tests/test_reap_config.py tests/test_architecture_factory.py tests/test_run_experiment_config.py
git commit -m "feat: wire REAP configs and runner support"
```

### Task 4: Tune REAP on subset traces and lock in regressions

**Files:**
- Modify: `src/architectures/recursive/reap.py`
- Modify: `prompts/reap_*.txt`
- Test: `tests/test_reap_rag.py`

**Step 1: Write the failing test**

After the first smoke run, add regression tests for any observed failure modes such as:
- planner repeating same unresolved requirement forever
- fact extractor marking weak evidence as `DIRECT_ANSWER`
- dependency substitution not happening after a direct answer
- synthesis preferring noisy facts over exact slot values

Example:

```python
def test_select_synthesis_candidate_prefers_direct_answer_fact():
    facts = [
        {"statement": "The capital of France is Paris.", "fulfillment_level": "DIRECT_ANSWER"},
        {"statement": "France is in Europe.", "fulfillment_level": "PARTIAL_CLUE"},
    ]
    selected = REAPRAG._select_direct_answer_fact("What is the capital of France?", facts)
    assert selected == "The capital of France is Paris."
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_reap_rag.py -q`

Expected: FAIL for the newly added regression(s).

**Step 3: Write minimal implementation**

Tune only the observed weak points:
- planner loop guards against no-progress repetition
- direct-answer selection prefers exact requirement-fulfilling facts
- active-requirement prompts become more specific if traces show drift
- cap active branches for cheap subset validation

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_reap_rag.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add src/architectures/recursive/reap.py prompts/reap_*.txt tests/test_reap_rag.py
git commit -m "refactor: tune REAP planning and synthesis on subset traces"
```

### Task 5: Docs, verification, audit, and PR readiness

**Files:**
- Modify: `README.md`
- Modify: `NEXT_STEPS.md`
- Modify: `docs/TECHNICAL_SPEC.md`
- Verify: `tests/test_reap_rag.py`
- Verify: `tests/test_reap_config.py`
- Verify: `tests/test_architecture_factory.py`
- Verify: `tests/test_run_experiment_config.py`

**Step 1: Write the failing test**

If needed, add lightweight assertions for config/documented subset workflow:

```python
def test_reap_full_config_uses_null_subset():
    config = load_config(Path("configs/reap_bm25_full.yaml"))
    assert config["data"]["subset_size"] is None
```

**Step 2: Run verification commands**

Run:
- `pytest tests/test_reap_rag.py tests/test_reap_config.py tests/test_architecture_factory.py tests/test_run_experiment_config.py -q`
- `ruff check src/ tests/`

Expected: all passing

**Step 3: Run subset smoke validation**

Run:
- `python scripts/run_experiment.py --config configs/reap.yaml --subset 5`

Then inspect a few saved traces and, if necessary, return to Task 4 once for targeted tuning.

**Step 4: Update docs**

Update docs to reflect reality:
- REAP implemented in `README.md`
- REAP no longer pending in `NEXT_STEPS.md`
- `docs/TECHNICAL_SPEC.md` includes actual config shape and architecture summary

**Step 5: Full audit and finish branch**

Run a final verification sweep before PR:
- `pytest tests/ -q`
- `ruff check src/ tests/`

Then inspect git diff against `dev`, summarize the full scope, push `feature/reap`, and create the PR to `dev`.

**Step 6: Commit**

```bash
git add README.md NEXT_STEPS.md docs/TECHNICAL_SPEC.md tests/test_reap_config.py
git commit -m "docs: add REAP implementation and validation notes"
```
