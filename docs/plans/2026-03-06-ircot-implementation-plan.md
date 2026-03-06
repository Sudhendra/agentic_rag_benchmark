# IRCoT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a faithful IRCoT architecture with sentence-level interleaved retrieval, wire it into the benchmark runner, and validate it on subset-oriented configs.

**Architecture:** Add a new recursive architecture at `src/architectures/recursive/ircot.py` that retrieves once for the question, then alternates between generating one evidence-grounded reasoning sentence and retrieving again using that sentence as the next query. Stop on an explicit answer trigger or step budget exhaustion, then run a final answer synthesis step over the accumulated evidence so the implementation stays close to the ACL 2023 IRCoT setup while remaining cheap enough for API-only subset evaluation.

**Tech Stack:** Python 3.11+, asyncio, existing `BaseRAG` / `RAGResponse` interfaces, existing retriever implementations, pytest + pytest-asyncio, YAML configs, text prompt templates.

---

### Task 1: Add failing IRCoT behavior tests

**Files:**
- Create: `tests/test_ircot_rag.py`
- Test: `tests/test_architecture_factory.py`
- Test: `tests/test_run_experiment_config.py`

**Step 1: Write the failing test**

Add focused tests for:

```python
@pytest.mark.asyncio
async def test_ircot_interleaves_retrieval_and_reasoning(mock_llm, mock_retriever, bridge_question, sample_corpus):
    mock_llm.generate.side_effect = [
        ("France's capital is Paris.", 10, 0.001),
        ("[ANSWER] Anne Hidalgo", 12, 0.001),
        ("Anne Hidalgo", 8, 0.001),
    ]

    rag = IRCoTRAG(mock_llm, mock_retriever, {"max_steps": 3, "top_k": 2})
    response = await rag.answer(bridge_question, sample_corpus)

    assert response.answer == "Anne Hidalgo"
    assert response.num_retrieval_calls == 2
    assert response.num_llm_calls == 3
    assert [step.action for step in response.reasoning_chain] == ["reason", "finish", "synthesize"]
```

```python
def test_factory_creates_ircot(mock_llm, mock_retriever):
    rag = create_architecture("ircot_rag", mock_llm, mock_retriever, {"ircot": {"max_steps": 4}})
    assert rag.get_name() == "ircot_rag"
    assert rag.config["max_steps"] == 4
```

```python
def test_build_rag_uses_ircot_nested_config(monkeypatch):
    config = {
        "architecture": {"name": "ircot_rag"},
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 1024},
        "retrieval": {"method": "bm25", "top_k": 3},
        "ircot": {"max_steps": 4, "max_context_tokens": 3000},
    }
    ...
    assert architecture_config["max_steps"] == 4
    assert architecture_config["top_k"] == 3
```

Also add parser-oriented unit tests for:
- explicit `[ANSWER]` extraction
- fallback answer extraction from `The answer is: ...`
- query cleanup from a factual reasoning sentence
- forced synthesis after `max_steps`
- duplicate passage deduplication across retrieval rounds

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ircot_rag.py tests/test_architecture_factory.py tests/test_run_experiment_config.py -q`

Expected: FAIL with import errors or unknown architecture errors because IRCoT has not been implemented yet.

**Step 3: Write minimal implementation**

Do not implement full polish yet. Add only the minimum code needed for one passing happy-path test:
- create `IRCoTRAG`
- implement `get_name`, `get_type`, `get_config_schema`
- implement one retrieval -> one reasoning sentence -> answer trigger -> synthesis path

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ircot_rag.py::test_ircot_interleaves_retrieval_and_reasoning -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ircot_rag.py tests/test_architecture_factory.py tests/test_run_experiment_config.py src/architectures/recursive/ircot.py
git commit -m "test: add IRCoT architecture coverage"
```

### Task 2: Implement the IRCoT architecture and prompt template

**Files:**
- Create: `src/architectures/recursive/ircot.py`
- Create: `prompts/ircot.txt`
- Test: `tests/test_ircot_rag.py`
- Test: `tests/test_ircot_prompt.py`

**Step 1: Write the failing test**

Add tests that lock in the full IRCoT loop behavior:

```python
@pytest.mark.asyncio
async def test_ircot_forces_final_synthesis_after_max_steps(...):
    mock_llm.generate.side_effect = [
        ("First evidence sentence.", 8, 0.001),
        ("Second evidence sentence.", 8, 0.001),
        ("Fallback final answer", 10, 0.001),
    ]
    rag = IRCoTRAG(mock_llm, mock_retriever, {"max_steps": 2})
    response = await rag.answer(question, corpus)
    assert response.answer == "Fallback final answer"
    assert response.num_llm_calls == 3
```

```python
def test_ircot_prompt_file_mentions_answer_trigger():
    prompt = Path("prompts/ircot.txt").read_text()
    assert "[ANSWER]" in prompt
    assert "one next reasoning sentence" in prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ircot_rag.py tests/test_ircot_prompt.py -q`

Expected: FAIL because the loop, prompt file, and parsing helpers are incomplete.

**Step 3: Write minimal implementation**

Implement the full feature in `src/architectures/recursive/ircot.py`:
- load optional prompt from `prompts/ircot.txt`, otherwise use a default inline prompt
- config schema with at least:
  - `top_k`
  - `max_steps`
  - `max_context_tokens`
  - `answer_trigger`
  - `prompt_path`
  - `final_prompt_path`
  - `max_docs`
- initial retrieval on `question.text`
- iterative sentence generation using cumulative evidence and prior reasoning
- terminal detection via `[ANSWER] <answer>` or `answer is: <answer>`
- retrieval query derived from the newest factual sentence
- cumulative evidence dedupe by document id
- final synthesis prompt when no terminal answer appears inside the loop
- structured `ReasoningStep` trace entries and correct token/cost/call accounting

Keep the implementation faithful to IRCoT by generating one reasoning sentence per turn and retrieving after every non-terminal turn. Do not turn it into free-form ReAct actions or sub-question planning.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ircot_rag.py tests/test_ircot_prompt.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add src/architectures/recursive/ircot.py prompts/ircot.txt tests/test_ircot_rag.py tests/test_ircot_prompt.py
git commit -m "feat: implement IRCoT recursive architecture"
```

### Task 3: Wire IRCoT into factory, runner, and config files

**Files:**
- Modify: `src/architectures/factory.py`
- Modify: `src/architectures/recursive/__init__.py`
- Modify: `scripts/run_experiment.py`
- Create: `configs/ircot.yaml`
- Create: `configs/ircot_dense.yaml`
- Create: `configs/ircot_hybrid.yaml`
- Test: `tests/test_ircot_config.py`
- Test: `tests/test_architecture_factory.py`
- Test: `tests/test_run_experiment_config.py`

**Step 1: Write the failing test**

Add config-level assertions such as:

```python
def test_ircot_yaml_config_loads():
    config = load_config(Path("configs/ircot.yaml"))
    assert config["architecture"]["name"] == "ircot_rag"
    assert config["ircot"]["max_steps"] == 4
```

```python
def test_factory_merges_ircot_nested_config(mock_llm, mock_retriever):
    rag = create_architecture("ircot_rag", mock_llm, mock_retriever, {"top_k": 7, "ircot": {"max_steps": 4}})
    assert rag.config["top_k"] == 7
    assert rag.config["max_steps"] == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ircot_config.py tests/test_architecture_factory.py tests/test_run_experiment_config.py -q`

Expected: FAIL because IRCoT config and wiring do not exist yet.

**Step 3: Write minimal implementation**

Update wiring so that:
- `create_architecture("ircot_rag", ...)` returns `IRCoTRAG`
- nested config key `ircot` is merged like `planner` and `rlm`
- `_build_rag()` adds an `ircot_rag` branch using retrieval `top_k` plus nested `ircot` config
- `configs/ircot.yaml` defaults to BM25 + `subset_size: 100`
- `configs/ircot_dense.yaml` and `configs/ircot_hybrid.yaml` mirror the existing architecture config naming convention

Suggested starter config:

```yaml
inherits: base.yaml

experiment:
  name: "ircot"

architecture:
  name: "ircot_rag"

ircot:
  max_steps: 4
  max_context_tokens: 3000
  max_docs: 8

retrieval:
  method: "bm25"
  top_k: 3

data:
  subset_size: 100
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ircot_config.py tests/test_architecture_factory.py tests/test_run_experiment_config.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add src/architectures/factory.py src/architectures/recursive/__init__.py scripts/run_experiment.py configs/ircot.yaml configs/ircot_dense.yaml configs/ircot_hybrid.yaml tests/test_ircot_config.py tests/test_architecture_factory.py tests/test_run_experiment_config.py
git commit -m "feat: wire IRCoT into configs and runner"
```

### Task 4: Update docs and subset-run workflow checks

**Files:**
- Modify: `README.md`
- Modify: `docs/TECHNICAL_SPEC.md`
- Modify: `NEXT_STEPS.md`
- Test: `tests/test_ircot_config.py`

**Step 1: Write the failing test**

If a docs-oriented assertion is useful, keep it lightweight:

```python
def test_ircot_subset_config_uses_small_dev_subset():
    config = load_config(Path("configs/ircot.yaml"))
    assert config["data"]["subset_size"] == 100
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ircot_config.py::test_ircot_subset_config_uses_small_dev_subset -q`

Expected: FAIL until the config exists.

**Step 3: Write minimal implementation**

Update docs so the repository state matches reality:
- mark IRCoT as implemented or in-progress, depending on actual code completion
- document the new config names and prompt file in `docs/TECHNICAL_SPEC.md`
- note that initial validation is subset-only, not full evaluation

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ircot_config.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add README.md docs/TECHNICAL_SPEC.md NEXT_STEPS.md tests/test_ircot_config.py
git commit -m "docs: add IRCoT benchmark workflow notes"
```

### Task 5: Validate the subset-oriented IRCoT path

**Files:**
- Verify: `tests/test_ircot_rag.py`
- Verify: `tests/test_ircot_config.py`
- Verify: `tests/test_architecture_factory.py`
- Verify: `tests/test_run_experiment_config.py`
- Verify: `configs/ircot.yaml`

**Step 1: Write the failing test**

Do not add more production code here. This task is pure verification.

**Step 2: Run test to verify current state**

Run:
- `pytest tests/test_ircot_rag.py tests/test_ircot_config.py tests/test_architecture_factory.py tests/test_run_experiment_config.py -q`
- `ruff check src/ tests/`

Expected: all passing

**Step 3: Run a cheap subset smoke check**

Run:
- `python scripts/run_experiment.py --config configs/ircot.yaml --subset 5`

Expected:
- architecture builds successfully
- retriever indexes corpus
- evaluator starts and completes a small run
- results are written without requiring a full evaluation sweep

If API credentials are unavailable, stop after `_build_rag()`/config-level verification and report the blocker explicitly.

**Step 4: Record what happened**

Note:
- which tests passed
- whether the subset smoke run completed
- any blockers such as missing API keys or rate limits

**Step 5: Commit**

```bash
git add .
git commit -m "test: validate IRCoT subset workflow"
```
