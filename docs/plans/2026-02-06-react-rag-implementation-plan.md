# ReAct RAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement full ReAct RAG with search+lookup+finish, integrate it into the experiment runner, and evaluate HotpotQA across BM25, Dense, and Hybrid retrievers.

**Architecture:** Add a `ReActRAG` architecture with an iterative Thought/Action/Observation loop and strict tool parsing, expose it through a factory used by `run_experiment.py`, and mirror vanilla configs to run ReAct with all retrievers. Evaluation remains unchanged via `Evaluator`.

**Tech Stack:** Python 3.11, async/await, OpenAI/Anthropic clients, existing retrievers, pytest.

---

### Task 1: Add ReAct prompt template

**Files:**
- Create: `prompts/react.txt`

**Step 1: Write the failing test**

```python
def test_react_prompt_file_exists() -> None:
    assert (Path("prompts/react.txt")).exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_react_prompt.py::test_react_prompt_file_exists -v`
Expected: FAIL with file missing

**Step 3: Write minimal implementation**

Create `prompts/react.txt` with:

```
Answer the following question by reasoning step-by-step and using the tools.

Available tools:
- search[query]: search for passages related to the query
- lookup[term]: lookup a term in previously retrieved passages
- finish[answer]: provide the final answer

Question: {question}

{scratchpad}

Thought:
Action:
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_react_prompt.py::test_react_prompt_file_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add prompts/react.txt tests/test_react_prompt.py
git commit -m "docs: add react prompt template"
```

### Task 2: Implement ReActRAG architecture

**Files:**
- Create: `src/architectures/agentic/react_rag.py`
- Modify: `src/architectures/agentic/__init__.py`
- Test: `tests/test_react_rag.py`

**Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_react_rag_search_and_finish(mock_llm, mock_retriever, sample_question, sample_corpus):
    rag = ReActRAG(mock_llm, mock_retriever, {"top_k": 3, "max_iterations": 2})
    response = await rag.answer(sample_question, sample_corpus)
    assert response.answer == "Final answer"
    assert response.num_retrieval_calls == 1
    assert response.num_llm_calls == 2

@pytest.mark.asyncio
async def test_react_rag_lookup_uses_retrieved_docs(mock_llm, mock_retriever, sample_question, sample_corpus):
    rag = ReActRAG(mock_llm, mock_retriever, {"top_k": 3, "max_iterations": 2})
    response = await rag.answer(sample_question, sample_corpus)
    assert "lookup" in response.reasoning_chain[1].action

@pytest.mark.asyncio
async def test_react_rag_max_iterations_fallback(mock_llm, mock_retriever, sample_question, sample_corpus):
    rag = ReActRAG(mock_llm, mock_retriever, {"top_k": 3, "max_iterations": 1})
    response = await rag.answer(sample_question, sample_corpus)
    assert response.answer
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_react_rag.py -v`
Expected: FAIL with `ReActRAG` missing

**Step 3: Write minimal implementation**

Implement:

```python
class ReActRAG(BaseRAG):
    def get_name(self) -> str:
        return "react_rag"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.AGENTIC

    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        return {
            "top_k": (int, False, 5),
            "max_iterations": (int, False, 7),
            "max_context_tokens": (int, False, 4000),
            "prompt_path": (str, False, "prompts/react.txt"),
        }

    async def answer(...):
        # iterative loop: build prompt, llm.generate(stop=["Observation:"]), parse
        # execute search/lookup/finish, append ReasoningStep, aggregate stats
```

Ensure:
- `search` triggers retriever and appends `RetrievalResult` to `retrieved_docs`.
- `lookup` scans previously retrieved docs for term.
- `finish` ends loop and returns.
- On parse failure, treat response as `finish`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_react_rag.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/architectures/agentic/react_rag.py src/architectures/agentic/__init__.py tests/test_react_rag.py
git commit -m "feat: add react rag architecture"
```

### Task 3: Add architecture factory and wire runner

**Files:**
- Create: `src/architectures/factory.py`
- Modify: `scripts/run_experiment.py`
- Test: `tests/test_architecture_factory.py`

**Step 1: Write the failing test**

```python
def test_create_architecture_react(mock_llm, mock_retriever):
    rag = create_architecture("react_rag", mock_llm, mock_retriever, {"react": {}})
    assert rag.get_name() == "react_rag"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_architecture_factory.py -v`
Expected: FAIL with missing factory

**Step 3: Write minimal implementation**

Implement `create_architecture(name, llm, retriever, config)` that maps:
- `vanilla_rag` -> `VanillaRAG`
- `react_rag` -> `ReActRAG`

Update `scripts/run_experiment.py` to read `config["architecture"]["name"]` and pass the appropriate sub-config (`config.get("react", {})`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_architecture_factory.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/architectures/factory.py scripts/run_experiment.py tests/test_architecture_factory.py
git commit -m "feat: add architecture factory and runner wiring"
```

### Task 4: Add ReAct configs for BM25/Dense/Hybrid

**Files:**
- Create: `configs/react.yaml`
- Create: `configs/react_dense.yaml`
- Create: `configs/react_hybrid.yaml`
- Create: `configs/react_bm25_full.yaml`
- Create: `configs/react_dense_full.yaml`
- Create: `configs/react_hybrid_full.yaml`

**Step 1: Write the failing test**

```python
def test_react_config_loads(tmp_path: Path):
    config = load_config(Path("configs/react.yaml"))
    assert config["architecture"]["name"] == "react_rag"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_react_config.py::test_react_config_loads -v`
Expected: FAIL with missing file

**Step 3: Write minimal implementation**

Add config files mirroring vanilla ones with `architecture.name: react_rag` and `react.max_iterations`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_react_config.py::test_react_config_loads -v`
Expected: PASS

**Step 5: Commit**

```bash
git add configs/react*.yaml tests/test_react_config.py
git commit -m "feat: add react config variants"
```

### Task 5: Update README and run evaluation matrix

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-02-06-react-rag-implementation-plan.md`

**Step 1: Update README results table**

Add ReAct rows for BM25/Dense/Hybrid (to be filled after running experiments).

**Step 2: Run experiments**

Run subset first:
`python scripts/run_experiment.py --config configs/react_dense.yaml --subset 100`

Then full runs (as budget allows):
- `python scripts/run_experiment.py --config configs/react_bm25_full.yaml`
- `python scripts/run_experiment.py --config configs/react_dense_full.yaml`
- `python scripts/run_experiment.py --config configs/react_hybrid_full.yaml`

**Step 3: Update README with final metrics**

Capture EM/F1 and cost from result summaries and add to README.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add react rag results"
```

### Task 6: Regression test sweep

**Files:**
- Test: `tests/`

**Step 1: Run full test suite**

Run: `pytest -q`
Expected: PASS

**Step 2: Commit (if any fixes required)**

```bash
git add -A
git commit -m "test: stabilize react rag changes"
```
