# Vanilla RAG + MLflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the vanilla RAG baseline with aggregated EM/F1 + efficiency metrics and log results to MLflow (local file store), using Poetry for dependency management.

**Architecture:** Add a config loader with inheritance, an async evaluator that aggregates metrics, a results writer, and a single experiment entrypoint that logs artifacts and metrics to MLflow.

**Tech Stack:** Python 3.11, venv + pip, MLflow (local file store), asyncio, PyYAML, pytest-asyncio

---

### Task 1: Add MLflow dependency and set up venv

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_mlflow_import.py`
- Modify: `README.md`
- Modify: `.gitignore`

**Step 1: Write the failing test**

```python
def test_mlflow_import_smoke():
    import mlflow

    assert mlflow.__version__
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mlflow_import.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlflow'`

**Step 3: Update dependencies and venv guidance**

- Add MLflow to `pyproject.toml` dependencies:

```toml
dependencies = [
    "openai>=1.0",
    "rank-bm25>=0.2.2",
    "datasets>=2.14",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "tqdm>=4.65",
    "tenacity>=8.2",
    "tiktoken>=0.5",
    "python-dotenv>=1.0",
    "mlflow>=2.10",
]
```

- Update `README.md` quick start to use `python -m venv .venv`, activation, and `pip install -e ".[dev]"`.
- Add `.venv/` and `mlruns/` to `.gitignore` so venv and MLflow artifacts stay untracked.

**Step 4: Install deps and re-run tests**

Run: `python -m venv .venv`
Expected: `.venv/` created

Run: `source .venv/bin/activate`
Expected: shell uses venv Python

Run: `pip install -e ".[dev]"`
Expected: dependencies installed into `.venv/`

Run: `pytest tests/test_mlflow_import.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml README.md .gitignore tests/test_mlflow_import.py
git commit -m "chore: add venv setup and mlflow dependency"
```

---

### Task 2: Implement config loader with inheritance

**Files:**
- Create: `src/utils/config.py`
- Modify: `src/utils/__init__.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
def test_config_inherits_base(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("""
llm:
  model: gpt-4o-mini
retrieval:
  top_k: 5
""")

    child = tmp_path / "child.yaml"
    child.write_text("""
inherits: base.yaml
retrieval:
  top_k: 3
""")

    config = load_config(child)
    assert config["llm"]["model"] == "gpt-4o-mini"
    assert config["retrieval"]["top_k"] == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `NameError: load_config is not defined`

**Step 3: Write minimal implementation**

```python
def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    if "inherits" in data:
        base_path = path.parent / data.pop("inherits")
        base = load_config(base_path)
        return deep_merge(base, data)
    return data
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/utils/config.py src/utils/__init__.py tests/test_config.py
git commit -m "feat: add config loader with inheritance"
```

---

### Task 3: Build evaluator and metric aggregation

**Files:**
- Create: `src/evaluation/evaluator.py`
- Modify: `src/evaluation/__init__.py`
- Create: `tests/test_evaluator.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_evaluator_aggregates_metrics(mock_rag, questions, corpus):
    evaluator = Evaluator(mock_rag, max_concurrency=2)
    result = await evaluator.evaluate(questions, corpus)

    assert result.num_questions == len(questions)
    assert 0.0 <= result.avg_exact_match <= 1.0
    assert 0.0 <= result.avg_f1 <= 1.0
    assert result.total_cost_usd >= 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluator.py -v`
Expected: FAIL with `NameError: Evaluator is not defined`

**Step 3: Write minimal implementation**

```python
class Evaluator:
    async def evaluate(self, questions, corpus) -> BenchmarkResult:
        # async semaphore + gather
        # compute EvaluationResult per question
        # aggregate to BenchmarkResult
        return BenchmarkResult(...)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/evaluation/evaluator.py src/evaluation/__init__.py tests/test_evaluator.py
git commit -m "feat: add evaluator and aggregation"
```

---

### Task 4: Add results writer and MLflow logging helpers

**Files:**
- Create: `src/utils/results.py`
- Create: `tests/test_results.py`

**Step 1: Write the failing test**

```python
def test_save_results_writes_files(tmp_path, benchmark_result):
    output_dir = tmp_path / "run"
    save_results(benchmark_result, output_dir, resolved_config={"x": 1})

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "resolved_config.yaml").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_results.py -v`
Expected: FAIL with `NameError: save_results is not defined`

**Step 3: Write minimal implementation**

```python
def save_results(result, output_dir: Path, resolved_config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # write summary.json, predictions.jsonl, resolved_config.yaml
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_results.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/utils/results.py tests/test_results.py
git commit -m "feat: add results writer"
```

---

### Task 5: Create experiment runner and vanilla config

**Files:**
- Create: `scripts/run_experiment.py`
- Modify: `configs/base.yaml`
- Create: `configs/vanilla.yaml`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
def test_run_experiment_loads_config(tmp_path):
    # write minimal config to tmp_path
    config = load_config(tmp_path / "config.yaml")
    assert "llm" in config
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_experiment_config.py -v`
Expected: FAIL with missing module or config loader import

**Step 3: Write minimal implementation**

- Implement `scripts/run_experiment.py` to:
  - load resolved config
  - init cache/logging/LLM/retriever/architecture
  - load data and index corpus
  - run evaluator and save results
  - log metrics and artifacts to MLflow

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_experiment_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/run_experiment.py configs/base.yaml configs/vanilla.yaml README.md tests/test_run_experiment_config.py
git commit -m "feat: add vanilla experiment runner"
```

---

### Task 6: Manual verification (baseline run)

**Step 1: Run a small experiment**

Run: `python scripts/run_experiment.py --config configs/vanilla.yaml`

Expected:
- MLflow run created in `mlruns/`
- Results written to `results/<run_id>/`
- Console summary with EM/F1 and cost

**Step 2: Inspect MLflow UI**

Run: `mlflow ui`
Expected: Local UI shows run parameters, metrics, and artifacts

**Step 3: Verify outputs are ignored**

Run: `git status -sb`
Expected: results and mlruns are not staged or tracked

---

## Notes
- Keep cache enabled to reduce API costs.
- Use `subset_size` in config for rapid testing.
- Avoid calling real APIs in tests; use mocks for LLM and retriever.
