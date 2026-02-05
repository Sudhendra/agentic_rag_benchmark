# GitHub Actions CI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GitHub Actions CI that runs linting and tests on every PR.

**Architecture:** Single workflow with one job running Ruff, Black (check), and pytest on Python 3.11 using pip caching.

**Tech Stack:** GitHub Actions, Python 3.11, pip, Ruff, Black, pytest

---

### Task 1: Add GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Write the failing test**

```python
def test_ci_workflow_exists():
    assert Path(".github/workflows/ci.yml").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ci_workflow.py::test_ci_workflow_exists -v`
Expected: FAIL with `AssertionError: False is not true`

**Step 3: Write minimal implementation**

```yaml
name: CI

on:
  pull_request:
  push:
    branches:
      - dev
      - main

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Lint
        run: ruff check src tests
      - name: Format check
        run: black --check src tests
      - name: Tests
        run: pytest tests/ -v
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ci_workflow.py::test_ci_workflow_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .github/workflows/ci.yml tests/test_ci_workflow.py docs/plans/2026-02-05-ci-github-actions.md docs/plans/2026-02-05-ci-github-actions-design.md
git commit -m "feat: add github actions ci workflow"
```
