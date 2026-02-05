# GitHub Actions CI Design

**Goal**
Add GitHub Actions CI to run linting and tests for every PR, with a simple, fast workflow that mirrors local dev tooling.

**Scope**
- Add a single workflow that runs on pull requests and on pushes to `dev` and `main`.
- Run Ruff and Black in check mode.
- Run pytest.
- Use pip caching for faster runs.

**Non-Goals**
- Multi-version Python matrix (future enhancement).
- Coverage reporting or badges.
- Running dataset downloads or API-dependent scripts.

## Workflow Structure
Create `.github/workflows/ci.yml` with one job (`lint-and-test`). The job uses `actions/checkout@v4` and `actions/setup-python@v5` with Python 3.11 and pip caching. It installs dev dependencies via `pip install -e ".[dev]"` and runs:

1. `ruff check src tests`
2. `black --check src tests`
3. `pytest tests/ -v`

## Triggers
- `pull_request` for all PRs
- `push` to `dev` and `main`

## Guardrails
- No secrets required
- No scripts that call paid APIs
- Keep runtime deterministic and fast

## Success Criteria
- CI runs for every PR
- Linting and tests must pass for merge
- Average runtime stays low (single job)
