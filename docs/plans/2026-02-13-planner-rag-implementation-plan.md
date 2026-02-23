# Planner RAG Implementation Plan

**Goal:** Implement TreePS-style planner inference with tracing and integrate it into the existing benchmark framework.

## Tasks

### Task 1: Prompts
- Add `prompts/planner_action.txt`.
- Add `prompts/planner_solve.txt`.
- Add `prompts/planner_synthesize.txt`.
- Add tests in `tests/test_planner_prompt.py`.

### Task 2: Planner Architecture
- Create `src/architectures/agentic/planner_rag.py`.
- Implement gate, iterative planner loop, node solve, rollout, backtrack, stop, synthesis.
- Emit structured `ReasoningStep` traces.
- Add tests in `tests/test_planner_rag.py`.

### Task 3: Wiring
- Export in `src/architectures/agentic/__init__.py`.
- Register in `src/architectures/factory.py`.
- Add planner branch in `scripts/run_experiment.py`.

### Task 4: Config Variants
- Add `configs/planner.yaml`.
- Add `configs/planner_dense.yaml` and `configs/planner_hybrid.yaml`.
- Add `configs/planner_bm25_full.yaml`, `configs/planner_dense_full.yaml`, `configs/planner_hybrid_full.yaml`.
- Add integration checks in `tests/test_planner_config.py`.

### Task 5: Spec and README Updates
- Update Planner section and architecture config examples in `docs/TECHNICAL_SPEC.md`.
- Add TreePS-RAG in key papers table in `docs/TECHNICAL_SPEC.md`.
- Update implementation checklist statuses in `docs/TECHNICAL_SPEC.md`.
- Mark Planner as implemented (results pending) in `README.md`.

### Task 6: Validation
- Run planner-focused tests (`test_planner_*`) and regression checks (`test_react_rag.py`, `test_self_rag.py`, `test_architecture_factory.py`).
- Note environment dependency blockers if full suite cannot run.
