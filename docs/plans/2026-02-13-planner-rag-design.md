# Planner RAG Design

**Goal**
Implement a planner-driven agentic RAG architecture that uses explicit tree actions (TRAVERSE, SELECT, ROLLOUT, BACKTRACK, STOP) for multi-hop reasoning, while preserving compatibility with the existing benchmark pipeline.

**Scope**
- Implement `PlannerRAG` as a `BaseRAG` subclass.
- Add prompt templates for planner action selection, node solving, and synthesis.
- Integrate into architecture factory and experiment runner.
- Add Planner configs for BM25/Dense/Hybrid (subset and full).
- Add focused unit tests for planner behavior and parser robustness.

**Non-Goals**
- Offline trajectory synthesis and model fine-tuning.
- New persisted trace artifacts beyond `reasoning_chain`.
- Dataset rollout beyond HotpotQA in this phase.

## Architecture

**Class**: `src/architectures/agentic/planner_rag.py`

**Flow**
1. Gate: decide direct answer vs recursive planning.
2. Planning loop (bounded by `max_iterations`) over a mutable plan tree.
3. Execute actions:
   - `SELECT`/`TRAVERSE`: retrieve + solve node + confidence check.
   - `ROLLOUT`: expand node into sub-questions (bounded branch factor/depth).
   - `BACKTRACK`: prune node and move active focus to parent.
   - `STOP`: end planning.
4. Synthesize final answer from solved/active nodes.

**Trace Contract**
- Each planner operation emits one `ReasoningStep`.
- Action names are preserved in `ReasoningStep.action`.
- Observations capture compact status transitions and confidence.

## Config Defaults
- `top_k`: 3
- `max_iterations`: 5
- `max_branching_factor`: 2
- `max_depth`: 3
- `min_stop_confidence`: 0.8
- `allow_direct_answer`: true

## Risks and Mitigations
- Prompt format drift: strict JSON parsing + repair + safe fallback.
- Cost growth: conservative bounded defaults and early-stop by confidence.
- Invalid planner actions: fallback to `SELECT root` or `STOP` when near budget cap.
