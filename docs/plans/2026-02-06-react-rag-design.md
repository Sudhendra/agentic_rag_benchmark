# ReAct RAG Design

**Goal**
Implement a full ReAct RAG agent (search + lookup + finish) that runs on HotpotQA and logs results through the existing evaluation pipeline, enabling direct comparison across BM25, Dense, and Hybrid retrieval.

**Scope**
- Implement `ReActRAG` as a `BaseRAG` subclass with an iterative Thought/Action/Observation loop.
- Provide prompt template and deterministic parsing for tool calls.
- Integrate architecture selection into the experiment runner.
- Add ReAct configs for BM25/Dense/Hybrid (subset + full).
- Evaluate ReAct on HotpotQA with the three retrievers and update results documentation.

**Non-Goals**
- Implement other agentic or recursive architectures (Self-RAG, IRCoT, etc.).
- Add new metrics beyond EM/F1 and existing efficiency metrics.
- Replace core retriever or LLM client implementations.

## Architecture

**Class**: `src/architectures/agentic/react_rag.py`

**Loop**
1. Build a ReAct prompt using the question and scratchpad.
2. Call the LLM with `stop=["Observation:"]` to force tool selection.
3. Parse `Thought:` and `Action:` lines.
4. Execute tool:
   - `search[query]`: call retriever, build observation from top-k documents.
   - `lookup[term]`: scan previously retrieved documents for matching sentences.
   - `finish[answer]`: return final answer.
5. Append step to scratchpad and repeat until finish or max iterations.

**Outputs**
- Reasoning chain includes a `ReasoningStep` per loop iteration.
- Retrieved docs includes each `search` call result.
- `num_retrieval_calls` and `num_llm_calls` are tracked for evaluation.

## Prompting and Parsing

**Prompt**: `prompts/react.txt` with strict format

```
Thought: <free text>
Action: search[query] | lookup[term] | finish[answer]
```

**Parsing**
- Regex or line-based parsing to extract `Thought:` and `Action:`.
- If parsing fails, treat the full response as `finish` to avoid infinite loops.

## Tools

**search**
- Calls retriever with `top_k` from config.
- Observation contains a compact context with titles and snippets.

**lookup**
- Searches across previously retrieved documents for a term.
- Returns matching sentences with document titles and sentence indices.

**finish**
- Returns final answer immediately.

## Integration

- Add architecture factory (e.g., `src/architectures/factory.py`) to map config name to class.
- Update `scripts/run_experiment.py` to use factory instead of hard-coded Vanilla.
- Add ReAct configs in `configs/` for subset and full runs, mirroring vanilla naming.
- Update README to reflect ReAct implementation status and results.

## Evaluation Matrix

Run ReAct on HotpotQA with:
- BM25
- Dense
- Hybrid

Use the same LLM model, subset size, and evaluation settings as the vanilla baseline for fair comparison.

## Error Handling

- Parsing failures produce a safe `finish` fallback.
- Max iterations force a final answer step.
- All tool calls are bounded by config values to control costs.

## Testing Strategy

- Unit tests for parsing and tool routing.
- Tests for lookup behavior over retrieved docs.
- Tests for max-iteration fallback.
- Factory selection test for architecture configuration.

## Risks and Mitigations

- **Prompt drift**: keep prompt fixed, versioned in `prompts/`.
- **Unexpected tool outputs**: strict parsing with safe fallbacks.
- **Cost blowups**: cap iterations and top-k, use caching, run subsets first.
