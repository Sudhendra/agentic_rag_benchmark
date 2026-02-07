# Vanilla RAG + MLflow Baseline Design

**Goal**
Deliver a reproducible vanilla RAG baseline that evaluates HotpotQA with EM/F1 and efficiency metrics, logs to MLflow (local file store by default), and produces stable run artifacts for later analysis.

**Scope**
- Implement an async evaluation harness that can run vanilla RAG over a dataset subset.
- Aggregate per-question metrics into dataset-level summaries.
- Log parameters, metrics, and artifacts to MLflow.
- Standardize experiment configuration and output layout.
- Adopt Poetry for dependency management and local venvs.

**Non-Goals**
- Implement other architectures (ReAct, IRCoT, etc.).
- Build advanced metrics (hallucination, chain fidelity) beyond EM/F1.
- Add remote tracking server configuration beyond optional environment variables.

## Architecture

**Entry Point**
`scripts/run_experiment.py` loads config, initializes components, runs evaluation, writes results, and logs to MLflow.

**Core Components**
- **Config loader** (`src/utils/config.py`): loads YAML config, resolves `inherits`, and returns a merged dictionary.
- **Evaluator** (`src/evaluation/evaluator.py`): async runner that calls `rag.answer` concurrently, computes per-question metrics, and aggregates results.
- **Results writer** (`src/utils/results.py`): writes `summary.json`, `predictions.jsonl`, and `resolved_config.yaml` to a run directory.
- **MLflow logger** (`src/utils/mlflow_utils.py` or in runner): logs params/metrics and stores artifacts.

## Data Flow
1. Load YAML config (base + override) and seed RNG.
2. Create cache, LLM client, retriever, and `VanillaRAG` instance.
3. Load HotpotQA questions and corpus. For distractor setting, derive per-question corpus by question ID prefix.
4. Index corpus (retriever-specific).
5. Evaluate questions with concurrency control; for each question:
   - Run `rag.answer` with question + question-specific corpus.
   - Compute EM/F1 (and optional supporting facts if enabled).
   - Capture efficiency metrics from `RAGResponse`.
6. Aggregate metrics (mean EM/F1, avg latency/tokens, total cost) and breakdown by question type.
7. Write artifacts to `results/<run_id>/` and log to MLflow.

## MLflow Logging
**Default Tracking**: local file store (`mlruns/`).
**Params**: dataset name, subset size, model, retriever method, top_k, max_tokens, prompt_path.
**Metrics**: avg EM, avg F1, avg latency, avg tokens, total cost, retrieval/LLM calls.
**Artifacts**: predictions.jsonl, summary.json, resolved_config.yaml.

## Error Handling
- Per-question failures are caught and recorded in predictions with error metadata; evaluation continues.
- MLflow logging is wrapped to avoid run failure when MLflow is unavailable.
- Cache and logging setup failures are surfaced early with actionable errors.

## Testing Strategy
- Unit tests for config resolution and evaluator aggregation with mocked RAG responses.
- Smoke test for MLflow import and logging path selection.
- Avoid network calls by mocking LLM/retriever and using tiny synthetic datasets.

## Risks and Mitigations
- **Poetry not installed**: provide installation instructions and fallback to pip if needed.
- **MLflow absent**: treat as optional dependency for logging; allow disable flag.
- **HotpotQA corpus size**: support subset size in config and caching for development cost control.
