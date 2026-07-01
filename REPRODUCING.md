# COMPASS: Reproducing Results

This document provides exact instructions for reproducing all results in the COMPASS paper.

## Prerequisites

```bash
# Python 3.11+
pip install -e ".[dev]"

# Set API keys
export OPENAI_API_KEY=sk-...
# Optional (for cross-model validation):
export GROQ_API_KEY=...
```

## 1. Main Experiments (HotpotQA + MuSiQue)

### Full benchmark runs (7 architectures x 3 retrievers x 2 datasets = 42 runs)

```bash
# HotpotQA - all architectures, all retrievers
python scripts/run_experiment.py --config configs/vanilla_dense_full.yaml
python scripts/run_experiment.py --config configs/vanilla_hybrid_full.yaml
python scripts/run_experiment.py --config configs/vanilla_bm25_full.yaml

python scripts/run_experiment.py --config configs/react_dense_full.yaml
python scripts/run_experiment.py --config configs/react_hybrid_full.yaml
python scripts/run_experiment.py --config configs/react_bm25_full.yaml

# ... repeat for selfrag, planner, ircot, reap, rlm
# Each config is in configs/<arch>_<retriever>_full.yaml
```

### Quick reproduction (subset for development)

```bash
# Use --subset flag for quick runs on 100 questions
python scripts/run_experiment.py --config configs/vanilla_dense_full.yaml --subset 100
```

## 2. Analysis Scripts (no API calls needed)

All analysis scripts operate on existing results and require no additional API calls.

### Generate all paper tables from raw data
```bash
python scripts/generate_paper_tables.py
# Output: paper_tables/ directory with LaTeX + CSV files
```

### Statistical significance testing
```bash
python scripts/significance_testing.py
# Output: results/significance/ (McNemar matrix + heatmap figures)
```

### Equal-compute normalization
```bash
python scripts/equal_compute_analysis.py
# Output: results/equal_compute/ (token-budget Pareto figures)
```

### Prompt Sensitivity Coefficient (PSC)
```bash
python scripts/compute_psc.py
# Output: results/psc/ (PSC values + bar charts)
```

### Theory validation plots
```bash
python scripts/plot_theory_validation.py
# Output: results/figures/fig_theory*.png
```

### Deeper analysis (error correlation, complementarity, per-retriever)
```bash
python scripts/deeper_analysis.py
# Output: results/deeper_analysis/ + figures
```

### Failure case studies
```bash
python scripts/failure_case_studies.py
# Output: results/failure_cases/ (CSV files for qualitative review)
```

## 3. Additional Experiments

### Prompt sensitivity (11 variants, ~$0.28)
```bash
python scripts/run_prompt_sensitivity.py
```

### Robustness ablations (~$0.36)
```bash
python scripts/robustness_evaluator.py
```

### REAP gpt-4o probe (~$3.50)
```bash
python scripts/investigate_reap.py
```

### 2WikiMultiHopQA experiments
```bash
python scripts/run_2wiki_experiments.py
```

## 4. Cached Responses

All LLM responses are cached in a SQLite database (`.cache/llm_cache.db`).
This means:
- Re-running experiments with the same config returns cached results instantly
- The cache can be shared for exact reproduction
- To force fresh API calls, delete the cache file

```bash
# Export cache for sharing
cp .cache/llm_cache.db compass_cache.db

# Import cache for reproduction
cp compass_cache.db .cache/llm_cache.db
```

## 5. Config Files

All experiment configurations are in `configs/`:
- `base.yaml` - shared defaults
- `<arch>_<retriever>_full.yaml` - main experiment configs
- `sensitivity/` - prompt sensitivity variant configs
- `robustness_*.yaml` - ablation configs
- `reap_gpt4o_200q.yaml` - REAP model-strength probe

## 6. Prompts

All prompts are in `prompts/`:
- `prompts/sensitivity/` - 11 prompt variants for sensitivity analysis
- Architecture-specific prompts are embedded in the source code under `src/architectures/`

## 7. Expected Costs

| Experiment | Cost | Time |
|-----------|------|------|
| Full HotpotQA (7 archs x 3 retrievers) | ~$50 | ~6 hours |
| Full MuSiQue (7 archs x 3 retrievers) | ~$15 | ~3 hours |
| Prompt sensitivity (11 variants) | ~$0.28 | ~30 min |
| Robustness ablations | ~$0.36 | ~15 min |
| REAP gpt-4o probe (200q) | ~$3.50 | ~45 min |
| Analysis scripts (no API) | $0 | ~10 min |
| **Total** | **~$70** | **~10 hours** |

## 8. Results Structure

```
results/
├── <run_id>/                    # One directory per experiment run
│   ├── summary.json             # Aggregate metrics
│   ├── predictions.jsonl        # Per-question predictions
│   └── config.yaml              # Run configuration
├── significance/                # McNemar test results
├── equal_compute/               # Token-budget analysis
├── psc/                         # Prompt Sensitivity Coefficient
├── theory/                      # Theory validation results
├── deeper_analysis/             # Error correlation, complementarity
├── failure_cases/               # Qualitative failure case CSVs
├── figures/                     # All publication figures
└── robustness/                  # Ablation sweep results
```

## 9. Run IDs (for verification)

Key runs referenced in the paper:

| Architecture | Dataset | Run ID |
|-------------|---------|--------|
| Vanilla RAG | HotpotQA (Dense) | bfc8f29304c5 |
| ReAct RAG | HotpotQA (Hybrid) | 25cc3f6b90df |
| Self-RAG | HotpotQA (Hybrid) | 7272b4eb8c81 |
| Planner RAG | HotpotQA (Dense) | b4284f7fb029 |
| IRCoT | HotpotQA (Hybrid) | 4d923d09821d |
| REAP | HotpotQA (Dense) | c4da1615351b |
| Recursive LM | HotpotQA (Hybrid) | 09581743885c |
| REAP (gpt-4o probe) | HotpotQA (200q) | 7b82fbbef37e |

## 10. Reproducibility Notes

- All experiments use temperature=0 (deterministic)
- Random seeds are fixed in configs (seed=42 default)
- The SQLite cache ensures identical results on re-run
- Cross-model validation uses Groq API (Llama-3.3-70B)
- BM25 retriever uses rank_bm25 library
- Dense retriever uses OpenAI text-embedding-3-small
