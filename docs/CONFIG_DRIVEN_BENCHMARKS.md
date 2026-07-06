# Config-Driven Benchmarks

Benchmark suites are declared as YAML matrices under `configs/suites/`. A suite expands component YAML fragments into the same resolved config shape accepted by `scripts/run_experiment.py`.

## Quick Start

Preview a suite without API calls:

```bash
python scripts/run_suite.py --suite configs/suites/dev_smoke.yaml --dry-run
```

Run after reviewing the expanded plan:

```bash
python scripts/run_suite.py --suite configs/suites/dev_smoke.yaml --yes --skip-existing
```

Run only matching experiments:

```bash
python scripts/run_suite.py --suite configs/suites/main_full_benchmark.yaml --dry-run --only architecture=ircot_rag
python scripts/run_suite.py --suite configs/suites/main_full_benchmark.yaml --dry-run --only dataset=musique --only retriever=dense
```

Supported filter aliases include `architecture`, `arch`, `dataset`, `retriever`, `retrieval`, `model`, and `seed`. Dotted config paths such as `data.dataset` also work.

## Structure

```text
configs/
  base.yaml
  components/
    architectures/
    datasets/
    retrievers/
    models/
  suites/
    dev_smoke.yaml
    main_full_benchmark.yaml
```

Component files should be small fragments. For example, retriever components set only `retrieval.method`, while architecture components set `architecture.name` and architecture-specific defaults.

## Suite Format

```yaml
suite:
  name: "dev_smoke"
  output_dir: "results/dev_smoke"
  base_config: "../base.yaml"

  defaults:
    experiment:
      seed: 42

  matrix:
    architecture:
      - "../components/architectures/vanilla.yaml"
    dataset:
      - "../components/datasets/hotpotqa_dev.yaml"
    retriever:
      - "../components/retrievers/bm25.yaml"
    model:
      - "../components/models/gpt4o_mini.yaml"

  exclude:
    - dataset: "2wikimultihop"
      retriever: "dense"
      reason: "optional cost control"
```

Merge order is `base_config`, then `defaults`, then one selected component from each matrix dimension. Later values override earlier values.

## Reproducibility

Each run still writes `resolved_config.yaml`, `summary.json`, and `predictions.jsonl`. Suite execution also writes `suite_manifest.json` and `suite_progress.json` in the suite output directory.

Existing single-run configs remain supported:

```bash
python scripts/run_experiment.py --config configs/react_musique_dense_full.yaml
```

## Nice-To-Have Next

- Add cost estimates to suite dry-run output.
- Add suite-level `exclude` rules to defer expensive combinations.
- Migrate `scripts/run_2wiki_experiments.py` and `scripts/run_prompt_sensitivity.py` to suite YAMLs.
- Add suite YAMLs for robustness and prompt sensitivity studies.
- Add `--limit N` for safe partial execution.
