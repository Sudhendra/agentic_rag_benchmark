#!/usr/bin/env python3
"""Run robustness evaluations for the RAG Benchmark.

This script executes smaller-scale sweeps to evaluate the brittleness
of different architectures to prompt variations and extreme top_k distractor scaling.
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_prompt_sensitivity(config_path: Path, prompt_variants: list[Path]):
    """Run evaluation sweep across different prompt wordings."""
    print(f"Running prompt sensitivity analysis for {config_path.name}")
    print("This will execute the runner with different prompt overrides.")
    # Implementation pending Core Runner API stabilization
    pass


def run_top_k_scaling(config_path: Path, k_values: list[int]):
    """Run evaluation sweep across increasing distractor noise."""
    print(f"Running top_k scaling analysis for {config_path.name}")
    for k in k_values:
        print(f" - Queuing run for top_k = {k}")
    # Implementation pending Core Runner API stabilization
    pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Robustness")
    parser.add_argument("--config", type=Path, required=True, help="Base configuration to test")
    parser.add_argument("--prompt-sensitivity", action="store_true", help="Run prompt variances")
    parser.add_argument("--top-k-sweep", action="store_true", help="Run distractor scaling sweep")
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Error: Config not found {args.config}", file=sys.stderr)
        sys.exit(1)
        
    print("="*60)
    print("Agentic RAG Benchmark: Robustness Evaluator")
    print("="*60)
        
    if args.prompt_sensitivity:
        # Mock variants for now
        variants = [Path("prompts/vanilla.txt"), Path("prompts/vanilla_variant_b.txt")]
        run_prompt_sensitivity(args.config, variants)
        
    if args.top_k_sweep:
        run_top_k_scaling(args.config, [3, 5, 10, 20])


if __name__ == "__main__":
    main()
