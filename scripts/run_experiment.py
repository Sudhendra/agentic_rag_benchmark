from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.architectures.factory import create_architecture
from src.core.llm_client import create_llm_client
from src.data.hotpotqa import load_hotpotqa
from src.evaluation.evaluator import Evaluator
from src.retrieval.hybrid import create_retriever
from src.utils.cache import SQLiteCache
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.results import save_results


def load_resolved_config(path: Path) -> dict[str, Any]:
    return load_config(path)


def _apply_overrides(config: dict[str, Any], subset_size: int | None) -> dict[str, Any]:
    if subset_size is not None:
        config.setdefault("data", {})["subset_size"] = subset_size
    return config


def _build_rag(config: dict[str, Any]):
    cache = None
    if config.get("cache", {}).get("enabled", False):
        cache_path = config.get("cache", {}).get("path", ".cache/llm_cache.db")
        cache = SQLiteCache(cache_path)

    llm_config = config.get("llm", {})
    llm = create_llm_client(
        provider=llm_config.get("provider", "openai"),
        model=llm_config.get("model"),
        cache=cache,
    )

    retrieval_config = config.get("retrieval", {})
    retriever = create_retriever(
        method=retrieval_config.get("method", "bm25"),
        bm25_weight=retrieval_config.get("bm25_weight", 0.5),
        dense_weight=retrieval_config.get("dense_weight", 0.5),
        embedding_model=retrieval_config.get("embedding_model", "text-embedding-3-small"),
    )

    architecture_name = config.get("architecture", {}).get("name", "vanilla_rag")
    common_config = {
        "top_k": retrieval_config.get("top_k", 5),
        "max_context_tokens": llm_config.get("max_tokens", 1024),
    }

    if architecture_name == "vanilla_rag":
        architecture_config = {
            **common_config,
            "prompt_path": config.get("prompt_path", "prompts/vanilla.txt"),
            **config.get("vanilla", {}),
        }
    elif architecture_name == "react_rag":
        architecture_config = {
            **common_config,
            **config.get("react", {}),
        }
    elif architecture_name == "self_rag":
        architecture_config = {
            **common_config,
            **config.get("self_rag", {}),
        }
    else:
        architecture_config = {**common_config, **config.get(architecture_name, {})}

    return create_architecture(architecture_name, llm, retriever, architecture_config)


async def run_experiment(config: dict[str, Any]) -> Path:
    logging_config = config.get("logging", {})
    setup_logging(
        level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("file"),
    )

    rag = _build_rag(config)

    data_config = config.get("data", {})
    dataset_name = data_config.get("dataset", "hotpotqa")
    if dataset_name != "hotpotqa":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    questions, corpus = load_hotpotqa(
        setting=data_config.get("setting", "distractor"),
        split=data_config.get("split", "validation"),
        subset_size=data_config.get("subset_size"),
    )

    await rag.retriever.index(corpus)

    evaluator = Evaluator(
        rag,
        max_concurrency=config.get("evaluation", {}).get("max_concurrency", 5),
        dataset_name=dataset_name,
    )
    benchmark_result = await evaluator.evaluate(questions, corpus)

    output_root = Path(config.get("experiment", {}).get("output_dir", "results"))

    run_id = None
    try:
        import mlflow

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    "dataset": dataset_name,
                    "subset_size": data_config.get("subset_size"),
                    "model": benchmark_result.model,
                    "retrieval_method": config.get("retrieval", {}).get("method"),
                    "top_k": config.get("retrieval", {}).get("top_k"),
                }
            )
            mlflow.log_metrics(
                {
                    "avg_exact_match": benchmark_result.avg_exact_match,
                    "avg_f1": benchmark_result.avg_f1,
                    "avg_latency_ms": benchmark_result.avg_latency_ms,
                    "avg_tokens_per_question": benchmark_result.avg_tokens_per_question,
                    "avg_retrieval_calls": benchmark_result.avg_retrieval_calls,
                    "avg_llm_calls": benchmark_result.avg_llm_calls,
                    "total_cost_usd": benchmark_result.total_cost_usd,
                }
            )
    except Exception:
        run_id = run_id or "manual"

    run_dir = output_root / (run_id or "manual")
    save_results(benchmark_result, run_dir, resolved_config=config)

    try:
        import mlflow

        mlflow.log_artifacts(str(run_dir), artifact_path="results")
    except Exception:
        pass

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RAG experiment and log to MLflow")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--subset", type=int, default=None, help="Override subset size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_resolved_config(Path(args.config))
    config = _apply_overrides(config, args.subset)
    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
