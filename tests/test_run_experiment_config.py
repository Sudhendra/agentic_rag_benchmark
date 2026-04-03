import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


from src.core.types import ArchitectureType, BenchmarkResult, Document, Question, QuestionType


def test_run_experiment_loads_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
llm:
  model: gpt-4o-mini
""")

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_experiment = importlib.import_module("run_experiment")
        load_resolved_config = run_experiment.load_resolved_config

        config = load_resolved_config(config_path)
    finally:
        sys.path.remove(str(scripts_dir))

    assert "llm" in config


def test_planner_uses_planner_default_context_tokens_when_not_set(monkeypatch) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_experiment = importlib.import_module("run_experiment")
    finally:
        sys.path.remove(str(scripts_dir))

    captured: dict[str, object] = {}

    def fake_create_llm_client(**kwargs):
        return SimpleNamespace(model="test-model")

    def fake_create_retriever(**kwargs):
        return object()

    def fake_create_architecture(name, llm, retriever, config):
        captured["name"] = name
        captured["config"] = config
        return SimpleNamespace()

    monkeypatch.setattr(run_experiment, "create_llm_client", fake_create_llm_client)
    monkeypatch.setattr(run_experiment, "create_retriever", fake_create_retriever)
    monkeypatch.setattr(run_experiment, "create_architecture", fake_create_architecture)

    config = {
        "architecture": {"name": "planner_rag"},
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 1024},
        "retrieval": {"method": "bm25", "top_k": 3},
        "planner": {"max_iterations": 5},
    }

    run_experiment._build_rag(config)

    assert captured["name"] == "planner_rag"
    architecture_config = captured["config"]
    assert isinstance(architecture_config, dict)
    assert architecture_config["top_k"] == 3
    assert "max_context_tokens" not in architecture_config


def test_build_rag_uses_ircot_nested_config(monkeypatch) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_experiment = importlib.import_module("run_experiment")
    finally:
        sys.path.remove(str(scripts_dir))

    captured: dict[str, object] = {}

    def fake_create_llm_client(**kwargs):
        return SimpleNamespace(model="test-model")

    def fake_create_retriever(**kwargs):
        return object()

    def fake_create_architecture(name, llm, retriever, config):
        captured["name"] = name
        captured["config"] = config
        return SimpleNamespace()

    monkeypatch.setattr(run_experiment, "create_llm_client", fake_create_llm_client)
    monkeypatch.setattr(run_experiment, "create_retriever", fake_create_retriever)
    monkeypatch.setattr(run_experiment, "create_architecture", fake_create_architecture)

    config = {
        "architecture": {"name": "ircot_rag"},
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 1024},
        "retrieval": {"method": "bm25", "top_k": 3},
        "ircot": {"max_steps": 4, "max_context_tokens": 3000},
    }

    run_experiment._build_rag(config)

    assert captured["name"] == "ircot_rag"
    architecture_config = captured["config"]
    assert isinstance(architecture_config, dict)
    assert architecture_config["top_k"] == 3
    assert architecture_config["max_steps"] == 4
    assert architecture_config["max_context_tokens"] == 3000


def test_build_rag_uses_reap_nested_config(monkeypatch) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_experiment = importlib.import_module("run_experiment")
    finally:
        sys.path.remove(str(scripts_dir))

    captured: dict[str, object] = {}

    def fake_create_llm_client(**kwargs):
        return SimpleNamespace(model="test-model")

    def fake_create_retriever(**kwargs):
        return object()

    def fake_create_architecture(name, llm, retriever, config):
        captured["name"] = name
        captured["config"] = config
        return SimpleNamespace()

    monkeypatch.setattr(run_experiment, "create_llm_client", fake_create_llm_client)
    monkeypatch.setattr(run_experiment, "create_retriever", fake_create_retriever)
    monkeypatch.setattr(run_experiment, "create_architecture", fake_create_architecture)

    config = {
        "architecture": {"name": "reap_rag"},
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 1024},
        "retrieval": {"method": "bm25", "top_k": 3},
        "reap": {"max_iterations": 4, "max_context_tokens": 3000},
    }

    run_experiment._build_rag(config)

    assert captured["name"] == "reap_rag"
    architecture_config = captured["config"]
    assert isinstance(architecture_config, dict)
    assert architecture_config["top_k"] == 3
    assert architecture_config["max_iterations"] == 4
    assert architecture_config["max_context_tokens"] == 3000


@pytest.mark.asyncio
async def test_run_experiment_skips_global_index_for_question_scoped_corpus(monkeypatch) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_experiment = importlib.import_module("run_experiment")
    finally:
        sys.path.remove(str(scripts_dir))

    question_corpus = [Document(id="q1_doc", title="Doc", text="Scoped evidence")]
    questions = [
        Question(
            id="q1",
            text="Question?",
            type=QuestionType.BRIDGE,
            gold_answer="Answer",
            candidate_corpus=question_corpus,
        )
    ]
    merged_corpus = question_corpus + [Document(id="q2_doc", title="Doc", text="Other scope")]

    class FakeEvaluator:
        def __init__(self, rag, max_concurrency, dataset_name):
            self.rag = rag

        async def evaluate(self, questions_arg, corpus_arg):
            return BenchmarkResult(
                architecture="vanilla_rag",
                architecture_type=ArchitectureType.VANILLA,
                model="test-model",
                dataset="hotpotqa",
                num_questions=len(questions_arg),
                avg_exact_match=1.0,
                avg_f1=1.0,
                avg_supporting_fact_em=None,
                avg_supporting_fact_f1=None,
                metrics_by_type={},
                avg_latency_ms=1.0,
                avg_tokens_per_question=1.0,
                avg_retrieval_calls=1.0,
                avg_llm_calls=1.0,
                total_cost_usd=0.0,
                total_tokens=1,
                per_question_results=[],
            )

    index_calls: list[list[str]] = []

    async def fake_index(corpus):
        index_calls.append([doc.id for doc in corpus])

    fake_rag = SimpleNamespace(retriever=SimpleNamespace(index=fake_index))

    monkeypatch.setattr(run_experiment, "_build_rag", lambda config: fake_rag)
    monkeypatch.setattr(
        run_experiment, "load_hotpotqa", lambda **kwargs: (questions, merged_corpus)
    )
    monkeypatch.setattr(run_experiment, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(run_experiment, "save_results", lambda *args, **kwargs: None)

    config = {
        "data": {"dataset": "hotpotqa", "setting": "distractor"},
        "experiment": {"output_dir": str(Path("/tmp/test-results"))},
    }

    await run_experiment.run_experiment(config)

    assert index_calls == []
