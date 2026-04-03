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


def test_build_rag_propagates_generation_config_without_overwriting_context(monkeypatch) -> None:
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
        "architecture": {"name": "vanilla_rag"},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.65,
            "max_tokens": 321,
        },
        "experiment": {"seed": 17},
        "retrieval": {"method": "bm25", "top_k": 3},
        "vanilla": {"max_context_tokens": 2222},
    }

    run_experiment._build_rag(config)

    assert captured["name"] == "vanilla_rag"
    architecture_config = captured["config"]
    assert isinstance(architecture_config, dict)
    assert architecture_config["generation_temperature"] == 0.65
    assert architecture_config["generation_max_tokens"] == 321
    assert architecture_config["generation_seed"] == 17
    assert architecture_config["max_context_tokens"] == 2222


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
        def __init__(
            self,
            rag,
            max_concurrency,
            dataset_name,
            compute_supporting_facts=False,
        ):
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


@pytest.mark.asyncio
async def test_run_experiment_passes_supporting_fact_flag_to_evaluator(monkeypatch) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_experiment = importlib.import_module("run_experiment")
    finally:
        sys.path.remove(str(scripts_dir))

    captured: dict[str, object] = {}

    class FakeEvaluator:
        def __init__(self, rag, max_concurrency, dataset_name, compute_supporting_facts=False):
            captured["compute_supporting_facts"] = compute_supporting_facts

        async def evaluate(self, questions_arg, corpus_arg):
            return BenchmarkResult(
                architecture="vanilla_rag",
                architecture_type=ArchitectureType.VANILLA,
                model="test-model",
                dataset="hotpotqa",
                num_questions=len(questions_arg),
                avg_exact_match=1.0,
                avg_f1=1.0,
                avg_supporting_fact_em=1.0,
                avg_supporting_fact_f1=1.0,
                metrics_by_type={},
                avg_latency_ms=1.0,
                avg_tokens_per_question=1.0,
                avg_retrieval_calls=1.0,
                avg_llm_calls=1.0,
                total_cost_usd=0.0,
                total_tokens=1,
                per_question_results=[],
                avg_joint_em=1.0,
                avg_joint_f1=1.0,
            )

    question = Question(
        id="q1",
        text="Question?",
        type=QuestionType.BRIDGE,
        gold_answer="Answer",
        supporting_facts=[("Doc1", 0)],
    )
    corpus = [Document(id="d1", title="Doc", text="Doc text")]

    monkeypatch.setattr(run_experiment, "_build_rag", lambda config: object())
    monkeypatch.setattr(run_experiment, "load_hotpotqa", lambda **kwargs: ([question], corpus))
    monkeypatch.setattr(run_experiment, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(run_experiment, "save_results", lambda *args, **kwargs: None)

    config = {
        "data": {"dataset": "hotpotqa", "setting": "distractor"},
        "evaluation": {"compute_supporting_facts": True},
        "experiment": {"output_dir": str(Path("/tmp/test-results"))},
    }

    await run_experiment.run_experiment(config)

    assert captured["compute_supporting_facts"] is True


@pytest.mark.asyncio
async def test_run_experiment_uses_unique_manual_run_dirs_and_saves_seed(
    monkeypatch, tmp_path: Path
) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        run_experiment = importlib.import_module("run_experiment")
    finally:
        sys.path.remove(str(scripts_dir))

    saved_configs: list[dict[str, object]] = []

    class FakeEvaluator:
        def __init__(self, rag, max_concurrency, dataset_name, compute_supporting_facts=False):
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
                avg_joint_em=None,
                avg_joint_f1=None,
            )

    monkeypatch.setattr(run_experiment, "_build_rag", lambda config: object())
    monkeypatch.setattr(
        run_experiment,
        "load_hotpotqa",
        lambda **kwargs: (
            [Question(id="q1", text="Question?", type=QuestionType.SINGLE_HOP, gold_answer="A")],
            [Document(id="d1", title="Doc", text="Doc text")],
        ),
    )
    monkeypatch.setattr(run_experiment, "Evaluator", FakeEvaluator)

    def fake_save_results(result, output_dir, resolved_config):
        saved_configs.append(resolved_config)

    monkeypatch.setattr(run_experiment, "save_results", fake_save_results)

    config = {
        "data": {"dataset": "hotpotqa", "setting": "distractor"},
        "experiment": {"output_dir": str(tmp_path), "seed": 1234},
    }

    run_dir_1 = await run_experiment.run_experiment(config)
    run_dir_2 = await run_experiment.run_experiment(config)

    assert run_dir_1 != run_dir_2
    assert run_dir_1.parent == tmp_path
    assert run_dir_2.parent == tmp_path
    assert saved_configs[0]["experiment"]["seed"] == 1234
    assert saved_configs[1]["experiment"]["seed"] == 1234
