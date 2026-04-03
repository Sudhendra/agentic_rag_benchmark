from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Iterable

from ..core.types import (
    ArchitectureType,
    BenchmarkResult,
    Document,
    EvaluationResult,
    Question,
    QuestionType,
)
from .metrics import exact_match, f1_score, joint_metrics, supporting_fact_evaluation

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        rag: object,
        max_concurrency: int = 5,
        dataset_name: str = "unknown",
        compute_supporting_facts: bool = False,
    ) -> None:
        self.rag = rag
        self.max_concurrency = max_concurrency
        self.dataset_name = dataset_name
        self.compute_supporting_facts = compute_supporting_facts

    async def evaluate(
        self,
        questions: list[Question],
        corpus: list[Document],
    ) -> BenchmarkResult:
        uses_question_scoped_corpus = any(
            self._get_question_corpus(question, corpus) is not corpus for question in questions
        )
        concurrency = 1 if uses_question_scoped_corpus else self.max_concurrency
        semaphore = asyncio.Semaphore(concurrency)
        total = len(questions)
        completed = 0
        errors = 0
        start_time = time.time()
        last_indexed_corpus: list[Document] | None = None

        architecture = getattr(self.rag, "get_name", lambda: "unknown")()
        print(
            f"\n[Evaluator] Starting {architecture} on {total} questions "
            f"(concurrency={concurrency})",
            flush=True,
        )

        if not uses_question_scoped_corpus:
            await self._index_corpus_if_needed(corpus, last_indexed_corpus)
            last_indexed_corpus = corpus

        async def run_one(question: Question) -> EvaluationResult:
            nonlocal completed, errors, last_indexed_corpus
            async with semaphore:
                question_corpus = self._get_question_corpus(question, corpus)
                try:
                    if uses_question_scoped_corpus and last_indexed_corpus is not question_corpus:
                        await self._index_corpus_if_needed(question_corpus, last_indexed_corpus)
                        last_indexed_corpus = question_corpus
                    response = await self.rag.answer(question, question_corpus)
                except Exception:
                    errors += 1
                    logger.exception("Error processing question %s", question.id)
                    raise

            gold_answer = question.gold_answer or ""
            answer_em = exact_match(response.answer, gold_answer)
            answer_f1 = f1_score(response.answer, gold_answer)
            predicted_supporting_facts = getattr(response, "supporting_facts", None)
            support_eval = supporting_fact_evaluation(
                pred_facts=predicted_supporting_facts,
                gold_facts=question.supporting_facts,
                compute_supporting_facts=self.compute_supporting_facts,
            )
            joint_em, joint_f1 = joint_metrics(
                answer_em, answer_f1, support_eval.em, support_eval.f1
            )

            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(
                    f"[Evaluator] {completed}/{total} done "
                    f"({completed / total * 100:.1f}%) | "
                    f"{rate:.1f} q/s | "
                    f"ETA: {eta / 60:.1f}m | "
                    f"errors: {errors}",
                    flush=True,
                )

            return EvaluationResult(
                question_id=question.id,
                question_type=question.type,
                exact_match=answer_em,
                f1=answer_f1,
                predicted_answer=response.answer,
                gold_answer=gold_answer,
                supporting_fact_em=support_eval.em,
                supporting_fact_f1=support_eval.f1,
                joint_em=joint_em,
                joint_f1=joint_f1,
                latency_ms=response.latency_ms,
                tokens_used=response.total_tokens,
                cost_usd=response.total_cost_usd,
                num_retrieval_calls=response.num_retrieval_calls,
                num_llm_calls=response.num_llm_calls,
                supporting_fact_status=support_eval.status,
                predicted_supporting_facts=predicted_supporting_facts,
                gold_supporting_facts=question.supporting_facts,
            )

        results = await asyncio.gather(*(run_one(question) for question in questions))

        num_questions = len(results)
        avg_exact_match = _safe_average([result.exact_match for result in results])
        avg_f1 = _safe_average([result.f1 for result in results])
        avg_latency_ms = _safe_average([result.latency_ms for result in results])
        avg_tokens_per_question = _safe_average([result.tokens_used for result in results])
        avg_retrieval_calls = _safe_average([result.num_retrieval_calls for result in results])
        avg_llm_calls = _safe_average([result.num_llm_calls for result in results])
        total_cost_usd = sum(result.cost_usd for result in results)
        total_tokens = sum(result.tokens_used for result in results)

        elapsed_total = time.time() - start_time
        print(
            f"\n[Evaluator] Complete! {num_questions} questions in {elapsed_total / 60:.1f}m | "
            f"EM: {avg_exact_match:.3f} | F1: {avg_f1:.3f} | "
            f"Cost: ${total_cost_usd:.4f}",
            flush=True,
        )

        supporting_fact_ems = [
            result.supporting_fact_em for result in results if result.supporting_fact_em is not None
        ]
        supporting_fact_f1s = [
            result.supporting_fact_f1 for result in results if result.supporting_fact_f1 is not None
        ]
        joint_ems = [result.joint_em for result in results if result.joint_em is not None]
        joint_f1s = [result.joint_f1 for result in results if result.joint_f1 is not None]
        avg_supporting_fact_em = _safe_average(supporting_fact_ems) if supporting_fact_ems else None
        avg_supporting_fact_f1 = _safe_average(supporting_fact_f1s) if supporting_fact_f1s else None
        avg_joint_em = _safe_average(joint_ems) if joint_ems else None
        avg_joint_f1 = _safe_average(joint_f1s) if joint_f1s else None

        metrics_by_type: dict[QuestionType, dict[str, float]] = {}
        grouped: dict[QuestionType, list[EvaluationResult]] = defaultdict(list)
        for result in results:
            grouped[result.question_type].append(result)
        for qtype, items in grouped.items():
            metrics_by_type[qtype] = {
                "exact_match": _safe_average([item.exact_match for item in items]),
                "f1": _safe_average([item.f1 for item in items]),
            }

        model = getattr(getattr(self.rag, "llm", None), "model", "unknown")
        architecture = getattr(self.rag, "get_name", lambda: "unknown")()
        architecture_type = getattr(self.rag, "get_type", lambda: ArchitectureType.VANILLA)()

        return BenchmarkResult(
            architecture=architecture,
            architecture_type=architecture_type,
            model=model,
            dataset=self.dataset_name,
            num_questions=num_questions,
            avg_exact_match=avg_exact_match,
            avg_f1=avg_f1,
            avg_supporting_fact_em=avg_supporting_fact_em,
            avg_supporting_fact_f1=avg_supporting_fact_f1,
            metrics_by_type=metrics_by_type,
            avg_latency_ms=avg_latency_ms,
            avg_tokens_per_question=avg_tokens_per_question,
            avg_retrieval_calls=avg_retrieval_calls,
            avg_llm_calls=avg_llm_calls,
            total_cost_usd=total_cost_usd,
            total_tokens=total_tokens,
            per_question_results=results,
            avg_joint_em=avg_joint_em,
            avg_joint_f1=avg_joint_f1,
        )

    @staticmethod
    def _get_question_corpus(question: Question, default_corpus: list[Document]) -> list[Document]:
        if question.candidate_corpus is not None:
            return question.candidate_corpus
        return default_corpus

    async def _index_corpus_if_needed(
        self,
        corpus: list[Document],
        last_indexed_corpus: list[Document] | None,
    ) -> None:
        retriever = getattr(self.rag, "retriever", None)
        if retriever is None or not hasattr(retriever, "index"):
            return
        if last_indexed_corpus is corpus:
            return
        await retriever.index(corpus)


def _safe_average(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)
