from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from src.architectures.agentic.planner_rag import PlannerRAG
from src.architectures.agentic.react_rag import ReActRAG
from src.architectures.agentic.self_rag import SelfRAG
from src.architectures.recursive.ircot import IRCoTRAG
from src.architectures.recursive.reap import REAPRAG
from src.architectures.rlm.recursive_lm import RecursiveLM
from src.architectures.vanilla_rag import VanillaRAG
from src.core.types import Document, Question, QuestionType, RetrievalResult


class RecordingLLM:
    def __init__(self, responses: Sequence[str]):
        self.model = "test-model"
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> tuple[str, int, float]:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                "seed": seed,
            }
        )
        response = self._responses.pop(0)
        return response, 11, 0.01


class StubRetriever:
    def __init__(self):
        self.documents = [Document(id="d1", title="Doc", text="Evidence about Paris.")]

    async def retrieve(self, query: str, corpus: list[Document], top_k: int = 5) -> RetrievalResult:
        return RetrievalResult(
            documents=self.documents[:top_k],
            scores=[0.9] * min(top_k, len(self.documents)),
            query=query,
            retrieval_time_ms=1.0,
            method="bm25",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rag_cls", "config", "responses", "question"),
    [
        (VanillaRAG, {"top_k": 1}, ["Paris"], Question("q1", "Capital?", QuestionType.SINGLE_HOP)),
        (
            ReActRAG,
            {"top_k": 1, "max_iterations": 1},
            ["Thought: done\nAction: finish[Paris]"],
            Question("q1", "Capital?", QuestionType.SINGLE_HOP),
        ),
        (
            SelfRAG,
            {"top_k": 1, "num_candidates": 1},
            ["[retrieval] no", "Paris", "[IsUse] 5"],
            Question("q1", "Capital?", QuestionType.SINGLE_HOP),
        ),
        (
            PlannerRAG,
            {"max_iterations": 1},
            ['{"direct_answer": true}', "Paris"],
            Question("q1", "Capital?", QuestionType.SINGLE_HOP),
        ),
        (
            IRCoTRAG,
            {"top_k": 1, "max_steps": 1},
            ["[ANSWER] Paris", "Paris"],
            Question("q1", "Capital?", QuestionType.SINGLE_HOP),
        ),
        (
            REAPRAG,
            {"top_k": 1, "max_iterations": 1},
            [
                '{"user_goal": "Capital?", "requirements": [{"requirement_id": "r1", "question": "Capital?", "depends_on": [], "status": "pending"}]}',
                '{"next_step": "SYNTHESIZE_ANSWER", "updated_plan": [{"requirement_id": "r1", "question": "Capital?", "depends_on": [], "status": "pending"}], "next_actions": []}',
                "Paris",
            ],
            Question("q1", "Capital?", QuestionType.SINGLE_HOP),
        ),
        (
            RecursiveLM,
            {"top_k": 1, "max_depth": 1, "memoization": False},
            ["DIRECT: Paris"],
            Question("q1", "Capital?", QuestionType.SINGLE_HOP),
        ),
    ],
)
async def test_architectures_propagate_generation_config(
    rag_cls,
    config: dict[str, Any],
    responses: list[str],
    question: Question,
) -> None:
    llm = RecordingLLM(responses)
    retriever = StubRetriever()
    rag = rag_cls(
        llm,
        retriever,
        {
            **config,
            "generation_temperature": 0.42,
            "generation_max_tokens": 256,
            "generation_seed": 99,
        },
    )

    await rag.answer(question, retriever.documents)

    assert llm.calls
    for call in llm.calls:
        assert call["temperature"] == pytest.approx(0.42)
        assert call["max_tokens"] == 256
        assert call["seed"] == 99
