"""IRCoT recursive RAG architecture implementation."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ...core.base_rag import BaseRAG
from ...core.llm_client import BaseLLMClient
from ...core.retriever import BaseRetriever
from ...core.types import (
    ArchitectureType,
    Document,
    Question,
    RAGResponse,
    ReasoningStep,
    RetrievalResult,
)

_DEFAULT_REASON_PROMPT = (
    "You are solving a multi-hop question with interleaved retrieval.\n\n"
    "Use the retrieved evidence and prior reasoning to write exactly one next reasoning sentence. "
    "Do not guess. Do not mention what you need to do, what evidence is missing, or that you need to identify something. "
    "State only the strongest grounded fact or bridgeable inference from the current evidence. "
    "If you now know the final answer, output `{answer_trigger} <answer>` instead.\n\n"
    "Question: {question}\n\n"
    "Evidence:\n{context}\n\n"
    "Reasoning so far:\n{reasoning_history}\n\n"
    "Write one next reasoning sentence or `{answer_trigger} <answer>`."
)

_DEFAULT_FINAL_PROMPT = (
    "Answer the question using the accumulated evidence below.\n"
    "Return only the final answer phrase with no explanation.\n\n"
    "Question: {question}\n\n"
    "Evidence:\n{context}\n\n"
    "Reasoning trace:\n{reasoning_history}\n\n"
    "Candidate answer: {candidate_answer}\n\n"
    "Answer:"
)


class IRCoTRAG(BaseRAG):
    """Interleave retrieval with chain-of-thought reasoning steps."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: BaseRetriever,
        config: dict,
    ):
        super().__init__(llm_client, retriever, config)

        prompt_path = self.config["prompt_path"]
        final_prompt_path = self.config["final_prompt_path"]
        self.reason_prompt = self._load_optional_prompt(prompt_path, _DEFAULT_REASON_PROMPT)
        self.final_prompt = self._load_optional_prompt(final_prompt_path, _DEFAULT_FINAL_PROMPT)

    def get_name(self) -> str:
        return "ircot_rag"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.RECURSIVE

    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        return {
            "top_k": (int, False, 3),
            "max_steps": (int, False, 4),
            "max_context_tokens": (int, False, 3000),
            "answer_trigger": (str, False, "[ANSWER]"),
            "prompt_path": (str, False, "prompts/ircot.txt"),
            "final_prompt_path": (str, False, "prompts/ircot_final.txt"),
            "max_docs": (int, False, 8),
        }

    async def answer(
        self,
        question: Question,
        corpus: list[Document],
    ) -> RAGResponse:
        start_time = time.perf_counter()
        reasoning_chain: list[ReasoningStep] = []
        retrieved_docs: list[RetrievalResult] = []
        reasoning_sentences: list[str] = []
        evidence_docs: list[Document] = []
        seen_doc_ids: set[str] = set()
        total_tokens = 0
        total_cost = 0.0
        num_llm_calls = 0
        num_retrieval_calls = 0
        step_id = 0
        candidate_answer = ""

        initial_result = await self.retriever.retrieve(
            query=question.text,
            corpus=corpus,
            top_k=self.config["top_k"],
        )
        retrieved_docs.append(initial_result)
        num_retrieval_calls += 1
        self._extend_evidence(evidence_docs, seen_doc_ids, initial_result.documents)

        for _ in range(self.config["max_steps"]):
            prompt = self._build_reason_prompt(question.text, evidence_docs, reasoning_sentences)
            messages = [{"role": "user", "content": prompt}]
            response_text, tokens_used, cost = await self.llm.generate(messages)
            num_llm_calls += 1
            total_tokens += tokens_used
            total_cost += cost

            step_text = self._first_line(response_text)
            reasoning_sentences.append(step_text)
            extracted_answer = self._extract_answer(step_text, self.config["answer_trigger"])

            step_id += 1
            if extracted_answer:
                candidate_answer = extracted_answer
                reasoning_chain.append(
                    ReasoningStep(
                        step_id=step_id,
                        thought=step_text,
                        action="finish",
                        action_input=question.text,
                        observation=extracted_answer,
                        tokens_used=tokens_used,
                        cost_usd=cost,
                    )
                )
                break

            query = self._extract_query(step_text)
            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought=step_text,
                    action="reason",
                    action_input=query,
                    observation=step_text,
                    tokens_used=tokens_used,
                    cost_usd=cost,
                )
            )

            retrieval_result = await self.retriever.retrieve(
                query=query or question.text,
                corpus=corpus,
                top_k=self.config["top_k"],
            )
            retrieved_docs.append(retrieval_result)
            num_retrieval_calls += 1
            self._extend_evidence(evidence_docs, seen_doc_ids, retrieval_result.documents)

        final_prompt = self._build_final_prompt(
            question.text, evidence_docs, reasoning_sentences, candidate_answer
        )
        final_messages = [{"role": "user", "content": final_prompt}]
        final_answer, final_tokens, final_cost = await self.llm.generate(final_messages)
        num_llm_calls += 1
        total_tokens += final_tokens
        total_cost += final_cost

        step_id += 1
        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought="Synthesize final answer from accumulated evidence.",
                action="synthesize",
                action_input=question.text,
                observation=final_answer.strip(),
                tokens_used=final_tokens,
                cost_usd=final_cost,
            )
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return RAGResponse(
            answer=final_answer.strip(),
            reasoning_chain=reasoning_chain,
            retrieved_docs=retrieved_docs,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            latency_ms=elapsed_ms,
            num_retrieval_calls=num_retrieval_calls,
            num_llm_calls=num_llm_calls,
            model=self.llm.model,
            architecture=self.get_name(),
        )

    def _build_reason_prompt(
        self,
        question: str,
        evidence_docs: list[Document],
        reasoning_sentences: list[str],
    ) -> str:
        return self.reason_prompt.format(
            answer_trigger=self.config["answer_trigger"],
            question=question,
            context=self._build_context(
                evidence_docs[: self.config["max_docs"]],
                max_tokens=self.config["max_context_tokens"],
            ),
            reasoning_history="\n".join(reasoning_sentences)
            if reasoning_sentences
            else "None yet.",
        )

    def _build_final_prompt(
        self,
        question: str,
        evidence_docs: list[Document],
        reasoning_sentences: list[str],
        candidate_answer: str,
    ) -> str:
        return self.final_prompt.format(
            question=question,
            context=self._build_context(
                evidence_docs[: self.config["max_docs"]],
                max_tokens=self.config["max_context_tokens"],
            ),
            reasoning_history="\n".join(reasoning_sentences) if reasoning_sentences else "None.",
            candidate_answer=candidate_answer or "None",
        )

    def _load_optional_prompt(self, prompt_path: str, default_prompt: str) -> str:
        if Path(prompt_path).exists():
            return self._load_prompt_template(prompt_path)
        return default_prompt

    def _extend_evidence(
        self,
        evidence_docs: list[Document],
        seen_doc_ids: set[str],
        new_docs: list[Document],
    ) -> None:
        for doc in new_docs:
            if doc.id in seen_doc_ids:
                continue
            evidence_docs.append(doc)
            seen_doc_ids.add(doc.id)
        if len(evidence_docs) > self.config["max_docs"]:
            del evidence_docs[self.config["max_docs"] :]

    @staticmethod
    def _first_line(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        return stripped.splitlines()[0].strip()

    @staticmethod
    def _extract_answer(text: str, answer_trigger: str = "[ANSWER]") -> str:
        trigger_pattern = re.escape(answer_trigger)
        trigger_match = re.search(rf"{trigger_pattern}\s*(.+)", text, re.IGNORECASE)
        if trigger_match:
            return trigger_match.group(1).strip()

        fallback_match = re.search(r"answer\s+is\s*:\s*(.+)", text, re.IGNORECASE)
        if fallback_match:
            return fallback_match.group(1).strip().rstrip(".")

        return ""

    @staticmethod
    def _extract_query(text: str) -> str:
        stopwords = {
            "a",
            "actress",
            "an",
            "and",
            "answer",
            "as",
            "by",
            "because",
            "determine",
            "finally",
            "for",
            "hence",
            "i",
            "if",
            "in",
            "indeed",
            "identify",
            "is",
            "it",
            "its",
            "need",
            "of",
            "portrayed",
            "so",
            "that",
            "the",
            "therefore",
            "this",
            "those",
            "through",
            "thus",
            "to",
            "was",
            "we",
            "were",
            "what",
            "who",
            "whom",
            "which",
            "whether",
            "woman",
            "we",
        }
        tokens = re.findall(r"[A-Za-z0-9]+", text)
        ordered: list[str] = []
        seen: set[str] = set()

        for token in tokens:
            lowered = token.lower()
            is_short_named_entity = len(token) <= 2 and token[:1].isupper()
            if (
                lowered in stopwords
                or lowered == "s"
                or (len(token) <= 2 and not is_short_named_entity)
            ):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(token)

        return " ".join(ordered)
