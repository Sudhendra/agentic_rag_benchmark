"""Self-RAG architecture implementation.

Self-RAG adds self-reflection tokens to decide whether to retrieve,
assess relevance of retrieved passages, critique generated answers
for support, and select the best candidate based on utility scoring.

Reference: Self-RAG: Learning to Retrieve, Generate, and Critique
(Asai et al., ICLR 2024) — https://arxiv.org/abs/2310.11511
"""

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

# Support level scores for candidate ranking
_SUPPORT_SCORES: dict[str, float] = {
    "fully supported": 1.0,
    "fully": 1.0,
    "partially supported": 0.5,
    "partially": 0.5,
    "no support": 0.0,
    "no": 0.0,
}


class SelfRAG(BaseRAG):
    """Self-RAG: self-reflective retrieval-augmented generation.

    Implements a 4-phase pipeline:
    1. Retrieval decision — should we retrieve?
    2. Retrieve + relevance filtering
    3. Generate candidates + critique support + rate utility
    4. Select best candidate by utility (tie-break with support)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: BaseRetriever,
        config: dict,
    ):
        super().__init__(llm_client, retriever, config)

        prompt_path = self.config.get("prompt_path", "prompts/self_rag.txt")
        if Path(prompt_path).exists():
            self.retrieval_decision_prompt = self._load_prompt_template(prompt_path)
        else:
            self.retrieval_decision_prompt = (
                "You are a self-reflective question answering system.\n\n"
                "Given the question, decide if you need to retrieve external "
                "information or can answer directly.\n\n"
                "Question: {question}\n\n"
                "Do you need to retrieve information to answer this question "
                'accurately?\nAnswer with exactly "yes" or "no":\n'
            )

    def get_name(self) -> str:
        return "self_rag"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.AGENTIC

    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        return {
            "top_k": (int, False, 5),
            "num_candidates": (int, False, 3),
            "max_context_tokens": (int, False, 4000),
            "prompt_path": (str, False, "prompts/self_rag.txt"),
        }

    async def answer(
        self,
        question: Question,
        corpus: list[Document],
    ) -> RAGResponse:
        start_time = time.perf_counter()
        reasoning_chain: list[ReasoningStep] = []
        retrieved_docs: list[RetrievalResult] = []
        total_tokens = 0
        total_cost = 0.0
        num_llm_calls = 0
        num_retrieval_calls = 0
        step_id = 0

        # ── Phase 1: Retrieval decision ──────────────────────────
        step_id += 1
        retrieval_needed, tokens, cost = await self._decide_retrieval(question)
        total_tokens += tokens
        total_cost += cost
        num_llm_calls += 1

        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought=f"Decide whether retrieval is needed for: {question.text}",
                action="retrieval_decision",
                action_input=question.text,
                observation=f"Retrieval needed: {retrieval_needed}",
                tokens_used=tokens,
                cost_usd=cost,
            )
        )

        # ── No-retrieval path ────────────────────────────────────
        if not retrieval_needed:
            step_id += 1
            answer_text, tokens, cost = await self._direct_answer(question)
            total_tokens += tokens
            total_cost += cost
            num_llm_calls += 1

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought="No retrieval needed; answering directly.",
                    action="direct_answer",
                    action_input=question.text,
                    observation=f"Direct answer: {answer_text}",
                    tokens_used=tokens,
                    cost_usd=cost,
                )
            )

            step_id += 1
            utility, util_tokens, util_cost = await self._rate_utility(question, answer_text)
            total_tokens += util_tokens
            total_cost += util_cost
            num_llm_calls += 1

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought="Rate utility of the direct answer.",
                    action="rate_utility",
                    action_input=answer_text[:200],
                    observation=f"Utility: {utility}/5",
                    tokens_used=util_tokens,
                    cost_usd=util_cost,
                )
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return RAGResponse(
                answer=answer_text,
                reasoning_chain=reasoning_chain,
                retrieved_docs=retrieved_docs,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                latency_ms=elapsed_ms,
                num_retrieval_calls=0,
                num_llm_calls=num_llm_calls,
                model=self.llm.model,
                architecture=self.get_name(),
            )

        # ── Phase 2: Retrieve passages ───────────────────────────
        step_id += 1
        retrieval_result = await self.retriever.retrieve(
            query=question.text,
            corpus=corpus,
            top_k=self.config["top_k"],
        )
        num_retrieval_calls += 1
        retrieved_docs.append(retrieval_result)

        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought="Retrieving passages for the question.",
                action="retrieve",
                action_input=question.text,
                observation=f"Retrieved {len(retrieval_result.documents)} passages.",
                tokens_used=0,
                cost_usd=0.0,
            )
        )

        # ── Phase 3: Relevance filtering ─────────────────────────
        retrieved_documents = retrieval_result.documents[: self.config["top_k"]]
        relevant_passages: list[Document] = []
        for passage in retrieved_documents:
            step_id += 1
            relevance, rel_tokens, rel_cost = await self._assess_relevance(question, passage)
            total_tokens += rel_tokens
            total_cost += rel_cost
            num_llm_calls += 1

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought=f"Assess relevance of passage '{passage.title}'.",
                    action="assess_relevance",
                    action_input=passage.text[:200],
                    observation=f"Relevance: {relevance}",
                    tokens_used=rel_tokens,
                    cost_usd=rel_cost,
                )
            )

            if relevance == "relevant":
                relevant_passages.append(passage)

        if not relevant_passages:
            relevant_passages = retrieved_documents

        candidate_passages = relevant_passages[: self.config["num_candidates"]]
        candidates: list[dict[str, Any]] = []

        for i, passage in enumerate(candidate_passages, start=1):
            # 3a: Generate answer from passage
            step_id += 1
            generation, gen_tokens, gen_cost = await self._generate_answer(question, passage)
            total_tokens += gen_tokens
            total_cost += gen_cost
            num_llm_calls += 1

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought=f"Generate answer from passage '{passage.title}'.",
                    action="generate_answer",
                    action_input=passage.text[:200],
                    observation=f"Generation: {generation[:100]}...",
                    tokens_used=gen_tokens,
                    cost_usd=gen_cost,
                )
            )

            # 3b: Critique support
            step_id += 1
            support, sup_tokens, sup_cost = await self._critique_support(
                question, passage, generation
            )
            total_tokens += sup_tokens
            total_cost += sup_cost
            num_llm_calls += 1

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought=f"Critique support for candidate {i}.",
                    action="critique_support",
                    action_input=generation[:200],
                    observation=f"Support: {support}",
                    tokens_used=sup_tokens,
                    cost_usd=sup_cost,
                )
            )

            # 3c: Rate utility
            step_id += 1
            utility, util_tokens, util_cost = await self._rate_utility(question, generation)
            total_tokens += util_tokens
            total_cost += util_cost
            num_llm_calls += 1

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought=f"Rate overall utility of candidate {i}.",
                    action="rate_utility",
                    action_input=generation[:200],
                    observation=f"Utility: {utility}/5",
                    tokens_used=util_tokens,
                    cost_usd=util_cost,
                )
            )

            candidates.append(
                {
                    "generation": generation,
                    "relevance": "relevant" if passage in relevant_passages else "irrelevant",
                    "support": support,
                    "utility": utility,
                    "passage_title": passage.title,
                }
            )

        # ── Phase 4: Select best candidate ───────────────────────
        best = self._select_best_candidate(candidates)
        answer_text = best["generation"]

        step_id += 1
        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought="Select the best candidate based on utility and support.",
                action="select_best",
                action_input=f"{len(candidates)} candidates evaluated",
                observation=(
                    f"Selected candidate from '{best['passage_title']}' "
                    f"(utility={best['utility']}, support={best['support']})"
                ),
                tokens_used=0,
                cost_usd=0.0,
            )
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            answer=answer_text,
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

    # ── Private helpers ──────────────────────────────────────────

    async def _decide_retrieval(self, question: Question) -> tuple[bool, int, float]:
        """Phase 1: Decide whether retrieval is needed.

        Returns:
            (retrieval_needed, tokens_used, cost)
        """
        prompt = self.retrieval_decision_prompt.format(question=question.text)
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)

        normalized = response_text.strip().lower()
        token_match = re.search(r"\[retrieval\]\s*(yes|no|continue)", normalized)
        if token_match:
            token_value = token_match.group(1)
            retrieval_needed = token_value == "yes"
        else:
            # Default to yes (retrieve) unless explicitly "no"
            retrieval_needed = "no" not in normalized.split()[:3]

        return retrieval_needed, tokens, cost

    async def _direct_answer(self, question: Question) -> tuple[str, int, float]:
        """Generate a direct answer without retrieval.

        Returns:
            (answer_text, tokens_used, cost)
        """
        prompt = (
            f"Answer the following question concisely and directly.\n\n"
            f"Question: {question.text}\n\n"
            f"Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)
        return response_text.strip(), tokens, cost

    async def _assess_relevance(
        self, question: Question, passage: Document
    ) -> tuple[str, int, float]:
        """Assess relevance of a passage to the question.

        Returns:
            (relevance, tokens_used, cost)
        """
        prompt = (
            f"Is the following passage relevant to the question?\n\n"
            f"Passage ({passage.title}): {passage.text}\n\n"
            f"Question: {question.text}\n\n"
            f"Answer with exactly: [IsRel] relevant or [IsRel] irrelevant"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)
        relevance = self._parse_relevance(response_text)
        return relevance, tokens, cost

    async def _generate_answer(
        self, question: Question, passage: Document
    ) -> tuple[str, int, float]:
        """Generate an answer from a passage.

        Returns:
            (generation, tokens_used, cost)
        """
        prompt = (
            f"Answer the question based on the provided passage.\n\n"
            f"Passage ({passage.title}): {passage.text}\n\n"
            f"Question: {question.text}\n\n"
            f"Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)
        return response_text.strip(), tokens, cost

    async def _critique_support(
        self, question: Question, passage: Document, generation: str
    ) -> tuple[str, int, float]:
        """Critique support of a generation using the passage.

        Returns:
            (support_level, tokens_used, cost)
        """
        prompt = (
            f"Is the answer supported by the passage?\n\n"
            f"Passage ({passage.title}): {passage.text}\n\n"
            f"Question: {question.text}\n"
            f"Answer: {generation}\n\n"
            f"Respond with exactly one of:\n"
            f"[IsSup] fully supported\n"
            f"[IsSup] partially supported\n"
            f"[IsSup] no support"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)
        support = self._parse_support(response_text)
        return support, tokens, cost

    async def _rate_utility(self, question: Question, generation: str) -> tuple[int, int, float]:
        """Rate the utility of a generation on a 1-5 scale.

        Returns:
            (utility_score, tokens_used, cost)
        """
        prompt = (
            f"Rate how useful the following answer is for the question "
            f"on a scale of 1 to 5 (1=useless, 5=perfect).\n\n"
            f"Question: {question.text}\n"
            f"Answer: {generation}\n\n"
            f"Respond with exactly: [IsUse] 1-5"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)

        utility = self._parse_utility(response_text)
        return utility, tokens, cost

    def _parse_relevance(self, response_text: str) -> str:
        """Parse relevance from LLM response.

        Returns:
            "relevant" or "irrelevant"
        """
        normalized = response_text.strip().lower()
        match = re.search(r"\[isrel\]\s*([^\n\r]+)", normalized)
        value = match.group(1).strip() if match else normalized
        if "irrelevant" in value:
            return "irrelevant"
        if "relevant" in value:
            return "relevant"
        return "relevant"

    def _parse_support(self, response_text: str) -> str:
        """Parse support level from LLM response.

        Returns:
            "fully supported", "partially supported", or "no support"
        """
        normalized = response_text.strip().lower()
        match = re.search(r"\[issup\]\s*([^\n\r]+)", normalized)
        value = match.group(1).strip() if match else normalized
        if "fully" in value:
            return "fully supported"
        if "no support" in value or value.strip() == "no":
            return "no support"
        if "partially" in value:
            return "partially supported"
        return "partially supported"

    def _parse_utility(self, response_text: str) -> int:
        """Parse a utility score (1-5) from LLM response.

        Returns:
            Integer utility score, defaults to 3 on parse failure.
        """
        normalized = response_text.strip().lower()
        token_match = re.search(r"\[isuse\]\s*([1-5])", normalized)
        if token_match:
            return int(token_match.group(1))
        match = re.search(r"[1-5]", normalized)
        if match:
            return int(match.group(0))
        return 3  # default to middle score

    def _select_best_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Select the best candidate based on composite score.

        Selection priority: highest utility, tie-break by support level.

        Returns:
            The best candidate dict.
        """
        if not candidates:
            return {
                "generation": "No answer.",
                "relevance": "irrelevant",
                "support": "no support",
                "utility": 0,
                "passage_title": "",
            }

        def _score(candidate: dict[str, Any]) -> tuple[int, float]:
            utility = candidate["utility"]
            support_bonus = _SUPPORT_SCORES.get(candidate["support"], 0.0)
            return utility, support_bonus

        return max(candidates, key=_score)
