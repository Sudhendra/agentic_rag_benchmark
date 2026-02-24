"""Recursive Language Model (RLM) RAG architecture implementation.

Based on the RLM paper (arXiv 2512.24601), this implements programmatic
self-recursion for multi-hop question answering. The LLM decides whether
to answer directly (with retrieval context) or decompose the question
into sub-questions, recursively solve each, and combine the results.

Key differences from the alexzhang13/rlm library:
- No REPL/code execution sandbox -- uses structured prompting instead.
- Integrated with our retrieval pipeline (BM25/Dense/Hybrid).
- Uses our LLM client with caching and cost tracking.
- Produces structured ReasoningStep chains for evaluation.
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
    QuestionType,
    RAGResponse,
    ReasoningStep,
    RetrievalResult,
)

# Default decomposition prompt when file is missing.
_DEFAULT_DECOMPOSE_PROMPT = (
    "You are solving a complex question by recursive decomposition with retrieval.\n\n"
    "Question: {question}\n\n"
    "{context_section}\n\n"
    "You have two options:\n"
    "1. Answer directly if the question is simple enough given the available context.\n"
    "   Format: DIRECT: <your answer>\n"
    "   IMPORTANT: The answer must be SHORT and EXTRACTIVE — just the answer itself "
    'with no explanation or reasoning. For example: "yes", "Paris", "42", "Albert Einstein".\n\n'
    "2. Decompose into sub-questions that can be answered independently, then combined.\n"
    "   Format:\n"
    "   DECOMPOSE:\n"
    "   - SUB: <sub_question_1>\n"
    "   - SUB: <sub_question_2>\n"
    "   COMBINE: <instruction for how to combine the sub-answers into a final answer>\n\n"
    "Guidelines:\n"
    "- Prefer DIRECT when the context already contains the answer or the question is simple.\n"
    "- Use DECOMPOSE when the question requires multiple pieces of information.\n"
    "- Sub-questions should be self-contained and answerable independently.\n"
    "- Keep the number of sub-questions small (2-4 is ideal).\n"
    "- The COMBINE instruction should clearly explain how to merge sub-answers.\n"
    '- For yes/no questions, always prefer DIRECT with just "yes" or "no".\n\n'
    "OUTPUT:\n"
)

_DEFAULT_COMBINE_PROMPT = (
    "You are combining answers to sub-questions to answer an original question.\n\n"
    "Original question: {question}\n\n"
    "Sub-questions and their answers:\n"
    "{sub_qa_pairs}\n\n"
    "Combination instruction: {combine_instruction}\n\n"
    "Based on the sub-answers above, provide a SHORT, EXTRACTIVE final answer to the "
    "original question.\n"
    "Give ONLY the answer with no explanation. "
    'For example: "yes", "Paris", "42", "Albert Einstein".\n\n'
    "Answer:"
)


class RecursiveLM(BaseRAG):
    """Recursive Language Model: programmatic self-recursion for multi-hop QA.

    The LLM examines a question (with retrieval context) and decides to either:
    1. Answer directly (base case), or
    2. Decompose into sub-questions, recursively solve each with retrieval,
       and combine results (recursive case).

    This mirrors the RLM paper's decompose-recurse-combine pattern while
    integrating with our retrieval and evaluation infrastructure.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: BaseRetriever,
        config: dict,
    ):
        super().__init__(llm_client, retriever, config)

        # Load decomposition prompt
        prompt_path = self.config.get("prompt_path", "prompts/rlm.txt")
        if Path(prompt_path).exists():
            self.decompose_prompt = self._load_prompt_template(prompt_path)
        else:
            self.decompose_prompt = _DEFAULT_DECOMPOSE_PROMPT

        # Load combination prompt
        combine_path = self.config.get("combine_prompt_path", "prompts/rlm_combine.txt")
        if Path(combine_path).exists():
            self.combine_prompt = self._load_prompt_template(combine_path)
        else:
            self.combine_prompt = _DEFAULT_COMBINE_PROMPT

    def get_name(self) -> str:
        return "recursive_lm"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.RLM

    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        return {
            "max_depth": (int, False, 3),
            "top_k": (int, False, 5),
            "memoization": (bool, False, True),
            "max_context_tokens": (int, False, 4000),
            "prompt_path": (str, False, "prompts/rlm.txt"),
            "combine_prompt_path": (str, False, "prompts/rlm_combine.txt"),
        }

    async def answer(
        self,
        question: Question,
        corpus: list[Document],
    ) -> RAGResponse:
        """Answer a question using recursive decomposition with retrieval.

        Args:
            question: The question to answer.
            corpus: List of documents to retrieve from.

        Returns:
            RAGResponse with answer, reasoning chain, and metadata.
        """
        start_time = time.perf_counter()

        # Shared mutable state across the recursion tree.
        state = _RecursionState(memoization=self.config["memoization"])

        answer_text = await self._recursive_answer(
            question_text=question.text,
            corpus=corpus,
            depth=0,
            state=state,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            answer=answer_text,
            reasoning_chain=state.reasoning_chain,
            retrieved_docs=state.retrieved_docs,
            total_tokens=state.total_tokens,
            total_cost_usd=state.total_cost,
            latency_ms=elapsed_ms,
            num_retrieval_calls=state.num_retrieval_calls,
            num_llm_calls=state.num_llm_calls,
            model=self.llm.model,
            architecture=self.get_name(),
        )

    async def _recursive_answer(
        self,
        question_text: str,
        corpus: list[Document],
        depth: int,
        state: _RecursionState,
    ) -> str:
        """Recursively answer a question, decomposing if needed.

        Args:
            question_text: The question string to answer.
            corpus: Document corpus for retrieval.
            depth: Current recursion depth (0-indexed).
            state: Shared state for tracking metrics across recursion.

        Returns:
            Answer string.
        """
        max_depth = self.config["max_depth"]

        # Check memo cache.
        if state.memoization:
            cache_key = question_text.strip().lower()
            if cache_key in state.memo:
                state.add_step(
                    thought=f"[depth={depth}] Cache hit for: {question_text}",
                    action="memo_hit",
                    action_input=question_text,
                    observation=state.memo[cache_key],
                )
                return state.memo[cache_key]

        # Retrieve context for this question.
        retrieval_result = await self.retriever.retrieve(
            query=question_text,
            corpus=corpus,
            top_k=self.config["top_k"],
        )
        state.num_retrieval_calls += 1
        state.retrieved_docs.append(retrieval_result)

        context = self._build_context(
            retrieval_result.documents,
            max_tokens=self.config["max_context_tokens"],
        )

        # At max depth, force a direct answer.
        if depth >= max_depth:
            answer = await self._direct_answer(question_text, context, state)
            state.add_step(
                thought=f"[depth={depth}] Max depth reached, forcing direct answer.",
                action="finish",
                action_input=question_text,
                observation=answer,
            )
            if state.memoization:
                state.memo[question_text.strip().lower()] = answer
            return answer

        # Ask the LLM to decide: DIRECT or DECOMPOSE.
        context_section = f"Context (retrieved documents):\n{context}" if context else ""
        prompt = self.decompose_prompt.format(
            question=question_text,
            context_section=context_section,
        )

        messages = [{"role": "user", "content": prompt}]
        response_text, tokens_used, cost = await self.llm.generate(messages)
        state.num_llm_calls += 1
        state.total_tokens += tokens_used
        state.total_cost += cost

        # Parse the response.
        decision = self._parse_decision(response_text)

        if decision["type"] == "direct":
            answer = decision["answer"]
            state.add_step(
                thought=f"[depth={depth}] Direct answer for: {question_text}",
                action="finish",
                action_input=question_text,
                observation=answer,
                tokens_used=tokens_used,
                cost_usd=cost,
            )
            if state.memoization:
                state.memo[question_text.strip().lower()] = answer
            return answer

        # DECOMPOSE path.
        sub_questions: list[str] = decision["sub_questions"]
        combine_instruction: str = decision["combine_instruction"]

        state.add_step(
            thought=f"[depth={depth}] Decomposing: {question_text}",
            action="decompose",
            action_input=question_text,
            observation=f"Sub-questions: {sub_questions}",
            tokens_used=tokens_used,
            cost_usd=cost,
        )

        # Recursively answer each sub-question.
        sub_answers: list[str] = []
        for sq in sub_questions:
            sub_answer = await self._recursive_answer(
                question_text=sq,
                corpus=corpus,
                depth=depth + 1,
                state=state,
            )
            sub_answers.append(sub_answer)

        # Combine the sub-answers.
        combined_answer = await self._combine_answers(
            question_text=question_text,
            sub_questions=sub_questions,
            sub_answers=sub_answers,
            combine_instruction=combine_instruction,
            state=state,
        )

        state.add_step(
            thought=f"[depth={depth}] Combined sub-answers for: {question_text}",
            action="recurse",
            action_input=combine_instruction,
            observation=combined_answer,
        )

        if state.memoization:
            state.memo[question_text.strip().lower()] = combined_answer

        return combined_answer

    async def _direct_answer(
        self,
        question_text: str,
        context: str,
        state: _RecursionState,
    ) -> str:
        """Generate a direct answer using retrieval context.

        Used when max depth is reached or the LLM decides the question is simple.
        """
        prompt = (
            "Answer the following question based on the provided context. "
            "Be concise and direct. Give only the answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question_text}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens_used, cost = await self.llm.generate(messages)
        state.num_llm_calls += 1
        state.total_tokens += tokens_used
        state.total_cost += cost
        return response_text.strip()

    async def _combine_answers(
        self,
        question_text: str,
        sub_questions: list[str],
        sub_answers: list[str],
        combine_instruction: str,
        state: _RecursionState,
    ) -> str:
        """Combine sub-answers into a final answer for the original question."""
        sub_qa_pairs = "\n".join(f"Q: {sq}\nA: {sa}" for sq, sa in zip(sub_questions, sub_answers))
        prompt = self.combine_prompt.format(
            question=question_text,
            sub_qa_pairs=sub_qa_pairs,
            combine_instruction=combine_instruction,
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens_used, cost = await self.llm.generate(messages)
        state.num_llm_calls += 1
        state.total_tokens += tokens_used
        state.total_cost += cost
        return response_text.strip()

    @staticmethod
    def _parse_decision(response_text: str) -> dict[str, Any]:
        """Parse the LLM's DIRECT/DECOMPOSE decision.

        Returns:
            dict with 'type' ('direct' or 'decompose') and associated fields.
        """
        text = response_text.strip()

        # Check for DIRECT answer.
        direct_match = re.search(r"DIRECT:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if direct_match and "DECOMPOSE:" not in text.upper():
            answer = direct_match.group(1).strip()
            # Clean up: if there's a COMBINE or SUB after, it's actually a decompose.
            if "SUB:" in answer.upper() or "COMBINE:" in answer.upper():
                pass  # Fall through to decompose parsing.
            else:
                return {"type": "direct", "answer": answer}

        # Check for DECOMPOSE.
        sub_questions: list[str] = []
        for match in re.finditer(r"-\s*SUB:\s*(.+)", text, re.IGNORECASE):
            sq = match.group(1).strip()
            if sq:
                sub_questions.append(sq)

        combine_match = re.search(r"COMBINE:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        combine_instruction = (
            combine_match.group(1).strip() if combine_match else "Combine the answers."
        )
        # Clean trailing sub-questions from combine instruction if regex was greedy.
        combine_instruction = re.split(r"\n-\s*SUB:", combine_instruction, flags=re.IGNORECASE)[0]
        combine_instruction = combine_instruction.strip()

        if sub_questions:
            return {
                "type": "decompose",
                "sub_questions": sub_questions,
                "combine_instruction": combine_instruction,
            }

        # Fallback: treat entire response as a direct answer.
        return {"type": "direct", "answer": text}


class _RecursionState:
    """Mutable state shared across the recursion tree.

    Tracks metrics (tokens, cost, call counts) and the reasoning chain
    across all recursive calls for a single top-level answer() invocation.
    """

    def __init__(self, memoization: bool = True):
        self.reasoning_chain: list[ReasoningStep] = []
        self.retrieved_docs: list[RetrievalResult] = []
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.num_llm_calls: int = 0
        self.num_retrieval_calls: int = 0
        self.memoization: bool = memoization
        self.memo: dict[str, str] = {}
        self._step_counter: int = 0

    def add_step(
        self,
        thought: str,
        action: str,
        action_input: str,
        observation: str,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Append a reasoning step to the chain."""
        self._step_counter += 1
        self.reasoning_chain.append(
            ReasoningStep(
                step_id=self._step_counter,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
            )
        )
