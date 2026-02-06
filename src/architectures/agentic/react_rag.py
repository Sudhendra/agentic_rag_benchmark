"""ReAct RAG architecture implementation."""

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


class ReActRAG(BaseRAG):
    """ReAct RAG: iterative reasoning with tool use."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: BaseRetriever,
        config: dict,
    ):
        super().__init__(llm_client, retriever, config)

        prompt_path = self.config.get("prompt_path", "prompts/react.txt")
        if Path(prompt_path).exists():
            self.prompt_template = self._load_prompt_template(prompt_path)
        else:
            self.prompt_template = (
                "Answer the question by reasoning step-by-step and using tools.\n\n"
                "Available tools:\n"
                "- search[query]: search for passages related to the query\n"
                "- lookup[term]: lookup a term in previously retrieved passages\n"
                "- finish[answer]: provide the final answer\n\n"
                "Question: {question}\n\n"
                "{scratchpad}\n\n"
                "Thought:\n"
                "Action:\n"
            )

    def get_name(self) -> str:
        return "react_rag"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.AGENTIC

    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        return {
            "top_k": (int, False, 5),
            "max_iterations": (int, False, 7),
            "max_context_tokens": (int, False, 4000),
            "prompt_path": (str, False, "prompts/react.txt"),
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

        scratchpad_entries: list[str] = []
        answer: str | None = None

        for step_id in range(1, self.config["max_iterations"] + 1):
            scratchpad = "\n".join(scratchpad_entries)
            prompt = self.prompt_template.format(
                question=question.text,
                scratchpad=scratchpad,
            )

            messages = [{"role": "user", "content": prompt}]
            response_text, tokens_used, cost = await self.llm.generate(
                messages,
                stop=["Observation:"],
            )
            num_llm_calls += 1
            total_tokens += tokens_used
            total_cost += cost

            thought, action, action_input = self._parse_response(response_text)
            if action is None:
                answer = response_text.strip()
                reasoning_chain.append(
                    ReasoningStep(
                        step_id=step_id,
                        thought=thought or "",
                        action="finish",
                        action_input=answer,
                        observation="Parsed as finish due to missing action.",
                        tokens_used=tokens_used,
                        cost_usd=cost,
                    )
                )
                break

            observation = ""
            if action == "search":
                retrieval_result = await self.retriever.retrieve(
                    query=action_input,
                    corpus=corpus,
                    top_k=self.config["top_k"],
                )
                num_retrieval_calls += 1
                retrieved_docs.append(retrieval_result)
                observation = self._build_context(
                    retrieval_result.documents,
                    max_tokens=self.config["max_context_tokens"],
                )
            elif action == "lookup":
                observation = self._lookup_term(action_input, retrieved_docs)
            elif action == "finish":
                answer = action_input.strip() if action_input else response_text.strip()
                observation = "Final answer provided."
            else:
                answer = response_text.strip()
                action = "finish"
                observation = "Unknown action; returning response as final answer."

            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought=thought or "",
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    tokens_used=tokens_used,
                    cost_usd=cost,
                )
            )

            scratchpad_entries.append(
                "\n".join(
                    [
                        f"Thought: {thought}",
                        f"Action: {action}[{action_input}]",
                        f"Observation: {observation}",
                    ]
                )
            )

            if action == "finish":
                break

        if not answer:
            answer = "No answer."

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            answer=answer,
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

    def _parse_response(self, response_text: str) -> tuple[str | None, str | None, str]:
        thought = None
        action = None
        action_input = ""

        for line in response_text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("thought:"):
                thought = stripped.split(":", 1)[1].strip()
            elif stripped.lower().startswith("action:"):
                action_text = stripped.split(":", 1)[1].strip()
                match = re.match(r"(\w+)\[(.*)\]", action_text)
                if match:
                    action = match.group(1).lower()
                    action_input = match.group(2).strip()
                else:
                    action = action_text.lower()
                    action_input = ""

        return thought, action, action_input

    def _lookup_term(self, term: str, retrieved_docs: list[RetrievalResult]) -> str:
        if not retrieved_docs:
            return "No retrieved documents available for lookup."

        normalized = term.lower().strip()
        matches: list[str] = []
        for result in retrieved_docs:
            for doc in result.documents:
                if normalized in doc.text.lower() or normalized in doc.title.lower():
                    matches.append(f"{doc.title}: {doc.text}")

        if not matches:
            return "No matches found in retrieved documents."

        return "\n".join(matches)
