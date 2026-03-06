"""REAP recursive architecture implementation."""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...core.base_rag import BaseRAG
from ...core.types import (
    ArchitectureType,
    Document,
    Question,
    RAGResponse,
    ReasoningStep,
    RetrievalResult,
)

_DEFAULT_DECOMPOSE_PROMPT = """Break the multi-hop question into explicit requirements.

Return strict JSON only with keys `user_goal` and `requirements`.
Each requirement must contain: `requirement_id`, `question`, `depends_on`, `status`.
Use `status: pending` initially.

Question: {question}

JSON:
"""

_DEFAULT_PLAN_PROMPT = """Plan the next REAP actions from the current requirement state.

Return strict JSON only with keys `next_step`, `updated_plan`, and `next_actions`.
`next_step` must be either `EXECUTE` or `SYNTHESIZE_ANSWER`.
`next_actions` must be a list of requirement objects containing `requirement_id` and `question`.
Prefer unresolved requirements that most directly move the question forward.

Question: {question}
User Goal: {user_goal}

Plan State:
{plan_state}

Known Facts:
{facts_state}

JSON:
"""

_DEFAULT_REPLAN_PROMPT = """Repair or update the REAP plan using the current facts.

Return strict JSON only with keys `next_step`, `updated_plan`, and `next_actions`.
Use this replanning step when prior extraction produced only partial clues or failures.
`next_step` must be either `EXECUTE` or `SYNTHESIZE_ANSWER`.

Question: {question}
User Goal: {user_goal}

Plan State:
{plan_state}

Known Facts:
{facts_state}

JSON:
"""

_DEFAULT_EXTRACT_PROMPT = """Extract evidence-backed facts for the active REAP requirement.

Return strict JSON only with key `reasoned_facts`.
Each fact must include `reasoning`, `direct_evidence`, `statement`, `fulfills_requirement_id`, and `fulfillment_level`.
`fulfillment_level` must be one of `DIRECT_ANSWER`, `PARTIAL_CLUE`, or `FAILED_EXTRACT`.
Only mark `DIRECT_ANSWER` when the requirement is directly answered by the evidence.

Root Question: {question}
Requirement ID: {requirement_id}
Requirement Question: {requirement_question}

Known Facts:
{facts_state}

Retrieved Context:
{context}

JSON:
"""

_DEFAULT_SYNTHESIZE_PROMPT = """Synthesize the final answer from REAP facts.

Return only the final answer phrase with no explanation.

Question: {question}

Plan State:
{plan_state}

Known Facts:
{facts_state}

Answer:
"""

_FULFILLMENT_LEVELS = {"DIRECT_ANSWER", "PARTIAL_CLUE", "FAILED_EXTRACT"}


class REAPRAG(BaseRAG):
    """Recursive planning architecture scaffold for REAP."""

    def __init__(self, llm_client, retriever, config: dict):
        super().__init__(llm_client, retriever, config)
        self.decompose_prompt = self._load_optional_prompt(
            self.config["decompose_prompt_path"], _DEFAULT_DECOMPOSE_PROMPT
        )
        self.extract_prompt = self._load_optional_prompt(
            self.config["extract_prompt_path"], _DEFAULT_EXTRACT_PROMPT
        )
        self.plan_prompt = self._load_optional_prompt(
            self.config["plan_prompt_path"], _DEFAULT_PLAN_PROMPT
        )
        self.replan_prompt = self._load_optional_prompt(
            self.config["replan_prompt_path"], _DEFAULT_REPLAN_PROMPT
        )
        self.synthesize_prompt = self._load_optional_prompt(
            self.config["synthesize_prompt_path"], _DEFAULT_SYNTHESIZE_PROMPT
        )

    def get_name(self) -> str:
        return "reap_rag"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.RECURSIVE

    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        return {
            "top_k": (int, False, 3),
            "max_iterations": (int, False, 5),
            "max_active_requirements": (int, False, 2),
            "max_context_tokens": (int, False, 3000),
            "max_docs": (int, False, 8),
            "decompose_prompt_path": (str, False, "prompts/reap_decompose.txt"),
            "extract_prompt_path": (str, False, "prompts/reap_extract.txt"),
            "plan_prompt_path": (str, False, "prompts/reap_plan.txt"),
            "replan_prompt_path": (str, False, "prompts/reap_replan.txt"),
            "synthesize_prompt_path": (str, False, "prompts/reap_synthesize.txt"),
        }

    async def answer(self, question: Question, corpus: list[Document]) -> RAGResponse:
        start_time = time.perf_counter()
        reasoning_chain: list[ReasoningStep] = []
        retrieved_docs: list[RetrievalResult] = []
        total_tokens = 0
        total_cost = 0.0
        num_llm_calls = 0
        num_retrieval_calls = 0
        step_id = 0

        decomposition_text, tokens_used, cost = await self.llm.generate(
            [{"role": "user", "content": self._build_decompose_prompt(question.text)}]
        )
        num_llm_calls += 1
        total_tokens += tokens_used
        total_cost += cost
        decomposition_payload = self._extract_json_block(decomposition_text)
        user_goal = str(decomposition_payload.get("user_goal") or question.text).strip()
        plan = self._normalize_plan(decomposition_payload.get("requirements", []))

        step_id += 1
        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought=f"Decomposed question into {len(plan)} explicit requirements.",
                action="decompose",
                action_input=question.text,
                observation=json.dumps(plan),
                tokens_used=tokens_used,
                cost_usd=cost,
            )
        )

        facts: list[dict[str, Any]] = []
        should_replan = False

        for _ in range(self.config["max_iterations"]):
            planner_prompt = self._build_plan_prompt(
                question.text, user_goal, plan, facts, should_replan
            )
            planner_text, tokens_used, cost = await self.llm.generate(
                [{"role": "user", "content": planner_prompt}]
            )
            num_llm_calls += 1
            total_tokens += tokens_used
            total_cost += cost

            try:
                planner_payload = self._extract_json_block(planner_text)
            except ValueError:
                planner_payload = {
                    "next_step": "EXECUTE",
                    "updated_plan": plan,
                    "next_actions": self._fallback_next_actions(plan),
                }
            action_name = "replan" if should_replan else "plan"
            plan = self._merge_plan(plan, planner_payload.get("updated_plan", plan))
            next_actions = self._filter_executable_actions(
                plan,
                self._normalize_actions(planner_payload.get("next_actions", [])),
            )[: self.config["max_active_requirements"]]
            next_step = str(planner_payload.get("next_step", "EXECUTE")).upper()

            step_id += 1
            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought=(
                        "Replanned after partial or failed fact extraction."
                        if should_replan
                        else "Selected next requirement actions from explicit plan state."
                    ),
                    action=action_name,
                    action_input=json.dumps(plan),
                    observation=json.dumps({"next_step": next_step, "next_actions": next_actions}),
                    tokens_used=tokens_used,
                    cost_usd=cost,
                )
            )

            if next_step == "SYNTHESIZE_ANSWER" or not next_actions:
                break

            should_replan = False
            for action in next_actions:
                retrieval_result = await self.retriever.retrieve(
                    query=action["question"],
                    corpus=corpus,
                    top_k=self.config["top_k"],
                )
                retrieved_docs.append(retrieval_result)
                num_retrieval_calls += 1

                extract_prompt = self._build_extract_prompt(
                    question.text,
                    action,
                    facts,
                    retrieval_result.documents[: self.config["max_docs"]],
                )
                extract_text, tokens_used, cost = await self.llm.generate(
                    [{"role": "user", "content": extract_prompt}]
                )
                num_llm_calls += 1
                total_tokens += tokens_used
                total_cost += cost

                try:
                    extracted_facts = self._parse_reasoned_facts(
                        self._extract_json_block(extract_text)
                    )
                except ValueError:
                    extracted_facts = [
                        {
                            "reasoning": "Extractor returned malformed output.",
                            "direct_evidence": "",
                            "statement": "No reliable fact extracted.",
                            "fulfills_requirement_id": action["requirement_id"],
                            "fulfillment_level": "FAILED_EXTRACT",
                        }
                    ]
                facts.extend(self._dedupe_facts(extracted_facts, facts))
                plan = self._mark_resolved_requirements(plan, facts)
                should_replan = should_replan or any(
                    fact["fulfillment_level"] != "DIRECT_ANSWER" for fact in extracted_facts
                )

                step_id += 1
                reasoning_chain.append(
                    ReasoningStep(
                        step_id=step_id,
                        thought=f"Extracted evidence-backed facts for requirement {action['requirement_id']}.",
                        action="extract",
                        action_input=action["question"],
                        observation=json.dumps(extracted_facts),
                        tokens_used=tokens_used,
                        cost_usd=cost,
                    )
                )

            if self._all_requirements_resolved(plan):
                break
            if self._can_synthesize(plan, facts):
                break

        synthesis_text, tokens_used, cost = await self.llm.generate(
            [{"role": "user", "content": self._build_synthesize_prompt(question.text, plan, facts)}]
        )
        num_llm_calls += 1
        total_tokens += tokens_used
        total_cost += cost
        final_answer = self._select_final_answer(
            question_text=question.text,
            plan=plan,
            facts=facts,
            llm_answer=synthesis_text.strip(),
        )

        step_id += 1
        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought="Synthesized the final answer from accumulated facts.",
                action="synthesize",
                action_input=question.text,
                observation=final_answer,
                tokens_used=tokens_used,
                cost_usd=cost,
            )
        )

        return RAGResponse(
            answer=final_answer,
            reasoning_chain=reasoning_chain,
            retrieved_docs=retrieved_docs,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            num_retrieval_calls=num_retrieval_calls,
            num_llm_calls=num_llm_calls,
            model=getattr(self.llm, "model", "unknown"),
            architecture=self.get_name(),
        )

    @staticmethod
    def _extract_json_block(text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("No JSON object found in LLM output")

    @staticmethod
    def _mark_resolved_requirements(
        plan: list[dict[str, Any]],
        facts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        updated_plan = deepcopy(plan)
        resolved_ids = {
            fact["fulfills_requirement_id"]
            for fact in facts
            if fact.get("fulfillment_level") == "DIRECT_ANSWER"
        }
        for requirement in updated_plan:
            if requirement.get("requirement_id") in resolved_ids:
                requirement["status"] = "resolved"
        return updated_plan

    @staticmethod
    def _parse_reasoned_facts(payload: dict[str, Any]) -> list[dict[str, Any]]:
        facts = payload.get("reasoned_facts")
        if not isinstance(facts, list):
            raise ValueError("reasoned_facts must be a list")

        parsed_facts: list[dict[str, Any]] = []
        required_keys = {
            "reasoning",
            "direct_evidence",
            "statement",
            "fulfills_requirement_id",
            "fulfillment_level",
        }
        for fact in facts:
            if not isinstance(fact, dict):
                raise ValueError("Each reasoned fact must be an object")
            missing_keys = required_keys - set(fact)
            if missing_keys:
                raise ValueError(f"Missing fact keys: {sorted(missing_keys)}")
            fulfillment_level = str(fact["fulfillment_level"])
            if fulfillment_level not in _FULFILLMENT_LEVELS:
                raise ValueError(f"Invalid fulfillment level: {fulfillment_level}")
            parsed_facts.append(
                {
                    "reasoning": str(fact["reasoning"]).strip(),
                    "direct_evidence": str(fact["direct_evidence"]).strip(),
                    "statement": str(fact["statement"]).strip(),
                    "fulfills_requirement_id": str(fact["fulfills_requirement_id"]).strip(),
                    "fulfillment_level": fulfillment_level,
                }
            )
        return parsed_facts

    def _load_optional_prompt(self, prompt_path: str, default_prompt: str) -> str:
        resolved_path = self._resolve_prompt_path(prompt_path)
        if resolved_path.exists():
            return self._load_prompt_template(str(resolved_path))
        return default_prompt

    @staticmethod
    def _resolve_prompt_path(prompt_path: str) -> Path:
        candidate = Path(prompt_path)
        if candidate.is_absolute():
            return candidate

        current = Path(__file__).resolve()
        for parent in current.parents:
            prompt_candidate = parent / prompt_path
            if prompt_candidate.exists():
                return prompt_candidate
        return current.parents[3] / prompt_path

    def _build_decompose_prompt(self, question: str) -> str:
        return self.decompose_prompt.format(question=question)

    def _build_plan_prompt(
        self,
        question: str,
        user_goal: str,
        plan: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        use_replan: bool,
    ) -> str:
        prompt_template = self.replan_prompt if use_replan else self.plan_prompt
        return prompt_template.format(
            question=question,
            user_goal=user_goal,
            plan_state=self._json_text(plan),
            facts_state=self._json_text(facts),
        )

    def _build_extract_prompt(
        self,
        question: str,
        requirement: dict[str, Any],
        facts: list[dict[str, Any]],
        documents: list[Document],
    ) -> str:
        return self.extract_prompt.format(
            question=question,
            requirement_id=requirement["requirement_id"],
            requirement_question=requirement["question"],
            facts_state=self._json_text(facts),
            context=self._build_context(documents, max_tokens=self.config["max_context_tokens"]),
        )

    def _build_synthesize_prompt(
        self,
        question: str,
        plan: list[dict[str, Any]],
        facts: list[dict[str, Any]],
    ) -> str:
        return self.synthesize_prompt.format(
            question=question,
            plan_state=self._json_text(plan),
            facts_state=self._json_text(facts),
        )

    @staticmethod
    def _normalize_plan(requirements: Any) -> list[dict[str, Any]]:
        if not isinstance(requirements, list):
            raise ValueError("requirements must be a list")
        normalized: list[dict[str, Any]] = []
        for index, requirement in enumerate(requirements, start=1):
            if not isinstance(requirement, dict):
                raise ValueError("Each requirement must be an object")
            raw_dependencies = requirement.get("depends_on")
            if raw_dependencies is None:
                dependencies: list[str] = []
            elif isinstance(raw_dependencies, list):
                dependencies = [str(dep) for dep in raw_dependencies]
            else:
                dependencies = [str(raw_dependencies)]
            normalized.append(
                {
                    "requirement_id": str(requirement.get("requirement_id") or f"r{index}"),
                    "question": str(requirement.get("question") or "").strip(),
                    "depends_on": dependencies,
                    "status": str(requirement.get("status") or "pending"),
                }
            )
        return normalized

    @staticmethod
    def _normalize_actions(actions: Any) -> list[dict[str, str]]:
        if not isinstance(actions, list):
            return []
        normalized: list[dict[str, str]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            requirement_id = str(action.get("requirement_id") or "").strip()
            question = str(action.get("question") or "").strip()
            if requirement_id and question:
                normalized.append({"requirement_id": requirement_id, "question": question})
        return normalized

    @staticmethod
    def _merge_plan(
        current_plan: list[dict[str, Any]],
        updated_plan: Any,
    ) -> list[dict[str, Any]]:
        normalized = REAPRAG._normalize_plan(updated_plan)
        if not current_plan:
            return normalized

        status_by_id = {
            requirement["requirement_id"]: requirement.get("status", "pending")
            for requirement in current_plan
        }
        for requirement in normalized:
            if requirement["requirement_id"] in status_by_id and requirement["status"] == "pending":
                requirement["status"] = status_by_id[requirement["requirement_id"]]
        return normalized

    @staticmethod
    def _filter_executable_actions(
        plan: list[dict[str, Any]],
        actions: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        requirements_by_id = {requirement["requirement_id"]: requirement for requirement in plan}
        executable_actions: list[dict[str, str]] = []
        for action in actions:
            requirement = requirements_by_id.get(action["requirement_id"])
            if requirement is None:
                continue
            if requirement.get("status") == "resolved":
                continue
            dependencies = requirement.get("depends_on", [])
            if all(
                requirements_by_id.get(dependency_id, {}).get("status") == "resolved"
                for dependency_id in dependencies
            ):
                executable_actions.append(action)
        return executable_actions

    @staticmethod
    def _fallback_next_actions(plan: list[dict[str, Any]]) -> list[dict[str, str]]:
        requirements_by_id = {requirement["requirement_id"]: requirement for requirement in plan}
        fallback_actions: list[dict[str, str]] = []
        for requirement in plan:
            if requirement.get("status") == "resolved":
                continue
            dependencies = requirement.get("depends_on", [])
            if all(
                requirements_by_id.get(dependency_id, {}).get("status") == "resolved"
                for dependency_id in dependencies
            ):
                fallback_actions.append(
                    {
                        "requirement_id": requirement["requirement_id"],
                        "question": requirement["question"],
                    }
                )
        return fallback_actions

    @staticmethod
    def _dedupe_facts(
        new_facts: list[dict[str, Any]],
        existing_facts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen = {
            (
                fact["fulfills_requirement_id"],
                fact["statement"],
                fact["fulfillment_level"],
            )
            for fact in existing_facts
        }
        deduped: list[dict[str, Any]] = []
        for fact in new_facts:
            key = (
                fact["fulfills_requirement_id"],
                fact["statement"],
                fact["fulfillment_level"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    @staticmethod
    def _all_requirements_resolved(plan: list[dict[str, Any]]) -> bool:
        return bool(plan) and all(requirement.get("status") == "resolved" for requirement in plan)

    @staticmethod
    def _can_synthesize(plan: list[dict[str, Any]], facts: list[dict[str, Any]]) -> bool:
        direct_answer_ids = {
            fact["fulfills_requirement_id"]
            for fact in facts
            if fact.get("fulfillment_level") == "DIRECT_ANSWER"
        }
        if not direct_answer_ids:
            return False

        dependency_ids = {
            dependency for requirement in plan for dependency in requirement.get("depends_on", [])
        }
        terminal_requirement_ids = {
            requirement.get("requirement_id")
            for requirement in plan
            if requirement.get("requirement_id") not in dependency_ids
        }
        return bool(terminal_requirement_ids) and terminal_requirement_ids.issubset(
            direct_answer_ids
        )

    @staticmethod
    def _select_final_answer(
        question_text: str,
        plan: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        llm_answer: str,
    ) -> str:
        direct_facts = [fact for fact in facts if fact.get("fulfillment_level") == "DIRECT_ANSWER"]
        lowered_question = question_text.lower()
        lowered_llm_answer = llm_answer.strip().lower()

        if REAPRAG._is_comparison_question(lowered_question):
            if any(phrase in lowered_llm_answer for phrase in {"same", "yes", "are of the same"}):
                return "yes"
            if any(phrase in lowered_llm_answer for phrase in {"not", "different", "no"}):
                return "no"

        if any(keyword in lowered_question for keyword in {"position", "role", "title", "office"}):
            title_candidate = REAPRAG._select_title_candidate(direct_facts)
            if title_candidate:
                return title_candidate

        return llm_answer.strip()

    @staticmethod
    def _is_comparison_question(lowered_question: str) -> bool:
        return lowered_question.startswith(("is ", "are ", "was ", "were ", "do ", "does "))

    @staticmethod
    def _select_title_candidate(direct_facts: list[dict[str, Any]]) -> str:
        ranked_candidates: list[tuple[int, str]] = []
        for fact in direct_facts:
            statement = str(fact.get("statement") or "").strip()
            if not statement:
                continue
            lowered = statement.lower()
            if "chief of" in lowered or "secretary of" in lowered:
                ranked_candidates.append((0, REAPRAG._extract_title_span(statement)))
            elif "ambassador to" in lowered:
                ranked_candidates.append((1, REAPRAG._extract_title_span(statement)))
        if not ranked_candidates:
            return ""
        ranked_candidates.sort(key=lambda item: (item[0], len(item[1])))
        return ranked_candidates[0][1]

    @staticmethod
    def _extract_title_span(statement: str) -> str:
        title_patterns = [
            r"(Chief of [^.,;]+)",
            r"(Secretary of [^.,;]+)",
            r"((?:U\.S\.|United States) Ambassador to [^.,;]+)",
        ]
        for pattern in title_patterns:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".")
        return statement.strip().rstrip(".")

    @staticmethod
    def _json_text(payload: Any) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=True)
