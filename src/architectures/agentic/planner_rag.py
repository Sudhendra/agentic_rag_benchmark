"""Planner RAG architecture implementation.

PlannerRAG uses an explicit planning loop inspired by TreePS-RAG action search
(Traverse/Select/Rollout/Backtrack/Stop) plus a RAGShaper-style planner/solver
split. This implementation focuses on inference-time planning and tracing.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

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

PlannerAction = Literal["TRAVERSE", "SELECT", "ROLLOUT", "BACKTRACK", "STOP"]
NodeStatus = Literal["open", "active", "solved", "pruned"]

_VALID_ACTIONS: set[str] = {"TRAVERSE", "SELECT", "ROLLOUT", "BACKTRACK", "STOP"}


@dataclass
class PlanNode:
    """Single node in the planner's reasoning tree."""

    node_id: str
    question: str
    parent_id: str | None
    depth: int
    status: NodeStatus = "open"
    answer: str = ""
    confidence: float = 0.0
    evidence_doc_ids: list[str] = field(default_factory=list)


class PlannerRAG(BaseRAG):
    """Planner-driven RAG with explicit tree actions and reasoning traces."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: BaseRetriever,
        config: dict,
    ):
        super().__init__(llm_client, retriever, config)

        self.planner_prompt_template = self._load_optional_prompt(
            self.config["planner_prompt_path"],
            self._default_planner_prompt(),
        )
        self.solver_prompt_template = self._load_optional_prompt(
            self.config["solver_prompt_path"],
            self._default_solver_prompt(),
        )
        self.synthesis_prompt_template = self._load_optional_prompt(
            self.config["synthesis_prompt_path"],
            self._default_synthesis_prompt(),
        )
        self.bridge_refine_prompt_template = self._load_optional_prompt(
            self.config["bridge_refine_prompt_path"],
            self._default_bridge_refine_prompt(),
        )

    def get_name(self) -> str:
        return "planner_rag"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.AGENTIC

    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        return {
            "top_k": (int, False, 3),
            "max_iterations": (int, False, 5),
            "max_branching_factor": (int, False, 2),
            "rollout_similarity_threshold": (float, False, 0.85),
            "max_depth": (int, False, 3),
            "min_stop_confidence": (float, False, 0.8),
            "allow_direct_answer": (bool, False, True),
            "max_context_tokens": (int, False, 4000),
            "planner_prompt_path": (str, False, "prompts/planner_action.txt"),
            "solver_prompt_path": (str, False, "prompts/planner_solve.txt"),
            "synthesis_prompt_path": (str, False, "prompts/planner_synthesize.txt"),
            "bridge_refine_enabled": (bool, False, True),
            "bridge_refine_max_attempts": (int, False, 1),
            "bridge_refine_prompt_path": (str, False, "prompts/planner_bridge_refine.txt"),
            "bridge_generic_answers": (
                list,
                False,
                ["yes", "no", "unknown", "none", "n/a"],
            ),
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
        child_solved_count = 0
        retrieval_cache: dict[tuple[str, int], RetrievalResult] = {}
        node_solve_cache: dict[str, dict[str, Any]] = {}

        nodes: dict[str, PlanNode] = {
            "root": PlanNode(
                node_id="root",
                question=question.text,
                parent_id=None,
                depth=0,
                status="open",
            )
        }
        active_node_id = "root"

        (
            direct_answer,
            gate_tokens,
            gate_cost,
            gate_raw,
            gate_called,
        ) = await self._decide_direct_answer(question)
        total_tokens += gate_tokens
        total_cost += gate_cost
        if gate_called:
            num_llm_calls += 1

        step_id += 1
        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought="Decide if direct answer is possible or recursive planning is needed.",
                action="gate",
                action_input=question.text,
                observation=f"direct_answer={direct_answer}; raw={gate_raw[:200]}",
                tokens_used=gate_tokens,
                cost_usd=gate_cost,
            )
        )

        if direct_answer:
            direct_text, direct_tokens, direct_cost = await self._direct_answer(question)
            direct_text = self._normalize_final_answer(question, direct_text)
            total_tokens += direct_tokens
            total_cost += direct_cost
            num_llm_calls += 1

            step_id += 1
            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought="Answer directly without recursive planning.",
                    action="direct_answer",
                    action_input=question.text,
                    observation=direct_text[:200],
                    tokens_used=direct_tokens,
                    cost_usd=direct_cost,
                )
            )

            return self._build_response(
                answer=direct_text,
                reasoning_chain=reasoning_chain,
                retrieved_docs=retrieved_docs,
                total_tokens=total_tokens,
                total_cost=total_cost,
                start_time=start_time,
                num_retrieval_calls=num_retrieval_calls,
                num_llm_calls=num_llm_calls,
            )

        for iteration in range(1, self.config["max_iterations"] + 1):
            root_node = nodes["root"]
            if (
                root_node.status == "solved"
                and (
                    question.type == QuestionType.COMPARISON
                    or root_node.confidence >= self.config["min_stop_confidence"]
                )
                and (
                    not self._is_bridge_like(question)
                    or child_solved_count > 0
                    or not self._has_children(nodes, "root")
                )
            ):
                if question.type == QuestionType.COMPARISON:
                    stop_observation = "comparison_root_solved_stop"
                else:
                    stop_observation = (
                        f"root_confidence={root_node.confidence:.2f} >= "
                        f"{self.config['min_stop_confidence']:.2f}"
                    )
                step_id += 1
                reasoning_chain.append(
                    ReasoningStep(
                        step_id=step_id,
                        thought="Root node confidence reached stopping threshold.",
                        action="STOP",
                        action_input="early_stop",
                        observation=stop_observation,
                    )
                )
                break

            open_nodes = self._open_node_ids(nodes)
            action_payload, action_tokens, action_cost, raw_action = await self._plan_action(
                question=question,
                nodes=nodes,
                active_node_id=active_node_id,
                open_node_ids=open_nodes,
                iteration=iteration,
            )
            total_tokens += action_tokens
            total_cost += action_cost
            num_llm_calls += 1

            action = action_payload["action"]
            target_node_id = action_payload.get("node_id")
            action_input = json.dumps(action_payload, ensure_ascii=True)
            iteration_tokens = action_tokens
            iteration_cost = action_cost
            observation_parts = [f"raw={raw_action[:120]}"]

            if (
                self._is_bridge_like(question)
                and iteration == 1
                and action in {"TRAVERSE", "BACKTRACK", "STOP"}
                and not self._has_children(nodes, "root")
            ):
                action = "ROLLOUT"
                target_node_id = "root"
                action_input = json.dumps(
                    {"action": "ROLLOUT", "node_id": "root"}, ensure_ascii=True
                )
                observation_parts.append("bridge_guardrail_forced_rollout_root")

            if action == "STOP":
                if self._can_stop(nodes):
                    if self._is_bridge_like(question) and nodes["root"].status != "solved":
                        action = "SELECT"
                        target_node_id = "root"
                        action_input = json.dumps(
                            {"action": "SELECT", "node_id": target_node_id}, ensure_ascii=True
                        )
                        observation_parts.append("bridge_guardrail_stop_overridden_to_root_select")
                    if (
                        self._is_bridge_like(question)
                        and action == "STOP"
                        and child_solved_count == 0
                        and self._has_children(nodes, "root")
                    ):
                        action = "SELECT"
                        target_node_id = self._default_open_node(nodes, prefer_non_root=True)
                        action_input = json.dumps(
                            {"action": "SELECT", "node_id": target_node_id}, ensure_ascii=True
                        )
                        observation_parts.append("bridge_guardrail_stop_overridden_to_child_select")
                    else:
                        observation_parts.append("planner requested stop")
                        step_id += 1
                        reasoning_chain.append(
                            ReasoningStep(
                                step_id=step_id,
                                thought="Planner selected action.",
                                action="STOP",
                                action_input=action_input,
                                observation="; ".join(observation_parts),
                                tokens_used=iteration_tokens,
                                cost_usd=iteration_cost,
                            )
                        )
                        break
                else:
                    action = "SELECT"
                    target_node_id = self._default_open_node(
                        nodes,
                        prefer_non_root=self._is_bridge_like(question),
                    )
                    action_input = json.dumps(
                        {"action": "SELECT", "node_id": target_node_id}, ensure_ascii=True
                    )
                    observation_parts.append("stop_overridden_to_select_due_to_unresolved_nodes")

            if action == "TRAVERSE":
                resolved_traverse_id = self._resolve_node_id(
                    target_node_id=target_node_id,
                    active_node_id=active_node_id,
                    nodes=nodes,
                )
                if resolved_traverse_id == active_node_id:
                    action = "SELECT"
                    target_node_id = self._default_open_node(
                        nodes,
                        prefer_non_root=self._is_bridge_like(question),
                    )
                    action_input = json.dumps(
                        {"action": "SELECT", "node_id": target_node_id}, ensure_ascii=True
                    )
                    observation_parts.append("traverse_noop_overridden_to_select")
                else:
                    target_node_id = resolved_traverse_id

            if action == "SELECT":
                resolved_node_id = self._resolve_node_id(
                    target_node_id=target_node_id,
                    active_node_id=active_node_id,
                    nodes=nodes,
                )
                if (
                    self._is_bridge_like(question)
                    and child_solved_count > 0
                    and nodes["root"].status != "solved"
                    and resolved_node_id != "root"
                ):
                    resolved_node_id = "root"
                    target_node_id = "root"
                    action_input = json.dumps(
                        {"action": "SELECT", "node_id": "root"},
                        ensure_ascii=True,
                    )
                    observation_parts.append("bridge_guardrail_redirect_select_to_root")
                active_node_id = resolved_node_id

                solve_result = await self._solve_node(
                    node=nodes[resolved_node_id],
                    root_question=question,
                    nodes=nodes,
                    corpus=corpus,
                    retrieval_cache=retrieval_cache,
                    node_solve_cache=node_solve_cache,
                )
                retrieved_docs.append(solve_result["retrieval_result"])
                num_retrieval_calls += solve_result["num_retrieval_calls"]
                num_llm_calls += solve_result["num_llm_calls"]
                total_tokens += solve_result["tokens"]
                total_cost += solve_result["cost"]
                iteration_tokens += solve_result["tokens"]
                iteration_cost += solve_result["cost"]

                if solve_result["status"] == "solved" and resolved_node_id == "root":
                    nodes["root"].status = "solved"
                if solve_result["status"] == "solved" and resolved_node_id != "root":
                    child_solved_count += 1

                observation_parts.append(
                    f"node={resolved_node_id}; status={solve_result['status']}; "
                    f"confidence={solve_result['confidence']:.2f}; "
                    f"retrieved={len(solve_result['retrieval_result'].documents)}"
                )

            elif action == "TRAVERSE":
                resolved_node_id = target_node_id or active_node_id
                active_node_id = resolved_node_id

                solve_result = await self._solve_node(
                    node=nodes[resolved_node_id],
                    root_question=question,
                    nodes=nodes,
                    corpus=corpus,
                    retrieval_cache=retrieval_cache,
                    node_solve_cache=node_solve_cache,
                )
                retrieved_docs.append(solve_result["retrieval_result"])
                num_retrieval_calls += solve_result["num_retrieval_calls"]
                num_llm_calls += solve_result["num_llm_calls"]
                total_tokens += solve_result["tokens"]
                total_cost += solve_result["cost"]
                iteration_tokens += solve_result["tokens"]
                iteration_cost += solve_result["cost"]

                if solve_result["status"] == "solved" and resolved_node_id == "root":
                    nodes["root"].status = "solved"
                if solve_result["status"] == "solved" and resolved_node_id != "root":
                    child_solved_count += 1

                observation_parts.append(
                    f"traversed_to={resolved_node_id}; status={solve_result['status']}; "
                    f"confidence={solve_result['confidence']:.2f}; "
                    f"retrieved={len(solve_result['retrieval_result'].documents)}"
                )

            elif action == "ROLLOUT":
                resolved_node_id = self._resolve_node_id(
                    target_node_id=target_node_id,
                    active_node_id=active_node_id,
                    nodes=nodes,
                )
                active_node_id = resolved_node_id

                created_ids, rollout_tokens, rollout_cost = await self._rollout_node(
                    node=nodes[resolved_node_id],
                    root_question=question,
                    nodes=nodes,
                )
                if rollout_tokens > 0 or rollout_cost > 0:
                    num_llm_calls += 1
                total_tokens += rollout_tokens
                total_cost += rollout_cost
                iteration_tokens += rollout_tokens
                iteration_cost += rollout_cost
                observation_parts.append(
                    f"node={resolved_node_id}; created_children={len(created_ids)}"
                )

            elif action == "BACKTRACK":
                resolved_node_id = self._resolve_node_id(
                    target_node_id=target_node_id,
                    active_node_id=active_node_id,
                    nodes=nodes,
                )
                if resolved_node_id != "root":
                    nodes[resolved_node_id].status = "pruned"
                parent_id = nodes[resolved_node_id].parent_id or "root"
                active_node_id = parent_id
                observation_parts.append(f"pruned={resolved_node_id}; new_active={active_node_id}")

            else:
                # Safety fallback.
                active_node_id = "root"
                solve_result = await self._solve_node(
                    node=nodes["root"],
                    root_question=question,
                    nodes=nodes,
                    corpus=corpus,
                    retrieval_cache=retrieval_cache,
                    node_solve_cache=node_solve_cache,
                )
                retrieved_docs.append(solve_result["retrieval_result"])
                num_retrieval_calls += solve_result["num_retrieval_calls"]
                num_llm_calls += solve_result["num_llm_calls"]
                total_tokens += solve_result["tokens"]
                total_cost += solve_result["cost"]
                iteration_tokens += solve_result["tokens"]
                iteration_cost += solve_result["cost"]
                observation_parts.append(
                    f"fallback_select_root; confidence={solve_result['confidence']:.2f}"
                )

            step_id += 1
            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought="Planner selected action.",
                    action=action,
                    action_input=action_input,
                    observation="; ".join(observation_parts),
                    tokens_used=iteration_tokens,
                    cost_usd=iteration_cost,
                )
            )

        if num_retrieval_calls == 0:
            forced_node_id = self._default_open_node(
                nodes,
                prefer_non_root=self._is_bridge_like(question),
            )
            forced_solve = await self._solve_node(
                node=nodes[forced_node_id],
                root_question=question,
                nodes=nodes,
                corpus=corpus,
                retrieval_cache=retrieval_cache,
                node_solve_cache=node_solve_cache,
            )
            retrieved_docs.append(forced_solve["retrieval_result"])
            num_retrieval_calls += forced_solve["num_retrieval_calls"]
            num_llm_calls += forced_solve["num_llm_calls"]
            total_tokens += forced_solve["tokens"]
            total_cost += forced_solve["cost"]

            step_id += 1
            reasoning_chain.append(
                ReasoningStep(
                    step_id=step_id,
                    thought="No retrieval-backed node was solved during planning; forcing one solve step.",
                    action="SELECT",
                    action_input=json.dumps({"node_id": forced_node_id}, ensure_ascii=True),
                    observation=(
                        f"forced_select_node={forced_node_id}; "
                        f"confidence={forced_solve['confidence']:.2f}; "
                        f"retrieved={len(forced_solve['retrieval_result'].documents)}"
                    ),
                    tokens_used=forced_solve["tokens"],
                    cost_usd=forced_solve["cost"],
                )
            )

        answer_text, synth_tokens, synth_cost = await self._synthesize_answer(question, nodes)
        total_tokens += synth_tokens
        total_cost += synth_cost
        num_llm_calls += 1

        step_id += 1
        reasoning_chain.append(
            ReasoningStep(
                step_id=step_id,
                thought="Synthesize final answer from solved plan nodes.",
                action="synthesize",
                action_input=question.text,
                observation=answer_text[:200],
                tokens_used=synth_tokens,
                cost_usd=synth_cost,
            )
        )

        answer_text = self._recover_unknown_answer(question, nodes, answer_text)
        bridge_candidates = self._collect_bridge_candidates(nodes)
        candidate_answers = [candidate.answer for candidate in bridge_candidates]

        if self._is_bridge_like(question):
            max_refine_attempts = max(0, int(self.config["bridge_refine_max_attempts"]))
            if self.config["bridge_refine_enabled"] and max_refine_attempts > 0:
                for attempt in range(1, max_refine_attempts + 1):
                    if not (
                        self._is_invalid_bridge_answer(answer_text)
                        or self._is_partial_span_bridge_answer(answer_text, candidate_answers)
                    ):
                        break

                    answer_before_refine = answer_text
                    (
                        refined_answer,
                        refine_tokens,
                        refine_cost,
                        raw_refine,
                    ) = await self._refine_bridge_answer(
                        question=question,
                        nodes=nodes,
                        initial_answer=answer_before_refine,
                    )
                    answer_text = refined_answer
                    total_tokens += refine_tokens
                    total_cost += refine_cost
                    num_llm_calls += 1

                    step_id += 1
                    reasoning_chain.append(
                        ReasoningStep(
                            step_id=step_id,
                            thought="Refine bridge/compositional answer to a specific entity/span.",
                            action="bridge_refine",
                            action_input=answer_before_refine,
                            observation=(
                                f"attempt={attempt}/{max_refine_attempts}; "
                                f"raw={raw_refine[:160]}; refined={refined_answer[:120]}"
                            ),
                            tokens_used=refine_tokens,
                            cost_usd=refine_cost,
                        )
                    )

            if self._is_invalid_bridge_answer(answer_text) or self._is_partial_span_bridge_answer(
                answer_text,
                candidate_answers,
            ):
                fallback_answer = self._select_bridge_fallback(answer_text, bridge_candidates)
                if fallback_answer != answer_text:
                    answer_text = fallback_answer
                    step_id += 1
                    reasoning_chain.append(
                        ReasoningStep(
                            step_id=step_id,
                            thought="Apply deterministic bridge fallback to avoid generic answer.",
                            action="bridge_refine",
                            action_input=question.text,
                            observation=f"fallback_answer={fallback_answer[:160]}",
                        )
                    )

        answer_text = self._normalize_final_answer(question, answer_text)

        return self._build_response(
            answer=answer_text,
            reasoning_chain=reasoning_chain,
            retrieved_docs=retrieved_docs,
            total_tokens=total_tokens,
            total_cost=total_cost,
            start_time=start_time,
            num_retrieval_calls=num_retrieval_calls,
            num_llm_calls=num_llm_calls,
        )

    def _build_response(
        self,
        answer: str,
        reasoning_chain: list[ReasoningStep],
        retrieved_docs: list[RetrievalResult],
        total_tokens: int,
        total_cost: float,
        start_time: float,
        num_retrieval_calls: int,
        num_llm_calls: int,
    ) -> RAGResponse:
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

    async def _decide_direct_answer(
        self,
        question: Question,
    ) -> tuple[bool, int, float, str, bool]:
        if not self.config["allow_direct_answer"]:
            return False, 0, 0.0, "direct answers disabled by config", False

        if question.type in {
            QuestionType.BRIDGE,
            QuestionType.COMPARISON,
            QuestionType.COMPOSITIONAL,
        }:
            return False, 0, 0.0, "forced_recursive_for_question_type", False

        prompt = (
            "Decide if the question needs recursive retrieval planning.\n"
            "Return strict JSON with key direct_answer.\n"
            'Format: {"direct_answer": true|false, "reason": "..."}\n\n'
            f"Question: {question.text}"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)

        normalized = response_text.strip()
        parsed = self._try_parse_json(normalized)
        if isinstance(parsed, dict) and "direct_answer" in parsed:
            return bool(parsed["direct_answer"]), tokens, cost, normalized, True

        lowered = normalized.lower()
        direct = "true" in lowered or "direct" in lowered
        if "false" in lowered or "plan" in lowered or "retrieve" in lowered:
            direct = False

        return direct, tokens, cost, normalized, True

    async def _direct_answer(self, question: Question) -> tuple[str, int, float]:
        prompt = (
            "Answer the question directly and concisely.\n"
            "Return only the final answer phrase.\n\n"
            f"Question: {question.text}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)
        return response_text.strip(), tokens, cost

    async def _plan_action(
        self,
        question: Question,
        nodes: dict[str, PlanNode],
        active_node_id: str,
        open_node_ids: list[str],
        iteration: int,
    ) -> tuple[dict[str, str], int, float, str]:
        tree_state = self._format_tree_state(nodes)
        prompt = self.planner_prompt_template.format(
            question=question.text,
            question_type=question.type.value,
            tree_state=tree_state,
            active_node_id=active_node_id,
            open_nodes=", ".join(open_node_ids) if open_node_ids else "none",
            iteration=iteration,
            max_iterations=self.config["max_iterations"],
            max_depth=self.config["max_depth"],
            max_branching_factor=self.config["max_branching_factor"],
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)

        action_payload = self._parse_action_response(
            response_text=response_text,
            open_node_ids=open_node_ids,
            iteration=iteration,
            max_iterations=self.config["max_iterations"],
        )
        return action_payload, tokens, cost, response_text.strip()

    async def _solve_node(
        self,
        node: PlanNode,
        root_question: Question,
        nodes: dict[str, PlanNode],
        corpus: list[Document],
        retrieval_cache: dict[tuple[str, int], RetrievalResult],
        node_solve_cache: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        query = self._build_retrieval_query(node, nodes)
        top_k = self.config["top_k"]
        if self._is_bridge_like(root_question) and node.node_id == "root":
            top_k = min(top_k + 2, 10)

        cached_result = node_solve_cache.get(node.node_id)
        if (
            cached_result
            and cached_result.get("status") == "solved"
            and cached_result.get("query") == query
            and cached_result.get("top_k") == top_k
        ):
            return {
                "status": cached_result["status"],
                "confidence": cached_result["confidence"],
                "retrieval_result": cached_result["retrieval_result"],
                "num_llm_calls": 0,
                "num_retrieval_calls": 0,
                "tokens": 0,
                "cost": 0.0,
            }

        retrieval_key = (query, top_k)
        if retrieval_key in retrieval_cache:
            retrieval_result = retrieval_cache[retrieval_key]
            retrieval_calls = 0
        else:
            retrieval_result = await self.retriever.retrieve(
                query=query,
                corpus=corpus,
                top_k=top_k,
            )
            retrieval_cache[retrieval_key] = retrieval_result
            retrieval_calls = 1

        context = self._build_context(
            retrieval_result.documents,
            max_tokens=self.config["max_context_tokens"],
        )

        solve_prompt = self.solver_prompt_template.format(
            root_question=root_question.text,
            node_id=node.node_id,
            node_question=node.question,
            context=context,
        )
        if root_question.type == QuestionType.COMPARISON and node.node_id == "root":
            solve_prompt += (
                "\n\nFor this comparison root node, set JSON answer to exactly yes or no."
            )
        solve_messages = [{"role": "user", "content": solve_prompt}]
        answer_text, solve_tokens, solve_cost = await self.llm.generate(solve_messages)
        node_answer = answer_text.strip()
        confidence = 0.5
        conf_tokens = 0
        conf_cost = 0.0
        solve_llm_calls = 1
        used_combined_solver_output = False

        parsed_solver_output = self._try_parse_json(node_answer)
        if (
            isinstance(parsed_solver_output, dict)
            and "answer" in parsed_solver_output
            and "confidence" in parsed_solver_output
        ):
            strict_confidence = self._parse_confidence_strict(parsed_solver_output["confidence"])
            if strict_confidence is not None:
                node_answer = str(parsed_solver_output["answer"]).strip()
                confidence = strict_confidence
                used_combined_solver_output = True

        if not used_combined_solver_output:
            confidence_prompt = (
                "Estimate confidence that the node answer is correct using the provided context.\n"
                "Return a single float between 0.0 and 1.0.\n\n"
                f"Root question: {root_question.text}\n"
                f"Node question: {node.question}\n"
                f"Candidate answer: {node_answer}\n"
                f"Context: {context[:1200]}\n\n"
                "Confidence:"
            )
            confidence_messages = [{"role": "user", "content": confidence_prompt}]
            confidence_text, conf_tokens, conf_cost = await self.llm.generate(confidence_messages)
            confidence = self._parse_confidence(confidence_text)
            solve_llm_calls = 2

        node.answer = node_answer
        node.confidence = confidence
        node.evidence_doc_ids = [doc.id for doc in retrieval_result.documents]

        if (
            (root_question.type == QuestionType.COMPARISON and node.node_id == "root")
            or node.depth >= self.config["max_depth"]
            or confidence >= self.config["min_stop_confidence"]
        ):
            node.status = "solved"
        else:
            node.status = "active"

        if node.status == "solved":
            node_solve_cache[node.node_id] = {
                "status": node.status,
                "confidence": node.confidence,
                "retrieval_result": retrieval_result,
                "query": query,
                "top_k": top_k,
            }

        return {
            "status": node.status,
            "confidence": node.confidence,
            "retrieval_result": retrieval_result,
            "num_llm_calls": solve_llm_calls,
            "num_retrieval_calls": retrieval_calls,
            "tokens": solve_tokens + conf_tokens,
            "cost": solve_cost + conf_cost,
        }

    async def _rollout_node(
        self,
        node: PlanNode,
        root_question: Question,
        nodes: dict[str, PlanNode],
    ) -> tuple[list[str], int, float]:
        if node.depth >= self.config["max_depth"]:
            return [], 0, 0.0

        prompt = (
            "Expand the current node into useful sub-questions.\n"
            'Return strict JSON: {"sub_questions": ["...", "..."]}\n'
            f"Generate at most {self.config['max_branching_factor'] * 2} candidates.\n\n"
            f"Root question: {root_question.text}\n"
            f"Node question: {node.question}\n"
            f"Node answer so far: {node.answer or 'N/A'}\n"
            f"Current depth: {node.depth}\n"
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)

        candidates = self._parse_sub_questions(response_text)
        candidates = self._prune_similar_sub_questions(
            candidates,
            threshold=self.config["rollout_similarity_threshold"],
        )
        if not candidates:
            return [], tokens, cost

        max_children = self.config["max_branching_factor"]
        created_ids: list[str] = []
        for sub_question in candidates[:max_children]:
            normalized = sub_question.strip()
            if not normalized:
                continue
            child_id = self._next_child_id(parent_id=node.node_id, nodes=nodes)
            nodes[child_id] = PlanNode(
                node_id=child_id,
                question=normalized,
                parent_id=node.node_id,
                depth=node.depth + 1,
                status="open",
            )
            created_ids.append(child_id)

        if created_ids and node.status != "solved":
            node.status = "active"

        return created_ids, tokens, cost

    async def _synthesize_answer(
        self,
        question: Question,
        nodes: dict[str, PlanNode],
    ) -> tuple[str, int, float]:
        solved_nodes = [
            node for node in nodes.values() if node.answer and node.status in {"solved", "active"}
        ]
        solved_nodes.sort(key=lambda item: (item.depth, item.node_id))

        if solved_nodes:
            non_unknown_nodes = [
                node for node in solved_nodes if node.answer.strip().lower() != "unknown"
            ]
            if non_unknown_nodes:
                solved_nodes = non_unknown_nodes

            node_summaries = []
            for node in solved_nodes:
                evidence_preview = (
                    ", ".join(node.evidence_doc_ids[:3]) if node.evidence_doc_ids else "none"
                )
                node_summaries.append(
                    f"- {node.node_id} | q={node.question} | a={node.answer} | "
                    f"conf={node.confidence:.2f} | evidence={evidence_preview}"
                )
            summary_text = "\n".join(node_summaries)
            prompt = self.synthesis_prompt_template.format(
                question=question.text,
                question_type=question.type.value,
                node_summaries=summary_text,
            )
        else:
            prompt = (
                "No solved planner nodes are available."
                " Answer the question directly as best as possible.\n\n"
                f"Question: {question.text}\n\n"
                "Answer:"
            )

        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)
        return response_text.strip(), tokens, cost

    async def _refine_bridge_answer(
        self,
        question: Question,
        nodes: dict[str, PlanNode],
        initial_answer: str,
    ) -> tuple[str, int, float, str]:
        candidates = self._collect_bridge_candidates(nodes)
        if candidates:
            summaries = []
            for node in candidates[:5]:
                evidence_preview = (
                    ", ".join(node.evidence_doc_ids[:3]) if node.evidence_doc_ids else "none"
                )
                summaries.append(
                    f"- {node.node_id} | q={node.question} | a={node.answer} | "
                    f"conf={node.confidence:.2f} | evidence={evidence_preview}"
                )
            node_summaries = "\n".join(summaries)
        else:
            node_summaries = "- none"

        prompt = self.bridge_refine_prompt_template.format(
            question=question.text,
            question_type=question.type.value,
            initial_answer=initial_answer,
            node_summaries=node_summaries,
        )
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens, cost = await self.llm.generate(messages)
        return response_text.strip(), tokens, cost, response_text.strip()

    def _recover_unknown_answer(
        self,
        question: Question,
        nodes: dict[str, PlanNode],
        answer_text: str,
    ) -> str:
        normalized = answer_text.strip()
        if normalized.lower() != "unknown":
            return normalized
        if not self._is_bridge_like(question):
            return normalized

        candidates = self._collect_bridge_candidates(nodes)
        return self._select_bridge_fallback(normalized, candidates)

    def _normalize_final_answer(self, question: Question, answer_text: str) -> str:
        normalized = answer_text.strip()
        if question.type == QuestionType.COMPARISON:
            lowered = normalized.lower()
            if re.search(r"\byes\b", lowered):
                return "yes"
            if re.search(r"\bno\b", lowered):
                return "no"
        return normalized

    def _is_bridge_like(self, question: Question) -> bool:
        return question.type in {QuestionType.BRIDGE, QuestionType.COMPOSITIONAL}

    def _is_generic_bridge_answer(self, answer: str) -> bool:
        normalized = answer.strip().lower()
        if not normalized:
            return True

        generic_answers = {
            str(item).strip().lower()
            for item in self.config["bridge_generic_answers"]
            if str(item).strip()
        }
        compact = re.sub(r"[^a-z0-9/ ]", "", normalized).strip()
        if normalized in generic_answers or compact in generic_answers:
            return True

        return bool(re.fullmatch(r"(yes|no|unknown|none|n/?a)", compact))

    def _is_invalid_bridge_answer(self, answer: str) -> bool:
        normalized = answer.strip()
        if not normalized:
            return True
        if self._is_generic_bridge_answer(normalized):
            return True

        lowered = normalized.lower()
        weak_patterns = [
            "not enough information",
            "cannot determine",
            "can't determine",
            "insufficient context",
            "not provided",
        ]
        if any(pattern in lowered for pattern in weak_patterns):
            return True
        return False

    def _is_partial_span_bridge_answer(self, answer: str, candidates: list[str]) -> bool:
        base = answer.strip()
        if self._is_invalid_bridge_answer(base):
            return False

        base_tokens = re.findall(r"[A-Za-z0-9]+", base)
        if len(base_tokens) < 2:
            return False

        base_lower = base.lower()
        for candidate in candidates:
            candidate_text = candidate.strip()
            if not candidate_text:
                continue
            if candidate_text.lower() == base_lower:
                continue
            candidate_tokens = re.findall(r"[A-Za-z0-9]+", candidate_text)
            if (
                base_lower in candidate_text.lower()
                and len(candidate_text) > len(base)
                and (
                    "," in candidate_text
                    or "(" in candidate_text
                    or len(candidate_tokens) >= len(base_tokens) + 1
                )
            ):
                return True
        return False

    def _collect_bridge_candidates(self, nodes: dict[str, PlanNode]) -> list[PlanNode]:
        candidates = [
            node
            for node in nodes.values()
            if node.answer
            and node.status in {"solved", "active"}
            and not self._is_invalid_bridge_answer(node.answer)
        ]
        candidates.sort(
            key=lambda node: (
                node.node_id == "root",
                node.confidence,
                len(node.answer.strip()),
                node.depth,
            ),
            reverse=True,
        )
        return candidates

    def _select_bridge_fallback(self, answer_text: str, candidates: list[PlanNode]) -> str:
        normalized = answer_text.strip()
        if not candidates:
            return normalized

        if normalized:
            containing_candidates = [
                candidate
                for candidate in candidates
                if normalized.lower() in candidate.answer.strip().lower()
                and len(candidate.answer.strip()) > len(normalized)
            ]
            if containing_candidates:
                containing_candidates.sort(
                    key=lambda node: (len(node.answer.strip()), node.confidence, node.depth),
                    reverse=True,
                )
                return containing_candidates[0].answer.strip()

        return candidates[0].answer.strip()

    def _can_stop(self, nodes: dict[str, PlanNode]) -> bool:
        root = nodes.get("root")
        if (
            root
            and root.status == "solved"
            and root.confidence >= self.config["min_stop_confidence"]
        ):
            return True
        open_non_root_nodes = [
            node
            for node in nodes.values()
            if node.node_id != "root"
            and node.status in {"open", "active"}
            and node.depth <= self.config["max_depth"]
        ]
        solved_non_root_nodes = [
            node
            for node in nodes.values()
            if node.node_id != "root" and node.status == "solved" and bool(node.answer.strip())
        ]
        if solved_non_root_nodes and not open_non_root_nodes:
            return True
        return len(self._open_node_ids(nodes)) == 0

    def _default_open_node(self, nodes: dict[str, PlanNode], prefer_non_root: bool = False) -> str:
        candidates = [
            nodes[node_id]
            for node_id in self._open_node_ids(nodes)
            if nodes[node_id].status != "pruned"
        ]
        if not candidates:
            return "root"

        if prefer_non_root:
            non_root = [node for node in candidates if node.node_id != "root"]
            if non_root:
                non_root.sort(key=lambda node: (node.depth, node.node_id))
                return non_root[0].node_id

        candidates.sort(key=lambda node: (node.depth, node.node_id))
        return candidates[0].node_id

    def _has_children(self, nodes: dict[str, PlanNode], parent_id: str) -> bool:
        return any(node.parent_id == parent_id for node in nodes.values())

    def _open_node_ids(self, nodes: dict[str, PlanNode]) -> list[str]:
        return [
            node_id
            for node_id, node in nodes.items()
            if node.status in {"open", "active"} and node.depth <= self.config["max_depth"]
        ]

    def _resolve_node_id(
        self,
        target_node_id: str | None,
        active_node_id: str,
        nodes: dict[str, PlanNode],
    ) -> str:
        if target_node_id and target_node_id in nodes:
            return target_node_id
        if active_node_id in nodes:
            return active_node_id
        if "root" in nodes:
            return "root"
        return next(iter(nodes.keys()))

    def _build_retrieval_query(self, node: PlanNode, nodes: dict[str, PlanNode]) -> str:
        lineage: list[str] = []
        cursor = node.parent_id
        while cursor:
            parent = nodes.get(cursor)
            if not parent:
                break
            if parent.answer and parent.answer.strip().lower() != "unknown":
                lineage.append(parent.answer)
            cursor = parent.parent_id

        root_node = nodes.get("root")
        root_question = root_node.question if root_node else ""
        parts = [node.question]
        if root_question and root_question != node.question:
            parts.append(root_question)
        if lineage:
            parts.append(" ".join(reversed(lineage)))

        return " ".join(parts).strip()

    def _format_tree_state(self, nodes: dict[str, PlanNode]) -> str:
        lines = []
        for node_id, node in sorted(nodes.items(), key=lambda item: item[0]):
            answer_preview = node.answer[:80] if node.answer else ""
            lines.append(
                f"{node_id} | parent={node.parent_id or 'none'} | depth={node.depth} | "
                f"status={node.status} | conf={node.confidence:.2f} | q={node.question} | a={answer_preview}"
            )
        return "\n".join(lines)

    def _parse_action_response(
        self,
        response_text: str,
        open_node_ids: list[str],
        iteration: int,
        max_iterations: int,
    ) -> dict[str, str]:
        parsed = self._try_parse_json(response_text)
        if isinstance(parsed, dict):
            action_value = str(parsed.get("action", "")).upper().strip()
            node_value = parsed.get("node_id")
            if action_value in _VALID_ACTIONS:
                payload: dict[str, str] = {"action": action_value}
                if node_value is not None:
                    payload["node_id"] = str(node_value)
                return payload

        repaired = self._repair_action_from_text(response_text)
        if repaired:
            return repaired

        if iteration >= max_iterations:
            return {"action": "STOP"}

        if open_node_ids:
            return {"action": "SELECT", "node_id": open_node_ids[0]}

        return {"action": "SELECT", "node_id": "root"}

    def _repair_action_from_text(self, response_text: str) -> dict[str, str] | None:
        lowered = response_text.lower()
        action: str | None = None
        if "backtrack" in lowered:
            action = "BACKTRACK"
        elif "rollout" in lowered or "expand" in lowered:
            action = "ROLLOUT"
        elif "traverse" in lowered:
            action = "TRAVERSE"
        elif "select" in lowered or "solve" in lowered:
            action = "SELECT"
        elif "stop" in lowered or "finish" in lowered:
            action = "STOP"

        if not action:
            return None

        node_match = re.search(r"(?:node_id|node)\s*[:=]\s*([\w\.-]+)", lowered)
        if node_match:
            return {"action": action, "node_id": node_match.group(1)}
        return {"action": action}

    def _parse_sub_questions(self, response_text: str) -> list[str]:
        parsed = self._try_parse_json(response_text)
        if isinstance(parsed, dict):
            items = parsed.get("sub_questions")
            if isinstance(items, list):
                return [str(item).strip() for item in items if str(item).strip()]

        lines = []
        for line in response_text.splitlines():
            stripped = line.strip().lstrip("-*").strip()
            if stripped and len(stripped) > 5:
                lines.append(stripped)
        return lines

    def _normalize_sub_question(self, question: str) -> str:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        return re.sub(r"[^a-z0-9 ]", "", normalized).strip()

    def _token_jaccard_similarity(self, first: str, second: str) -> float:
        first_tokens = set(first.split())
        second_tokens = set(second.split())
        if not first_tokens and not second_tokens:
            return 1.0
        if not first_tokens or not second_tokens:
            return 0.0
        intersection = first_tokens & second_tokens
        union = first_tokens | second_tokens
        return len(intersection) / len(union)

    def _prune_similar_sub_questions(self, candidates: list[str], threshold: float) -> list[str]:
        kept: list[str] = []
        kept_normalized: list[str] = []

        for candidate in candidates:
            cleaned = candidate.strip()
            if not cleaned:
                continue
            normalized = self._normalize_sub_question(cleaned)
            if not normalized:
                continue

            is_redundant = False
            for existing in kept_normalized:
                if normalized == existing:
                    is_redundant = True
                    break

                similarity = SequenceMatcher(None, normalized, existing).ratio()
                token_similarity = self._token_jaccard_similarity(normalized, existing)
                if similarity >= threshold and token_similarity >= 0.5:
                    is_redundant = True
                    break
            if is_redundant:
                continue

            kept.append(cleaned)
            kept_normalized.append(normalized)

        return kept

    def _parse_confidence(self, response_text: str) -> float:
        match = re.search(r"([01](?:\.\d+)?)", response_text)
        if not match:
            return 0.5
        value = float(match.group(1))
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _parse_confidence_strict(self, confidence_value: Any) -> float | None:
        text = str(confidence_value).strip()
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return None
        value = float(text)
        if 0.0 <= value <= 1.0:
            return value
        return None

    def _next_child_id(self, parent_id: str, nodes: dict[str, PlanNode]) -> str:
        index = 1
        while True:
            candidate = f"{parent_id}.{index}"
            if candidate not in nodes:
                return candidate
            index += 1

    def _try_parse_json(self, response_text: str) -> Any:
        cleaned = response_text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _load_optional_prompt(self, prompt_path: str, default_prompt: str) -> str:
        if Path(prompt_path).exists():
            return self._load_prompt_template(prompt_path)
        return default_prompt

    def _default_planner_prompt(self) -> str:
        return (
            "You are a planning controller for multi-hop QA.\n"
            "Choose exactly one next action from: TRAVERSE, SELECT, ROLLOUT, BACKTRACK, STOP.\n"
            "Do not choose STOP unless root is solved with high confidence or no open nodes remain.\n"
            'Return strict JSON only: {{"action": "...", "node_id": "optional"}}.\n\n'
            "Root Question: {question}\n"
            "Question Type: {question_type}\n"
            "Iteration: {iteration}/{max_iterations}\n"
            "Active Node: {active_node_id}\n"
            "Open Nodes: {open_nodes}\n"
            "Max Depth: {max_depth}\n"
            "Max Branching Factor: {max_branching_factor}\n\n"
            "Tree State:\n{tree_state}\n"
        )

    def _default_solver_prompt(self) -> str:
        return (
            "Answer the node question using only the provided context.\n"
            "If context is clearly insufficient, answer exactly: unknown.\n"
            "For bridge/compositional-style questions, return a specific entity/span phrase "
            "and avoid yes/no unless the node question is explicitly yes/no.\n"
            "Return strict JSON only with keys answer and confidence.\n"
            'Format: {{"answer": "...", "confidence": 0.0}}\n'
            "Confidence must be a float in [0.0, 1.0].\n\n"
            "Root Question: {root_question}\n"
            "Node ID: {node_id}\n"
            "Node Question: {node_question}\n\n"
            "Context:\n{context}\n\n"
            "JSON:"
        )

    def _default_synthesis_prompt(self) -> str:
        return (
            "Synthesize a final answer to the root question using solved planner nodes.\n"
            "Be concise and factual. Return only the answer text.\n"
            "For bridge/compositional questions, output an entity/span phrase and do not output yes/no.\n"
            "For comparison questions, return exactly one token: yes or no.\n\n"
            "Root Question: {question}\n\n"
            "Question Type: {question_type}\n\n"
            "Solved Node Summaries:\n{node_summaries}\n\n"
            "Final Answer:"
        )

    def _default_bridge_refine_prompt(self) -> str:
        return (
            "Refine the bridge/compositional final answer to the most specific entity/span.\n"
            "Never output yes/no for bridge/compositional questions.\n"
            "If evidence is insufficient, output unknown.\n"
            "Return only answer text.\n\n"
            "Root Question: {question}\n"
            "Question Type: {question_type}\n"
            "Initial Answer: {initial_answer}\n\n"
            "Solved Node Summaries:\n{node_summaries}\n\n"
            "Refined Final Answer:"
        )
