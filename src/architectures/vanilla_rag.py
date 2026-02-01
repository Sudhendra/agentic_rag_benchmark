"""Vanilla RAG - Simple retrieve-then-read baseline."""

import time
from pathlib import Path
from typing import Any

from ..core.base_rag import BaseRAG
from ..core.llm_client import BaseLLMClient
from ..core.retriever import BaseRetriever
from ..core.types import (
    ArchitectureType,
    Document,
    Question,
    RAGResponse,
    ReasoningStep,
)


class VanillaRAG(BaseRAG):
    """Vanilla RAG: Simple retrieve-then-read approach.
    
    This is the baseline architecture:
    1. Retrieve top-k documents for the question
    2. Concatenate documents into context
    3. Generate answer with single LLM call
    
    No iterative retrieval, no reasoning chain.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: BaseRetriever,
        config: dict,
    ):
        """Initialize Vanilla RAG.
        
        Args:
            llm_client: LLM API client
            retriever: Document retriever
            config: Configuration dict with 'top_k', 'max_context_tokens', etc.
        """
        super().__init__(llm_client, retriever, config)
        
        # Load prompt template
        prompt_path = self.config.get("prompt_path", "prompts/vanilla.txt")
        if Path(prompt_path).exists():
            self.prompt_template = self._load_prompt_template(prompt_path)
        else:
            self.prompt_template = (
                "Answer the question based on the provided context. "
                "Be concise and direct. Give only the answer without explanation.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )

    def get_name(self) -> str:
        return "vanilla_rag"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.VANILLA

    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        return {
            "top_k": (int, False, 5),
            "max_context_tokens": (int, False, 4000),
            "prompt_path": (str, False, "prompts/vanilla.txt"),
        }

    async def answer(
        self,
        question: Question,
        corpus: list[Document],
    ) -> RAGResponse:
        """Answer using simple retrieve-then-read.
        
        Args:
            question: The question to answer
            corpus: Document corpus to retrieve from
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.perf_counter()
        
        # Step 1: Retrieve documents
        retrieval_result = await self.retriever.retrieve(
            query=question.text,
            corpus=corpus,
            top_k=self.config["top_k"],
        )
        
        # Step 2: Build context from retrieved documents
        context = self._build_context(
            retrieval_result.documents,
            max_tokens=self.config["max_context_tokens"],
        )
        
        # Step 3: Generate answer
        prompt = self.prompt_template.format(
            context=context,
            question=question.text,
        )
        
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens_used, cost = await self.llm.generate(messages)
        
        # Clean up answer
        answer = response_text.strip()
        
        # Build response
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Create a single reasoning step representing the full process
        reasoning_step = ReasoningStep(
            step_id=1,
            thought="Retrieve relevant passages and generate answer.",
            action="retrieve_and_answer",
            action_input=question.text,
            observation=f"Retrieved {len(retrieval_result.documents)} documents. Generated answer.",
            tokens_used=tokens_used,
            cost_usd=cost,
        )
        
        return RAGResponse(
            answer=answer,
            reasoning_chain=[reasoning_step],
            retrieved_docs=[retrieval_result],
            total_tokens=tokens_used,
            total_cost_usd=cost,
            latency_ms=elapsed_ms,
            num_retrieval_calls=1,
            num_llm_calls=1,
            model=self.llm.model,
            architecture=self.get_name(),
        )
