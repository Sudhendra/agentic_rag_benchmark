"""Abstract base class for RAG architectures."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from .llm_client import BaseLLMClient
from .retriever import BaseRetriever
from .types import ArchitectureType, Document, Question, RAGResponse


class BaseRAG(ABC):
    """Abstract base class for all RAG architectures.
    
    All implementations (Vanilla, ReAct, Self-RAG, IRCoT, etc.) inherit from this.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: BaseRetriever,
        config: dict,
    ):
        """Initialize the RAG architecture.
        
        Args:
            llm_client: LLM API client for generation
            retriever: Retriever for document search
            config: Architecture-specific configuration
        """
        self.llm = llm_client
        self.retriever = retriever
        self.config = config
        self._validate_config()

    @abstractmethod
    async def answer(
        self,
        question: Question,
        corpus: list[Document],
    ) -> RAGResponse:
        """Answer a question using the RAG architecture.
        
        This is the main entry point that all architectures must implement.
        
        Args:
            question: The question to answer
            corpus: List of documents to retrieve from
            
        Returns:
            RAGResponse with answer, reasoning chain, and metadata
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the architecture name (e.g., 'vanilla_rag', 'react_rag').
        
        Returns:
            String identifier for the architecture
        """
        pass

    @abstractmethod
    def get_type(self) -> ArchitectureType:
        """Return the architecture category.
        
        Returns:
            ArchitectureType enum value (VANILLA, AGENTIC, RECURSIVE, RLM)
        """
        pass

    @abstractmethod
    def get_config_schema(self) -> dict[str, tuple[type, bool, Any]]:
        """Return the configuration schema for validation.
        
        Returns:
            Dict mapping config keys to (type, required, default) tuples
        """
        pass

    def _validate_config(self) -> None:
        """Validate configuration against schema.
        
        Raises:
            ValueError: If required config key is missing
            TypeError: If config value has wrong type
        """
        schema = self.get_config_schema()
        for key, (key_type, required, default) in schema.items():
            if key not in self.config:
                if required:
                    raise ValueError(f"Missing required config key: {key}")
                self.config[key] = default
            elif not isinstance(self.config[key], key_type):
                raise TypeError(
                    f"Config key '{key}' must be {key_type.__name__}, "
                    f"got {type(self.config[key]).__name__}"
                )

    def _load_prompt_template(self, prompt_path: str) -> str:
        """Load a prompt template from file.
        
        Args:
            prompt_path: Path to the prompt file
            
        Returns:
            Prompt template string
        """
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_context(
        self,
        documents: list[Document],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Build context string from retrieved documents.
        
        Args:
            documents: List of documents to include
            max_tokens: Optional token limit (approximate)
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[{i}] {doc.title}\n{doc.text}")
        
        context = "\n\n".join(context_parts)
        
        # Rough token limiting (4 chars ≈ 1 token)
        if max_tokens:
            max_chars = max_tokens * 4
            if len(context) > max_chars:
                context = context[:max_chars] + "..."
        
        return context
