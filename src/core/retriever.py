"""Abstract base class for retriever implementations."""

from abc import ABC, abstractmethod
from typing import Optional

from .types import Document, RetrievalResult


class BaseRetriever(ABC):
    """Abstract base class for retrievers.
    
    All retriever implementations (BM25, Dense, Hybrid) inherit from this.
    """

    def __init__(self):
        """Initialize the retriever."""
        self.is_indexed = False
        self.corpus_size = 0
        self._corpus: Optional[list[Document]] = None

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        corpus: list[Document],
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query string
            corpus: List of documents to search
            top_k: Number of documents to retrieve
            
        Returns:
            RetrievalResult with documents and scores
        """
        pass

    @abstractmethod
    async def index(self, corpus: list[Document]) -> None:
        """Pre-index a corpus for faster retrieval.
        
        This should be called before retrieve() for optimal performance.
        
        Args:
            corpus: List of documents to index
        """
        pass

    async def batch_retrieve(
        self,
        queries: list[str],
        corpus: list[Document],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve for multiple queries.
        
        Default implementation calls retrieve() sequentially.
        Subclasses can override for parallel/batch optimization.
        
        Args:
            queries: List of search queries
            corpus: List of documents to search
            top_k: Number of documents per query
            
        Returns:
            List of RetrievalResult objects
        """
        results = []
        for query in queries:
            result = await self.retrieve(query, corpus, top_k)
            results.append(result)
        return results

    def _ensure_indexed(self, corpus: list[Document]) -> bool:
        """Check if corpus needs re-indexing.
        
        Args:
            corpus: Current corpus
            
        Returns:
            True if re-indexing is needed
        """
        if not self.is_indexed:
            return True
        if self._corpus is not corpus and len(corpus) != self.corpus_size:
            return True
        return False
