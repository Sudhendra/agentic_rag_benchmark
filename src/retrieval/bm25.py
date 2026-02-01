"""BM25 sparse retrieval implementation."""

import time
from typing import Optional

from rank_bm25 import BM25Okapi

from ..core.retriever import BaseRetriever
from ..core.types import Document, RetrievalResult


class BM25Retriever(BaseRetriever):
    """BM25 sparse retrieval using rank-bm25 library."""

    def __init__(self, tokenizer=None):
        """Initialize BM25 retriever.
        
        Args:
            tokenizer: Optional custom tokenizer function.
                       Default: lowercase and split on whitespace.
        """
        super().__init__()
        self.tokenizer = tokenizer or self._default_tokenizer
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def _default_tokenizer(text: str) -> list[str]:
        """Default tokenizer: lowercase and split on whitespace."""
        return text.lower().split()

    async def index(self, corpus: list[Document]) -> None:
        """Index the corpus for BM25 retrieval.
        
        Args:
            corpus: List of documents to index
        """
        self._corpus = corpus
        self.corpus_size = len(corpus)
        
        # Tokenize all documents
        tokenized_corpus = [self.tokenizer(doc.text) for doc in corpus]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_indexed = True

    async def retrieve(
        self,
        query: str,
        corpus: list[Document],
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve documents using BM25 scoring.
        
        Args:
            query: Search query
            corpus: Document corpus (will re-index if different from indexed)
            top_k: Number of documents to retrieve
            
        Returns:
            RetrievalResult with ranked documents
        """
        start_time = time.perf_counter()
        
        # Re-index if needed
        if self._ensure_indexed(corpus):
            await self.index(corpus)
        
        # Tokenize query
        tokenized_query = self.tokenizer(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Build results
        result_docs = [self._corpus[i] for i in top_indices]
        result_scores = [float(scores[i]) for i in top_indices]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResult(
            documents=result_docs,
            scores=result_scores,
            query=query,
            retrieval_time_ms=elapsed_ms,
            method="bm25",
        )
