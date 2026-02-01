"""Hybrid retrieval combining BM25 and Dense using Reciprocal Rank Fusion."""

import time
from typing import Optional

from ..core.retriever import BaseRetriever
from ..core.types import Document, RetrievalResult
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval using Reciprocal Rank Fusion (RRF).
    
    Combines BM25 sparse retrieval with dense embedding retrieval
    using RRF for score fusion.
    """

    def __init__(
        self,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        embedding_model: str = "text-embedding-3-small",
    ):
        """Initialize hybrid retriever.
        
        Args:
            bm25_weight: Weight for BM25 scores in RRF
            dense_weight: Weight for dense scores in RRF
            rrf_k: RRF constant (default 60 is standard)
            embedding_model: OpenAI embedding model for dense retrieval
        """
        super().__init__()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model=embedding_model)

    async def index(self, corpus: list[Document]) -> None:
        """Index the corpus for both retrievers.
        
        Args:
            corpus: List of documents to index
        """
        self._corpus = corpus
        self.corpus_size = len(corpus)
        
        # Index both retrievers
        await self.bm25.index(corpus)
        await self.dense.index(corpus)
        
        self.is_indexed = True

    def _reciprocal_rank_fusion(
        self,
        bm25_result: RetrievalResult,
        dense_result: RetrievalResult,
        top_k: int,
    ) -> tuple[list[Document], list[float]]:
        """Combine rankings using Reciprocal Rank Fusion.
        
        RRF score = sum(weight / (k + rank))
        
        Args:
            bm25_result: BM25 retrieval result
            dense_result: Dense retrieval result
            top_k: Number of documents to return
            
        Returns:
            Tuple of (ranked documents, RRF scores)
        """
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        
        # Add BM25 scores
        for rank, doc in enumerate(bm25_result.documents):
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
            doc_scores[doc.id] = doc_scores.get(doc.id, 0) + rrf_score
            doc_map[doc.id] = doc
        
        # Add Dense scores
        for rank, doc in enumerate(dense_result.documents):
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            doc_scores[doc.id] = doc_scores.get(doc.id, 0) + rrf_score
            doc_map[doc.id] = doc
        
        # Sort by combined RRF score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        sorted_ids = sorted_ids[:top_k]
        
        result_docs = [doc_map[doc_id] for doc_id in sorted_ids]
        result_scores = [doc_scores[doc_id] for doc_id in sorted_ids]
        
        return result_docs, result_scores

    async def retrieve(
        self,
        query: str,
        corpus: list[Document],
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve documents using hybrid BM25 + Dense with RRF.
        
        Args:
            query: Search query
            corpus: Document corpus
            top_k: Number of documents to retrieve
            
        Returns:
            RetrievalResult with ranked documents
        """
        start_time = time.perf_counter()
        
        # Re-index if needed
        if self._ensure_indexed(corpus):
            await self.index(corpus)
        
        # Get results from both retrievers (fetch more for fusion)
        fetch_k = min(top_k * 2, self.corpus_size)
        
        bm25_result = await self.bm25.retrieve(query, corpus, top_k=fetch_k)
        dense_result = await self.dense.retrieve(query, corpus, top_k=fetch_k)
        
        # Fuse rankings
        result_docs, result_scores = self._reciprocal_rank_fusion(
            bm25_result, dense_result, top_k
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResult(
            documents=result_docs,
            scores=result_scores,
            query=query,
            retrieval_time_ms=elapsed_ms,
            method="hybrid",
        )


def create_retriever(
    method: str = "hybrid",
    **kwargs,
) -> BaseRetriever:
    """Factory function to create a retriever.
    
    Args:
        method: 'bm25', 'dense', or 'hybrid'
        **kwargs: Additional arguments for the retriever
        
    Returns:
        Configured retriever instance
    """
    if method == "bm25":
        return BM25Retriever(**kwargs)
    elif method == "dense":
        return DenseRetriever(**kwargs)
    elif method == "hybrid":
        return HybridRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
