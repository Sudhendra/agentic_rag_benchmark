"""Dense retrieval using OpenAI embeddings."""

import os
import time

import numpy as np
import openai
from dotenv import load_dotenv

from ..core.retriever import BaseRetriever
from ..core.types import Document, RetrievalResult

load_dotenv()


class DenseRetriever(BaseRetriever):
    """Dense retrieval using OpenAI embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 2048,
    ):
        """Initialize dense retriever.

        Args:
            model: OpenAI embedding model to use
            api_key: Optional API key (defaults to env var)
            batch_size: Max texts per embedding API call
        """
        super().__init__()
        self.model = model
        self.batch_size = batch_size

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.embeddings: np.ndarray | None = None

    async def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            NumPy array of embeddings (n_texts, embedding_dim)
        """
        all_embeddings = []

        # Batch embedding requests
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    async def index(self, corpus: list[Document]) -> None:
        """Index the corpus by computing embeddings.

        Args:
            corpus: List of documents to index
        """
        self._corpus = corpus
        self.corpus_size = len(corpus)

        # Get document texts
        texts = [doc.text for doc in corpus]

        # Compute embeddings
        self.embeddings = await self._embed_texts(texts)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms

        self.is_indexed = True

    async def retrieve(
        self,
        query: str,
        corpus: list[Document],
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve documents using dense embedding similarity.

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

        # Embed query
        query_embedding = await self._embed_texts([query])
        query_embedding = query_embedding[0]

        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compute cosine similarity (dot product since normalized)
        scores = np.dot(self.embeddings, query_embedding)

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
            method="dense",
        )
