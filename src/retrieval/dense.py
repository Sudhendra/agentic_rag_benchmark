"""Dense retrieval using OpenAI embeddings."""

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..core.retriever import BaseRetriever
from ..core.types import Document, RetrievalResult

load_dotenv()


def _get_cache_dir() -> Path:
    """Get the embedding cache directory."""
    cache_dir = Path(os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_key(corpus: list[Document], model: str) -> str:
    """Generate a cache key based on corpus and model."""
    corpus_hash = hashlib.sha256()
    for doc in corpus:
        corpus_hash.update(doc.id.encode())
        corpus_hash.update(doc.text[:1000].encode())

    key_data = f"{model}_{len(corpus)}_{corpus_hash.hexdigest()[:16]}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def _load_cached_embeddings(cache_path: Path) -> tuple[np.ndarray, list[str]] | None:
    """Load cached embeddings if available and valid."""
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        doc_ids = data["doc_ids"].tolist()
        return embeddings, doc_ids
    except Exception:
        return None


def _save_embeddings(cache_path: Path, embeddings: np.ndarray, doc_ids: list[str]) -> None:
    """Save embeddings to cache."""
    np.savez(cache_path, embeddings=embeddings, doc_ids=np.array(doc_ids))


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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.BadRequestError,
                openai.APIConnectionError,
            )
        ),
        reraise=True,
    )
    async def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            NumPy array of embeddings (n_texts, embedding_dim)
        """
        all_embeddings = []

        # Batch embedding requests
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Log batch info for debugging
            batch_text = batch[0] if batch else ""
            print(f"  Embedding batch {i // self.batch_size + 1}, first text: {batch_text[:50]}...")

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
        doc_ids = [doc.id for doc in corpus]

        # Check for cached embeddings
        cache_dir = _get_cache_dir()
        cache_key = _get_cache_key(corpus, self.model)
        cache_path = cache_dir / f"dense_{self.model.replace('-', '_')}_{cache_key}.npz"

        print("DenseRetriever: Checking for cached embeddings...")
        cached = _load_cached_embeddings(cache_path)

        if cached is not None:
            cached_embeddings, cached_doc_ids = cached
            # Verify cache is for the same corpus
            if len(cached_embeddings) == len(texts) and cached_doc_ids == doc_ids:
                print(f"DenseRetriever: Loaded {len(cached_embeddings)} embeddings from cache!")
                self.embeddings = cached_embeddings
                self.is_indexed = True
                return
            else:
                print("DenseRetriever: Cache size mismatch, recomputing...")

        # Compute embeddings
        print(f"DenseRetriever: Computing embeddings for {len(texts)} documents...")
        start_time = time.time()
        self.embeddings = await self._embed_texts(texts)
        elapsed = time.time() - start_time
        print(f"DenseRetriever: Computed embeddings in {elapsed:.1f}s")

        # Save to cache
        print("DenseRetriever: Saving embeddings to cache...")
        _save_embeddings(cache_path, self.embeddings, doc_ids)
        print("DenseRetriever: Cached embeddings saved!")

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
