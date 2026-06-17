"""LLM API clients with caching and cost tracking."""

import os
from abc import ABC, abstractmethod

import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..utils.cache import SQLiteCache

# Load environment variables
load_dotenv()


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(
        self,
        model: str,
        cache: SQLiteCache | None = None,
        track_costs: bool = True,
    ):
        """Initialize the LLM client.

        Args:
            model: Model identifier (e.g., 'gpt-4o-mini')
            cache: Optional SQLite cache for responses
            track_costs: Whether to track token usage and costs
        """
        self.model = model
        self.cache = cache
        self.track_costs = track_costs

        # Statistics
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> tuple[str, int, float]:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            stop: Optional stop sequences

        Returns:
            Tuple of (response_text, tokens_used, cost_usd)
        """
        pass

    def _make_cache_key(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> str:
        """Create a deterministic cache key for a request."""
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            "seed": seed,
        }
        return SQLiteCache.make_key(data)

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "model": self.model,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible API client with caching and cost tracking.

    Supports any OpenAI-compatible provider (OpenAI, Groq, Together AI, etc.)
    by configuring base_url and api_key_env_var.
    """

    # Pricing per 1M tokens (as of Jan 2026)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    EMBEDDING_PRICING = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        cache: SQLiteCache | None = None,
        track_costs: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env_var: str | None = None,
        pricing: dict | None = None,
    ):
        """Initialize OpenAI-compatible client.

        Args:
            model: Model identifier (e.g. 'gpt-4o-mini', 'llama3-70b-8192')
            cache: Optional response cache
            track_costs: Whether to track token usage and costs
            api_key: Optional API key (defaults to api_key_env_var env var)
            base_url: Optional custom API base URL (for Groq, Together, etc.)
            api_key_env_var: Environment variable name for API key (default: OPENAI_API_KEY)
            pricing: Optional custom pricing dict to merge into PRICING
        """
        super().__init__(model, cache, track_costs)

        env_var = api_key_env_var or "OPENAI_API_KEY"
        api_key = api_key or os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found. Set {env_var} environment variable.")

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = openai.AsyncOpenAI(**client_kwargs)

        if pricing:
            self.PRICING = {**self.PRICING, **pricing}

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for the given token counts.

        Returns 0.0 for models not in the pricing table (e.g. free-tier providers).
        """
        pricing = self.PRICING.get(self.model)
        if pricing is None:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.PermissionDeniedError,
                openai.BadRequestError,
            )
        ),
    )
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> tuple[str, int, float]:
        """Generate a response from OpenAI.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Optional stop sequences

        Returns:
            Tuple of (response_text, tokens_used, cost_usd)
        """
        # Check cache first
        cache_key = self._make_cache_key(messages, temperature, max_tokens, stop, seed)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return tuple(cached)

        # Make API call
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            kwargs["stop"] = stop
        if seed is not None:
            kwargs["seed"] = seed

        response = await self.client.chat.completions.create(**kwargs)

        # Extract response and metrics
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = input_tokens + output_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)

        # Update stats
        if self.track_costs:
            self.total_tokens += total_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.call_count += 1

        result = (content, total_tokens, cost)

        # Cache the result
        if self.cache:
            self.cache.set(cache_key, list(result))

        return result

    async def embed(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> tuple[list[list[float]], int, float]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            Tuple of (embeddings, tokens_used, cost_usd)
        """
        response = await self.client.embeddings.create(
            model=model,
            input=texts,
        )

        embeddings = [e.embedding for e in response.data]
        tokens_used = response.usage.total_tokens

        pricing = self.EMBEDDING_PRICING.get(model, 0.02)
        cost = (tokens_used / 1_000_000) * pricing

        if self.track_costs:
            self.total_tokens += tokens_used
            self.total_cost += cost

        return embeddings, tokens_used, cost


def create_llm_client(
    provider: str = "openai",
    model: str | None = None,
    cache: SQLiteCache | None = None,
    **kwargs,
) -> BaseLLMClient:
    """Factory function to create an LLM client.

    Args:
        provider: 'openai', 'groq', or 'anthropic' (not yet implemented)
        model: Model identifier (uses defaults if not specified)
        cache: Optional response cache
        **kwargs: Additional arguments for the client (e.g. base_url, api_key_env_var)

    Returns:
        Configured LLM client
    """
    if provider == "openai":
        model = model or "gpt-4o-mini"
        return OpenAIClient(model=model, cache=cache, **kwargs)
    elif provider == "groq":
        kwargs.setdefault("api_key_env_var", "GROQ_API_KEY")
        kwargs.setdefault("base_url", "https://api.groq.com/openai/v1")
        model = model or "llama3-70b-8192"
        return OpenAIClient(model=model, cache=cache, **kwargs)
    elif provider == "anthropic":
        raise NotImplementedError(
            "Anthropic client not yet implemented. "
            "Add anthropic to dependencies and implement AnthropicClient."
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
