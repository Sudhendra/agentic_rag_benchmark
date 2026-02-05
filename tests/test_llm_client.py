"""Unit tests for LLM client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.llm_client import OpenAIClient, BaseLLMClient, create_llm_client
from src.utils.cache import SQLiteCache


class TestOpenAIClientCostCalculation:
    """Tests for OpenAI client cost calculation."""

    def test_gpt4o_mini_cost(self):
        """Test cost calculation for gpt-4o-mini."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = OpenAIClient(model="gpt-4o-mini")

                # gpt-4o-mini: $0.15/1M input, $0.60/1M output
                cost = client._calculate_cost(input_tokens=1000, output_tokens=500)

                expected = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
                assert abs(cost - expected) < 0.0001

    def test_gpt4o_cost(self):
        """Test cost calculation for gpt-4o."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = OpenAIClient(model="gpt-4o")

                # gpt-4o: $2.50/1M input, $10.00/1M output
                cost = client._calculate_cost(input_tokens=1000, output_tokens=500)

                expected = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)
                assert abs(cost - expected) < 0.0001

    def test_unknown_model_uses_default(self):
        """Test that unknown models use default pricing."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = OpenAIClient(model="unknown-model")

                # Should use gpt-4o-mini pricing as fallback
                cost = client._calculate_cost(input_tokens=1000, output_tokens=500)

                expected = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
                assert abs(cost - expected) < 0.0001


class TestOpenAIClientStats:
    """Tests for OpenAI client statistics tracking."""

    def test_initial_stats_are_zero(self):
        """Test that initial stats are zero."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = OpenAIClient(model="gpt-4o-mini")

                assert client.total_tokens == 0
                assert client.total_input_tokens == 0
                assert client.total_output_tokens == 0
                assert client.total_cost == 0.0
                assert client.call_count == 0

    def test_get_stats_returns_dict(self):
        """Test that get_stats returns a dictionary."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = OpenAIClient(model="gpt-4o-mini")

                stats = client.get_stats()

                assert isinstance(stats, dict)
                assert stats["model"] == "gpt-4o-mini"
                assert stats["call_count"] == 0
                assert stats["total_tokens"] == 0
                assert stats["total_cost_usd"] == 0.0

    def test_reset_stats(self):
        """Test that reset_stats clears all statistics."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = OpenAIClient(model="gpt-4o-mini")

                # Manually set some stats
                client.total_tokens = 100
                client.call_count = 5
                client.total_cost = 0.01

                client.reset_stats()

                assert client.total_tokens == 0
                assert client.call_count == 0
                assert client.total_cost == 0.0


class TestOpenAIClientCaching:
    """Tests for OpenAI client caching behavior."""

    def test_cache_key_deterministic(self):
        """Test that cache keys are deterministic."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = OpenAIClient(model="gpt-4o-mini")

                messages = [{"role": "user", "content": "test"}]
                key1 = client._make_cache_key(messages, 0.0, 1024)
                key2 = client._make_cache_key(messages, 0.0, 1024)

                assert key1 == key2

    def test_cache_key_different_for_different_messages(self):
        """Test that different messages produce different keys."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = OpenAIClient(model="gpt-4o-mini")

                key1 = client._make_cache_key([{"role": "user", "content": "hello"}], 0.0, 1024)
                key2 = client._make_cache_key([{"role": "user", "content": "world"}], 0.0, 1024)

                assert key1 != key2

    def test_cache_key_different_for_different_params(self):
        """Test that different parameters produce different keys."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = OpenAIClient(model="gpt-4o-mini")

                messages = [{"role": "user", "content": "test"}]
                key1 = client._make_cache_key(messages, 0.0, 1024)
                key2 = client._make_cache_key(messages, 0.5, 1024)  # Different temperature

                assert key1 != key2


class TestOpenAIClientGenerate:
    """Tests for OpenAI client generate method."""

    @pytest.mark.asyncio
    async def test_generate_returns_tuple(self):
        """Test that generate returns (text, tokens, cost) tuple."""
        with patch("src.core.llm_client.openai.AsyncOpenAI") as mock_openai:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                # Setup mock response
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
                mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

                mock_client = AsyncMock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(model="gpt-4o-mini")

                result = await client.generate([{"role": "user", "content": "test"}])

                assert isinstance(result, tuple)
                assert len(result) == 3
                assert result[0] == "test response"
                assert result[1] == 15  # 10 + 5 tokens
                assert result[2] > 0  # Some cost

    @pytest.mark.asyncio
    async def test_generate_uses_cache(self):
        """Test that generate checks and uses cache."""
        with patch("src.core.llm_client.openai.AsyncOpenAI") as mock_openai:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                mock_cache = MagicMock(spec=SQLiteCache)
                mock_cache.get.return_value = ["cached response", 50, 0.001]

                mock_openai.return_value = AsyncMock()

                client = OpenAIClient(model="gpt-4o-mini", cache=mock_cache)

                result = await client.generate([{"role": "user", "content": "test"}])

                # Should return cached result
                assert result[0] == "cached response"
                mock_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_updates_stats(self):
        """Test that generate updates statistics."""
        with patch("src.core.llm_client.openai.AsyncOpenAI") as mock_openai:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=MagicMock(content="response"))]
                mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

                mock_client = AsyncMock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(model="gpt-4o-mini", track_costs=True)

                await client.generate([{"role": "user", "content": "test"}])

                assert client.total_tokens == 150
                assert client.total_input_tokens == 100
                assert client.total_output_tokens == 50
                assert client.call_count == 1


class TestCreateLLMClient:
    """Tests for the create_llm_client factory function."""

    def test_create_openai_client(self):
        """Test creating an OpenAI client."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = create_llm_client(provider="openai", model="gpt-4o-mini")

                assert isinstance(client, OpenAIClient)
                assert client.model == "gpt-4o-mini"

    def test_create_openai_default_model(self):
        """Test that OpenAI client uses default model if not specified."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                client = create_llm_client(provider="openai")

                assert client.model == "gpt-4o-mini"

    def test_create_anthropic_raises(self):
        """Test that Anthropic client raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            create_llm_client(provider="anthropic")

    def test_create_unknown_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client(provider="unknown")


class TestOpenAIClientInitialization:
    """Tests for OpenAI client initialization."""

    def test_raises_without_api_key(self):
        """Test that initialization raises without API key."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict("os.environ", {}, clear=True):
                # Clear any existing OPENAI_API_KEY
                import os

                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]

                with pytest.raises(ValueError, match="API key not found"):
                    OpenAIClient(model="gpt-4o-mini")

    def test_accepts_api_key_parameter(self):
        """Test that API key can be passed as parameter."""
        with patch("src.core.llm_client.openai.AsyncOpenAI"):
            # Should not raise even without env var
            client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")

            assert client.model == "gpt-4o-mini"
