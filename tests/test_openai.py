"""
Unit tests for OpenAI client integration, provider switching, and token/cost tracking.
All OpenAI API calls are mocked using unittest.mock to ensure zero credit consumption.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import openai
from app.config import get_settings, get_model_pricing
from app.llm.base import BaseLLMClient
from app.llm.openai_client import OpenAIClient
from app.llm.groq_client import GroqClient
from app.llm.factory import get_llm_client


def _create_mock_chat_completion(
    content: str,
    prompt_tokens: int = 150,
    completion_tokens: int = 50,
    model: str = "gpt-4o-mini"
):
    """Helper to construct a mock OpenAI chat completion object with usage statistics."""
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens

    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = model
    return mock_response


def test_openai_client_initialization_success():
    """Test 1: OpenAI client initializes properly with valid key and default model."""
    client = OpenAIClient(api_key="sk-proj-testkey12345678901234567890", model="gpt-4o-mini")
    assert client.provider_name == "openai"
    assert client.active_model == "gpt-4o-mini"
    assert isinstance(client, BaseLLMClient)


def test_openai_client_missing_key_raises_error():
    """Test 1b: OpenAI client raises ValueError when API key is missing or placeholder."""
    with patch("app.llm.openai_client.get_settings") as mock_settings:
        mock_settings.return_value.OPENAI_API_KEY = "your_openai_api_key_here"
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
            OpenAIClient(api_key=None)


@pytest.mark.asyncio
async def test_openai_successful_chat_completion():
    """Test 2: Successful OpenAI response using mocked AsyncOpenAI client."""
    client = OpenAIClient(api_key="sk-proj-testkey12345678901234567890")

    mock_resp = _create_mock_chat_completion(
        content="OpenAI synthesized research answer",
        prompt_tokens=200,
        completion_tokens=80
    )

    with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_resp

        result = await client.chat_completion(
            messages=[{"role": "user", "content": "What is Qdrant?"}]
        )

        assert result.choices[0].message.content == "OpenAI synthesized research answer"
        assert mock_create.await_count == 1
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_openai_authentication_failure():
    """Test 3: Authentication failure raises cleanly and does not expose API key."""
    client = OpenAIClient(api_key="sk-proj-secretcredentialthatmustnotleak12345")

    # Construct mock 401 AuthenticationError
    mock_request = MagicMock()
    auth_error = openai.AuthenticationError(
        message="Incorrect API key provided.",
        response=MagicMock(status_code=401, headers={}),
        body={"error": {"message": "Incorrect API key provided."}}
    )

    with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = auth_error

        with pytest.raises(openai.AuthenticationError) as exc_info:
            await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}]
            )

        err_msg = str(exc_info.value)
        assert "Incorrect API key" in err_msg
        assert "secretcredentialthatmustnotleak" not in err_msg


@pytest.mark.asyncio
async def test_openai_rate_limit_backoff():
    """Test 4: Rate-limit error triggers backoff retry and succeeds on subsequent attempt."""
    client = OpenAIClient(api_key="sk-proj-testkey12345678901234567890", max_retries=2)

    rate_limit_err = openai.RateLimitError(
        message="Rate limit exceeded.",
        response=MagicMock(status_code=429, headers={}),
        body={"error": {"message": "Rate limit exceeded."}}
    )
    success_resp = _create_mock_chat_completion("Recovered from rate limit")

    with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        # First call fails with 429, second call succeeds
        mock_create.side_effect = [rate_limit_err, success_resp]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.chat_completion(
                messages=[{"role": "user", "content": "Test rate limit"}]
            )

            assert result.choices[0].message.content == "Recovered from rate limit"
            assert mock_create.await_count == 2
            assert mock_sleep.await_count == 1


@pytest.mark.asyncio
async def test_openai_invalid_json_handling():
    """Test 5: Structured JSON generation raises ValueError on unparseable JSON."""
    client = OpenAIClient(api_key="sk-proj-testkey12345678901234567890")

    bad_json_resp = _create_mock_chat_completion("Not valid JSON: { broken syntax")

    with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = bad_json_resp

        with pytest.raises(ValueError, match="malformed JSON"):
            await client.generate_structured_json(
                messages=[{"role": "user", "content": "Return JSON"}],
                target_schema_name="TestSchema"
            )


@pytest.mark.asyncio
async def test_openai_token_usage_and_cost_tracking():
    """Test 6: Accurate token usage accumulation and cost estimation based on gpt-4o-mini pricing."""
    client = OpenAIClient(api_key="sk-proj-testkey12345678901234567890", model="gpt-4o-mini")
    client.reset_request_usage()

    # Prompt: 10,000 tokens ($0.15/1M -> $0.0015)
    # Completion: 5,000 tokens ($0.60/1M -> $0.0030)
    # Expected total cost: $0.004500
    mock_resp1 = _create_mock_chat_completion(
        content='{"key": "value"}',
        prompt_tokens=10000,
        completion_tokens=5000,
        model="gpt-4o-mini"
    )

    with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_resp1

        await client.generate_structured_json(
            messages=[{"role": "user", "content": "Query 1"}],
            target_schema_name="TestSchema"
        )

        usage = client.get_request_usage()
        assert usage["llm_calls_count"] == 1
        assert usage["prompt_tokens"] == 10000
        assert usage["completion_tokens"] == 5000
        assert usage["total_tokens"] == 15000
        assert usage["provider"] == "openai"
        assert usage["model"] == "gpt-4o-mini"
        assert usage["estimated_cost_usd"] is not None
        assert round(usage["estimated_cost_usd"], 6) == 0.004500

        # Run second call to ensure accumulation across request
        mock_resp2 = _create_mock_chat_completion(
            content='{"key": "value2"}',
            prompt_tokens=20000,
            completion_tokens=10000,
            model="gpt-4o-mini"
        )
        mock_create.return_value = mock_resp2

        await client.generate_structured_json(
            messages=[{"role": "user", "content": "Query 2"}],
            target_schema_name="TestSchema"
        )

        usage2 = client.get_request_usage()
        assert usage2["llm_calls_count"] == 2
        assert usage2["prompt_tokens"] == 30000
        assert usage2["completion_tokens"] == 15000
        assert usage2["total_tokens"] == 45000
        # 30,000 * 0.15/1M + 15,000 * 0.60/1M = 0.0045 + 0.0090 = $0.0135
        assert round(usage2["estimated_cost_usd"], 6) == 0.013500


def test_unknown_model_pricing_returns_none():
    """Test 6b: Unknown model returns None for estimated cost instead of inventing zero."""
    client = OpenAIClient(api_key="sk-proj-testkey12345678901234567890", model="unlisted-custom-model")
    client._record_usage(prompt_tokens=1000, completion_tokens=500, model="unlisted-custom-model")

    usage = client.get_request_usage()
    assert usage["prompt_tokens"] == 1000
    assert usage["completion_tokens"] == 500
    assert usage["estimated_cost_usd"] is None


def test_provider_switching_factory():
    """Test 7: Provider switching correctly selects Groq or OpenAI client based on parameter or env."""
    # 1. Explicit Groq selection
    groq_client = get_llm_client("groq")
    assert isinstance(groq_client, GroqClient)
    assert groq_client.provider_name == "groq"

    # 2. Explicit OpenAI selection with mock key
    with patch("app.llm.factory.get_settings") as mock_settings:
        mock_settings.return_value.OPENAI_API_KEY = "sk-proj-testmockkey1234567890123"
        mock_settings.return_value.OPENAI_MODEL = "gpt-4o-mini"
        mock_settings.return_value.OPENAI_TEMPERATURE = 0.1
        mock_settings.return_value.OPENAI_MAX_TOKENS = 4096
        mock_settings.return_value.MAX_REQUEST_BUDGET_USD = 0.10

        openai_client = get_llm_client("openai")
        assert isinstance(openai_client, OpenAIClient)
        assert openai_client.provider_name == "openai"

    # 3. Invalid provider raises descriptive ValueError
    with pytest.raises(ValueError, match="Unsupported LLM provider 'anthropic'"):
        get_llm_client("anthropic")


def test_groq_functionality_intact():
    """Test 8: Groq client remains fully functional and conforms to BaseLLMClient interface."""
    client = GroqClient()
    assert client.provider_name == "groq"
    assert client.active_model == client.primary_model
    assert isinstance(client, BaseLLMClient)

    # Usage tracking interface
    client.reset_request_usage()
    client._record_usage(prompt_tokens=500, completion_tokens=250, model=client.primary_model)
    usage = client.get_request_usage()
    assert usage["provider"] == "groq"
    assert usage["prompt_tokens"] == 500
    assert usage["completion_tokens"] == 250
    assert usage["total_tokens"] == 750
