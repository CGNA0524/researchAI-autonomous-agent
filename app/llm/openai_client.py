"""
Modular OpenAI API Client Integration.
Provides async completions, structured JSON output, retry backoff,
token usage capture, and verified cost estimation.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError, APITimeoutError
from app.config import get_settings, get_model_pricing
from app.llm.base import BaseLLMClient, BudgetExceededError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """
    Modular client wrapping the official AsyncOpenAI SDK.
    Supports structured JSON generation, rate limit retry backoff,
    hard pre-call budget guard, and granular token cost accounting.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None
    ) -> None:
        self.settings = get_settings()
        self._api_key = api_key if api_key is not None else self.settings.OPENAI_API_KEY
        self._model = model or self.settings.OPENAI_MODEL
        self._max_retries = max_retries if max_retries is not None else self.settings.MAX_RETRIES

        if not self._api_key or self._api_key.strip() in ("", "your_openai_api_key_here"):
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Please configure a valid OpenAI API key in your .env file or constructor."
            )

        self._client: Optional[AsyncOpenAI] = None
        # Per-request usage accumulator
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0
        self._llm_calls_count: int = 0
        self._retries_count: int = 0
        self._fallback_calls_count: int = 0
        self._last_model_used: str = self._model

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def active_model(self) -> str:
        return self._model

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of AsyncOpenAI client without leaking credentials."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self.settings.HTTP_TIMEOUT_SECONDS
            )
        return self._client

    def reset_request_usage(self) -> None:
        """Resets token usage accumulator for a new research workflow."""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._llm_calls_count = 0
        self._retries_count = 0
        self._fallback_calls_count = 0

    def get_current_estimated_cost(self) -> float:
        """Returns the current accumulated cost in USD."""
        pricing = get_model_pricing(self._last_model_used)
        manual_cost = getattr(self, "total_cost_usd", 0.0)
        if not pricing:
            return manual_cost
        token_cost = (
            (self._prompt_tokens / 1_000_000.0 * pricing["input"]) +
            (self._completion_tokens / 1_000_000.0 * pricing["output"])
        )
        return max(token_cost, manual_cost)

    def estimate_call_cost(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        prompt_chars: Optional[int] = None
    ) -> float:
        """Estimates potential cost in USD before making an API call."""
        target_model = model or self._model
        pricing = get_model_pricing(target_model)
        if not pricing:
            return 0.0
        if prompt_chars is not None:
            total_chars = prompt_chars
        elif messages:
            total_chars = sum(len(m.get("content", "")) for m in messages) + (len(messages) * 40)
        else:
            total_chars = 1000

        est_prompt = int(total_chars / 3.8)
        est_completion = max_tokens or 1500
        cost = (
            (est_prompt / 1_000_000.0 * pricing["input"]) +
            (est_completion / 1_000_000.0 * pricing["output"])
        )
        return cost

    def get_request_usage(self) -> Dict[str, Any]:
        """
        Returns the aggregated token usage and calculated API cost for the current request.
        If model pricing is unconfigured, estimated_cost_usd is reported as None (unavailable).
        """
        pricing = get_model_pricing(self._last_model_used)
        estimated_cost: Optional[float] = None
        current_cost = self.get_current_estimated_cost()
        if pricing is not None:
            estimated_cost = round(current_cost, 6)

        budget_cap = self.settings.MAX_REQUEST_BUDGET_USD
        remaining = max(0.0, budget_cap - current_cost)

        return {
            "provider": self.provider_name,
            "model": self._last_model_used,
            "llm_calls_count": self._llm_calls_count,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": estimated_cost,
            "retries_count": self._retries_count,
            "fallback_calls_count": self._fallback_calls_count,
            "budget_cap_usd": budget_cap,
            "budget_remaining_usd": round(remaining, 6)
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Any:
        """
        Executes an OpenAI chat completion with pre-call budget guarding,
        exponential retry backoff, and token tracking.
        """
        target_model = model or self._model
        temp = temperature if temperature is not None else self.settings.OPENAI_TEMPERATURE
        tokens = max_tokens or self.settings.OPENAI_MAX_TOKENS

        # ----------------------------------------------------------------------
        # Pre-Call Hard Budget Guard (Safety margin 1.25x)
        # ----------------------------------------------------------------------
        projected_call_cost = self.estimate_call_cost(messages, target_model, tokens)
        current_cost = self.get_current_estimated_cost()
        projected_total = current_cost + (projected_call_cost * 1.25)

        if projected_total > self.settings.MAX_REQUEST_BUDGET_USD:
            msg = (
                f"Pre-call budget guard blocked completion: Projected request cost (${projected_total:.4f}) "
                f"exceeds configured per-request budget cap (${self.settings.MAX_REQUEST_BUDGET_USD:.2f})."
            )
            logger.critical(msg)
            raise BudgetExceededError(msg)

        last_exception: Optional[Exception] = None
        max_attempts = self._max_retries

        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(
                    f"Invoking OpenAI model '{target_model}' (attempt {attempt}/{max_attempts})"
                )

                kwargs: Dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice

                response = await self.client.chat.completions.create(**kwargs)

                # Record usage metrics from response metadata
                self._record_usage(response, target_model)
                return response

            except AuthenticationError:
                logger.critical(
                    "OpenAI Authentication Error: Invalid or missing API key. Please verify OPENAI_API_KEY in .env."
                )
                raise

            except RateLimitError as e:
                self._retries_count += 1
                wait_time = self.settings.BACKOFF_FACTOR ** attempt
                logger.warning(
                    f"OpenAI rate limit hit on model '{target_model}'. Backing off for {wait_time:.1f}s. Reason: {str(e)}"
                )
                await asyncio.sleep(wait_time)
                last_exception = e

            except APITimeoutError as e:
                self._retries_count += 1
                wait_time = self.settings.BACKOFF_FACTOR ** attempt
                logger.warning(f"OpenAI API timeout on model '{target_model}'. Retrying in {wait_time:.1f}s.")
                await asyncio.sleep(wait_time)
                last_exception = e
                await asyncio.sleep(wait_time)
                last_exception = e

            except APIError as e:
                logger.error(f"OpenAI API Error on model '{target_model}': {str(e)}")
                wait_time = self.settings.BACKOFF_FACTOR ** attempt
                await asyncio.sleep(wait_time)
                last_exception = e

            except Exception as e:
                logger.error(f"Unexpected error communicating with OpenAI: {str(e)}")
                last_exception = e
                break

        raise RuntimeError(f"All OpenAI completion attempts failed: {str(last_exception)}")

    async def generate_structured_json(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str = "You are a helpful research assistant that outputs valid JSON.",
        target_schema_name: str = "JSONOutput",
    ) -> Dict[str, Any]:
        """
        Executes a completion enforcing JSON object output and returns a validated dictionary.
        """
        formatted_messages = [
            {"role": "system", "content": system_instruction},
            *messages
        ]

        response = await self.chat_completion(
            messages=formatted_messages,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"Empty content received from OpenAI while generating {target_schema_name}")

        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from OpenAI: {content}")
            raise ValueError(f"OpenAI produced malformed JSON for {target_schema_name}: {str(e)}") from e

    def _record_usage(
        self,
        response: Any = None,
        model_name: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> None:
        """Safely captures usage metrics from ChatCompletion response or explicit token counts."""
        target_model_name = model or model_name
        if prompt_tokens is not None and completion_tokens is not None:
            prompt_toks = prompt_tokens
            comp_toks = completion_tokens
            tot_toks = prompt_toks + comp_toks
            current_model = target_model_name or self._last_model_used
        else:
            usage = getattr(response, "usage", None)
            if not usage:
                return
            prompt_toks = getattr(usage, "prompt_tokens", 0)
            comp_toks = getattr(usage, "completion_tokens", 0)
            tot_toks = getattr(usage, "total_tokens", prompt_toks + comp_toks)
            current_model = getattr(response, "model", target_model_name or self._last_model_used)

        self._prompt_tokens += prompt_toks
        self._completion_tokens += comp_toks
        self._total_tokens += tot_toks
        self._llm_calls_count += 1
        self._last_model_used = current_model

        pricing = get_model_pricing(self._last_model_used)
        if pricing:
            cost = (
                (prompt_toks / 1_000_000.0 * pricing["input"]) +
                (comp_toks / 1_000_000.0 * pricing["output"])
            )
            logger.info(
                f"OpenAI API Call [{self._last_model_used}]: {prompt_toks} in, {comp_toks} out ({tot_toks} total) | Est. Cost: ${cost:.5f}"
            )
            # Check budget warning threshold
            cumulative_cost = (
                (self._prompt_tokens / 1_000_000.0 * pricing["input"]) +
                (self._completion_tokens / 1_000_000.0 * pricing["output"])
            )
            if cumulative_cost >= self.settings.MAX_REQUEST_BUDGET_USD:
                logger.warning(
                    f"Cumulative request cost (${cumulative_cost:.4f}) reached or exceeded configured budget safety cap (${self.settings.MAX_REQUEST_BUDGET_USD:.2f})."
                )
        else:
            logger.info(
                f"OpenAI API Call [{self._last_model_used}]: {prompt_toks} in, {comp_toks} out ({tot_toks} total) | Pricing: Unavailable"
            )
