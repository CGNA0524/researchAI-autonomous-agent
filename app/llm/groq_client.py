"""
Groq API Client Integration with Retries, Model Fallbacks, and Structured Output.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from groq import AsyncGroq, APIError, RateLimitError, AuthenticationError
from app.config import get_settings
from app.llm.base import BaseLLMClient
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GroqClient(BaseLLMClient):
    """
    Wrapper around the official AsyncGroq client with resilient error handling,
    model fallback, rate limit backoff, JSON validation, and token usage tracking.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: Optional[AsyncGroq] = None
        # Per-request usage accumulator
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0
        self._llm_calls_count: int = 0
        self._last_model_used: str = self.settings.GROQ_MODEL

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def active_model(self) -> str:
        return self.settings.GROQ_MODEL

    @property
    def primary_model(self) -> str:
        return self.settings.GROQ_MODEL

    @property
    def fallback_model(self) -> str:
        return self.settings.GROQ_FALLBACK_MODEL

    def reset_request_usage(self) -> None:
        """Resets token usage accumulator for a new research workflow."""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._llm_calls_count = 0

    def get_request_usage(self) -> Dict[str, Any]:
        """Returns the aggregated token usage for the current request (Groq developer tier is free)."""
        return {
            "provider": self.provider_name,
            "model": self._last_model_used,
            "llm_calls_count": self._llm_calls_count,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": 0.0  # Groq on-demand developer tier
        }

    @property
    def client(self) -> AsyncGroq:
        """Lazy initialization of the AsyncGroq client."""
        if self._client is None:
            if not self.settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY is not configured. Real LLM completions will fail unless provided.")
            self._client = AsyncGroq(api_key=self.settings.GROQ_API_KEY or "dummy_key_for_init")
        return self._client

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
        Executes a chat completion call with automated retries and fallback model handling.
        """
        primary_model = model or self.settings.GROQ_MODEL
        fallback_model = self.settings.GROQ_FALLBACK_MODEL
        temp = temperature if temperature is not None else self.settings.GROQ_TEMPERATURE
        tokens = max_tokens or self.settings.GROQ_MAX_TOKENS

        models_to_try = [primary_model]
        if fallback_model and fallback_model != primary_model:
            models_to_try.append(fallback_model)

        last_exception: Optional[Exception] = None

        for current_model in models_to_try:
            for attempt in range(1, self.settings.MAX_RETRIES + 1):
                try:
                    logger.debug(
                        f"Invoking Groq model '{current_model}' (attempt {attempt}/{self.settings.MAX_RETRIES})"
                    )

                    kwargs: Dict[str, Any] = {
                        "model": current_model,
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
                    self._record_usage(response, current_model)
                    return response

                except AuthenticationError as e:
                    logger.critical(f"Groq Authentication Error: Invalid or missing API key. {str(e)}")
                    raise RuntimeError("Invalid or missing Groq API Key. Please verify GROQ_API_KEY in .env.") from e

                except RateLimitError as e:
                    wait_time = self.settings.BACKOFF_FACTOR ** attempt
                    logger.warning(
                        f"Groq Rate limit hit on model '{current_model}'. Backing off for {wait_time:.1f}s. Error: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                    last_exception = e

                except APIError as e:
                    logger.error(f"Groq API Error on model '{current_model}': {str(e)}")
                    wait_time = self.settings.BACKOFF_FACTOR ** attempt
                    await asyncio.sleep(wait_time)
                    last_exception = e

                except Exception as e:
                    logger.error(f"Unexpected error communicating with Groq: {str(e)}")
                    last_exception = e
                    break

            logger.warning(f"Exhausted retries on model '{current_model}'. Trying next model if available.")

        raise RuntimeError(f"All Groq completion attempts failed: {str(last_exception)}")

    async def generate_structured_json(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str,
        target_schema_name: str,
    ) -> Dict[str, Any]:
        """
        Executes a completion enforcing JSON object output and parses the result safely.
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
            raise ValueError(f"Empty content received from LLM while generating {target_schema_name}")

        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM: {content}")
            raise ValueError(f"LLM produced malformed JSON for {target_schema_name}: {str(e)}") from e

    def _record_usage(
        self,
        response: Any = None,
        model_name: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> None:
        """Safely captures usage metrics from Groq ChatCompletion response or explicit token counts."""
        target_model = model or model_name or self.settings.GROQ_MODEL
        if prompt_tokens is not None and completion_tokens is not None:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._total_tokens += (prompt_tokens + completion_tokens)
            self._llm_calls_count += 1
            self._last_model_used = target_model
            return

        usage = getattr(response, "usage", None)
        if usage:
            prompt_toks = getattr(usage, "prompt_tokens", 0)
            comp_toks = getattr(usage, "completion_tokens", 0)
            tot_toks = getattr(usage, "total_tokens", prompt_toks + comp_toks)

            self._prompt_tokens += prompt_toks
            self._completion_tokens += comp_toks
            self._total_tokens += tot_toks
            self._llm_calls_count += 1
            self._last_model_used = getattr(response, "model", model_name or self.settings.GROQ_MODEL)

