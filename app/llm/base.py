"""
Base LLM Client Abstract Interface.
Defines common contracts for multi-provider support (Groq, OpenAI).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BudgetExceededError(RuntimeError):
    """Raised when an LLM call is blocked by the proactive budget guard."""
    pass


class BaseLLMClient(ABC):
    """Abstract interface that all LLM provider clients must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the identifier of the LLM provider (e.g. 'groq', 'openai')."""
        pass

    @property
    @abstractmethod
    def active_model(self) -> str:
        """Returns the primary model currently configured for completions."""
        pass

    @abstractmethod
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
        """Executes a chat completion call with automated retries and error handling."""
        pass

    @abstractmethod
    async def generate_structured_json(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str = "You are a helpful research assistant that outputs valid JSON.",
        target_schema_name: str = "JSONOutput",
    ) -> Dict[str, Any]:
        """Executes a completion enforcing JSON object output and returns parsed dictionary."""
        pass

    @abstractmethod
    def get_request_usage(self) -> Dict[str, Any]:
        """Returns aggregated token usage, call counts, and estimated cost for the current request."""
        pass

    @abstractmethod
    def reset_request_usage(self) -> None:
        """Resets the per-request token usage accumulator."""
        pass
