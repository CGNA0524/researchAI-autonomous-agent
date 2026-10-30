"""
Abstract Base Tool Interface for Agent Research Tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base class defining the contract for all agent tools."""

    name: str
    description: str

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the tool asynchronously and returns a structured dictionary result.
        Must handle exceptions internally and never raise uncaught errors to the agent.
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Returns the JSON Schema description of the tool for LLM tool-calling."""
        pass
