"""
LLM Provider Factory.
Instantiates the configured LLM client (Groq or OpenAI) with explicit logging
and validation of allowed providers.
"""

from typing import Optional
from app.config import get_settings
from app.llm.base import BaseLLMClient
from app.llm.groq_client import GroqClient
from app.llm.openai_client import OpenAIClient
from app.utils.logging import get_logger

logger = get_logger(__name__)


def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    """
    Returns an instance of BaseLLMClient based on the specified provider or environment setting.
    Allowed providers: 'groq', 'openai'.
    Explicitly logs provider selection and never silently switches providers without logging.
    """
    settings = get_settings()
    active_provider = (provider or settings.LLM_PROVIDER).strip().lower()

    if active_provider == "openai":
        logger.info(
            f"Active LLM Provider selected: 'openai' | Model: '{settings.OPENAI_MODEL}' "
            f"| Max Budget: ${settings.MAX_REQUEST_BUDGET_USD:.2f}"
        )
        return OpenAIClient(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)

    elif active_provider == "groq":
        logger.info(
            f"Active LLM Provider selected: 'groq' | Primary: '{settings.GROQ_MODEL}' "
            f"| Fallback: '{settings.GROQ_FALLBACK_MODEL}'"
        )
        return GroqClient()

    else:
        logger.error(f"Invalid LLM_PROVIDER specified: '{active_provider}'")
        raise ValueError(
            f"Unsupported LLM provider '{active_provider}'. "
            f"Allowed providers in ResearchAI are: 'groq', 'openai'."
        )
