"""
Request models for API endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ResearchRequest(BaseModel):
    """Payload for submitting a research query to the agent."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The research question or topic to investigate",
        examples=["Compare the top 3 open-source vector databases for a startup building RAG products."]
    )
    max_search_results: Optional[int] = Field(
        default=None,
        ge=1,
        le=15,
        description="Optional override for the number of web search results per query"
    )
    max_pages_to_extract: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional override for the number of web pages to extract in detail"
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Research query cannot be empty or purely whitespace.")
        return trimmed
