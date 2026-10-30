"""
Response models strictly conforming to Section 5 requirements and extended with
observability, coverage breakdown, source classification, and claim citations.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Allowed confidence levels for the synthesized research report."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SourceItem(BaseModel):
    """Represents a validated source actually retrieved and used during research."""
    title: str = Field(..., description="Title of the retrieved web source")
    url: str = Field(..., description="Direct URL of the retrieved web source")
    relevance: str = Field(..., description="Explanation of how this source contributes to the answer")
    category: Optional[str] = Field(default=None, description="Authority category tier of the source")
    is_official: Optional[bool] = Field(default=None, description="Whether this is an official documentation or repo source")


class TokenUsageSummary(BaseModel):
    """Token consumption and cost tracking across all LLM operations in a research request."""
    provider: str = Field(default="", description="Active LLM provider (e.g. groq, openai)")
    model: str = Field(default="", description="Model ID utilized for completions")
    llm_calls_count: int = Field(default=0, description="Number of distinct completion calls made")
    prompt_tokens: int = Field(default=0, description="Input tokens consumed across all calls")
    completion_tokens: int = Field(default=0, description="Output tokens generated across all calls")
    total_tokens: int = Field(default=0, description="Total tokens consumed across all calls")
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        description="Estimated API cost in USD based on official token pricing (None if pricing is unavailable)"
    )
    retries_count: int = Field(default=0, description="Count of retry attempts across calls")
    fallback_calls_count: int = Field(default=0, description="Count of fallback provider/model invocations")
    budget_cap_usd: float = Field(default=0.10, description="Configured per-request budget cap in USD")
    budget_remaining_usd: Optional[float] = Field(default=None, description="Remaining USD budget before hitting cap")


class ExecutionTrace(BaseModel):
    """Diagnostic metadata illustrating agent orchestration steps, multi-hop retrieval, and budget."""
    plan_objective: str = ""
    tasks_planned: List[str] = Field(default_factory=list)
    tools_executed: List[str] = Field(default_factory=list)
    search_queries_used: List[str] = Field(default_factory=list)
    pages_extracted_count: int = 0
    total_execution_time_seconds: float = 0.0
    llm_provider_used: str = Field(default="", description="Active LLM provider: groq or openai")
    llm_model_used: str = Field(default="", description="Active model name")
    retrieval_rounds: int = Field(default=1, description="Number of search and extraction rounds executed (max 2)")
    round_1_queries: List[str] = Field(default_factory=list, description="Initial discovery search queries")
    round_2_queries: List[str] = Field(default_factory=list, description="Targeted second-hop search queries for missing evidence")
    token_usage: Optional[TokenUsageSummary] = Field(
        default=None,
        description="Granular token consumption and estimated API cost summary"
    )


class ResearchResponse(BaseModel):
    """
    Validated research response conforming strictly to Section 5 schema:
    {
      "user_question": "",
      "short_answer": "",
      "key_findings": [],
      "sources_used": [
        {
          "title": "",
          "url": "",
          "relevance": ""
        }
      ],
      "confidence_level": "",
      "assumptions": [],
      "limitations": [],
      "suggested_next_steps": []
    }
    """
    user_question: str = Field(..., description="Original user research question")
    short_answer: str = Field(..., description="Concise, high-level executive summary")
    key_findings: List[str] = Field(..., description="List of granular, evidence-backed findings")
    sources_used: List[SourceItem] = Field(..., description="List of actual sources retrieved and utilized")
    confidence_level: str = Field(..., description="Assessed confidence level (HIGH, MEDIUM, LOW, INSUFFICIENT_EVIDENCE)")
    assumptions: List[str] = Field(..., description="Key assumptions made during analysis")
    limitations: List[str] = Field(..., description="Information gaps, unverified claims, or constraints")
    suggested_next_steps: List[str] = Field(..., description="Actionable follow-up investigations")

    # Optional enriched structured extensions
    confidence_explanation: Optional[str] = Field(
        default=None,
        description="Transparent mathematical rationale for the assigned confidence score"
    )
    evidence_coverage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed evidence coverage matrix across entities and dimensions"
    )
    source_classification: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Categorization of verified sources by authority tier"
    )
    claims_and_citations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Granular claim-level validation status and citations"
    )
    comparison_table: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured entity-by-dimension comparison matrix where applicable"
    )

    # Observability trace
    execution_trace: Optional[ExecutionTrace] = Field(
        default=None,
        description="Observability trace documenting plan, tools called, retrieval rounds, and execution timing"
    )
