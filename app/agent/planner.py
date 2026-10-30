"""
Research Planner component.
Analyzes user questions, decomposes them into entities and evaluation dimensions,
and formulates targeted search queries without stale hardcoded years.
"""

from datetime import datetime
from typing import Dict, Any, List, Union, Optional
from app.llm.base import BaseLLMClient
from app.llm.groq_client import GroqClient
from app.models.plan import ResearchPlan, ResearchTask
from app.agent.coverage import EvidenceCoverageValidator
from app.utils.logging import get_logger

logger = get_logger(__name__)

CURRENT_YEAR = datetime.now().year

PLANNER_SYSTEM_PROMPT = f"""You are an expert Research Planning Specialist in an Agentic AI system.
Current reference year: {CURRENT_YEAR}.
Your job is to analyze the user's research question, extract target entities and comparison dimensions, and construct a targeted, efficient research plan.

CRITICAL PLANNING GUIDELINES:
1. Deconstruct the research question to identify:
   - All target entities being researched or compared.
   - Core evaluation dimensions (e.g. Performance, Architecture, Async, Scalability, Deployment, Security, Limitations).
   - Specific user constraints (e.g. official documentation, startup context, recent developments).
2. For comparative tasks, generate search queries that systematically cover EACH entity and major evaluation dimension.
3. Keep queries concise, keyword-dense, and objective (e.g. 'FastAPI async architecture benchmarks', 'Django ORM performance scale').
4. AVOID stale hardcoded years (do NOT use 2023 or 2024; reference {CURRENT_YEAR} only if time-sensitive).
5. Output MUST be valid JSON adhering strictly to this schema:
{{
  "original_question": "...",
  "primary_objective": "...",
  "needs_web_search": true,
  "tasks": [
    {{
      "task_id": 1,
      "objective": "...",
      "search_queries": ["query 1", "query 2"],
      "aspects_to_investigate": ["aspect 1", "aspect 2"]
    }}
  ],
  "initial_rationale": "..."
}}
"""


class ResearchPlanner:
    """Deconstructs user questions into structured research tasks and queries."""

    def __init__(self, llm_client: Union[BaseLLMClient, GroqClient, Any]) -> None:
        self.llm = llm_client

    async def create_plan(self, user_question: str) -> ResearchPlan:
        """
        Creates a structured ResearchPlan for the user's research question.
        Falls back to a deterministic heuristic plan if LLM is unavailable or fails.
        """
        logger.info(f"Formulating research plan for: '{user_question}'")
        try:
            raw_json = await self.llm.generate_structured_json(
                messages=[{"role": "user", "content": f"Create a comprehensive research plan for: {user_question}"}],
                system_instruction=PLANNER_SYSTEM_PROMPT,
                target_schema_name="ResearchPlan"
            )
            plan = ResearchPlan.model_validate(raw_json)
            logger.info(f"Successfully generated research plan with {len(plan.tasks)} tasks.")
            return plan
        except Exception as e:
            logger.warning(f"LLM planning failed ({str(e)}). Generating deterministic fallback research plan.")
            return self._generate_fallback_plan(user_question)

    def _generate_fallback_plan(self, user_question: str) -> ResearchPlan:
        """
        Deterministic fallback planner ensuring system continuity even when LLM generation fails.
        Extracts entities and dimensions to formulate comprehensive search queries.
        """
        entities, dimensions = EvidenceCoverageValidator.extract_entities_and_dimensions(user_question)

        tasks: List[ResearchTask] = []
        task_id = 1

        # Task 1: Primary entity technical overviews
        task1_queries: List[str] = []
        for entity in entities[:3]:
            task1_queries.append(f"{entity} official documentation architecture")
            task1_queries.append(f"{entity} core features benchmarks")

        tasks.append(ResearchTask(
            task_id=task_id,
            objective=f"Retrieve official documentation and technical specifications for {', '.join(entities)}",
            search_queries=task1_queries[:4],
            aspects_to_investigate=["Core architecture", "Official documentation", "Primary capabilities"]
        ))
        task_id += 1

        # Task 2: Comparative benchmarks and operational tradeoffs
        task2_queries: List[str] = []
        if len(entities) >= 2:
            task2_queries.append(f"{' vs '.join(entities[:3])} performance comparison")
            task2_queries.append(f"{' '.join(entities[:2])} production limitations tradeoffs")
        else:
            task2_queries.append(f"{entities[0]} production limitations benchmarks {CURRENT_YEAR}")

        tasks.append(ResearchTask(
            task_id=task_id,
            objective="Investigate performance benchmarks, deployment complexity, and production limitations",
            search_queries=task2_queries[:3],
            aspects_to_investigate=dimensions[:3] if dimensions else ["Performance", "Scalability", "Limitations"]
        ))

        return ResearchPlan(
            original_question=user_question,
            primary_objective=f"Investigate and synthesize authoritative findings for: '{user_question}'",
            needs_web_search=True,
            tasks=tasks,
            initial_rationale="Deterministic fallback plan generated covering all target entities and major dimensions."
        )
