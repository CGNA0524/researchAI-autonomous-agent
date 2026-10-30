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
   - Target entities: Preserve multi-word product/framework names as single units (e.g. 'Microsoft AutoGen' MUST remain 'Microsoft AutoGen', NEVER split into 'Microsoft' and 'AutoGen').
   - Instruction verbs (e.g. 'Evaluate', 'Compare', 'Analyze', 'Assess', 'Building') and company prefixes alone MUST NEVER be treated as target research entities.
   - Core evaluation dimensions: Retain and preserve ALL explicit evaluation dimensions requested by the user (e.g. Architecture, Orchestration, Tool Integration, Memory Management, Scalability, Error Handling, Limitations). Never drop or replace requested dimensions.
   - Specific user constraints (e.g. official documentation, startup context, recent developments).
2. For comparative tasks, generate search queries that systematically cover EACH entity and major evaluation dimension.
3. Keep queries concise, keyword-dense, and objective (e.g. 'LangGraph stateful graph architecture', 'Microsoft AutoGen multi-agent orchestration').
4. AVOID stale hardcoded years (do NOT use 2023 or 2024; reference {CURRENT_YEAR} only if time-sensitive).
5. Output MUST be valid JSON adhering strictly to this schema:
{{
  "original_question": "...",
  "primary_objective": "...",
  "target_entities": ["Entity 1", "Entity 2"],
  "evaluation_dimensions": ["Dimension 1", "Dimension 2"],
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

        is_software = any(w in user_question.lower() for w in [
            "framework", "frameworks", "library", "libraries", "api", "backend", "saas",
            "sdk", "code", "official documentation", "database", "orm", "fastapi", "django",
            "flask", "langchain", "llamaindex", "haystack", "langgraph", "crewai", "autogen"
        ])

        # Task 1: Primary entity technical overviews
        task1_queries: List[str] = []
        for entity in entities[:4]:
            if is_software:
                task1_queries.append(f"{entity} official documentation architecture")
            else:
                task1_queries.append(f"{entity} research overview {CURRENT_YEAR}")

        tasks.append(ResearchTask(
            task_id=task_id,
            objective=f"Retrieve foundational evidence and specifications for {', '.join(entities)}",
            search_queries=task1_queries,
            aspects_to_investigate=["Core architecture", "Foundational principles", "Primary capabilities"]
        ))
        task_id += 1

        # Task 2: Comparative analysis, key dimensions, and operational tradeoffs
        task2_queries: List[str] = []
        if len(entities) >= 2:
            if is_software:
                task2_queries.append(f"{' vs '.join(entities[:3])} comparison benchmarks")
                for entity in entities[:3]:
                    task2_queries.append(f"{entity} production limitations tradeoffs")
            else:
                task2_queries.append(f"{' '.join(entities[:2])} analysis {CURRENT_YEAR}")
                for dim in dimensions[:3]:
                    task2_queries.append(f"{entities[0]} {dim.lower()} {CURRENT_YEAR}")
        else:
            primary_entity = entities[0] if entities else "Primary Topic"
            if dimensions:
                for dim in dimensions[:3]:
                    task2_queries.append(f"{primary_entity} {dim.lower()} research")
            else:
                task2_queries.append(f"{primary_entity} analysis and challenges {CURRENT_YEAR}")

        tasks.append(ResearchTask(
            task_id=task_id,
            objective="Investigate requested evaluation dimensions, real-world implementations, and challenges",
            search_queries=task2_queries[:max(3, len(entities))],
            aspects_to_investigate=dimensions[:4] if dimensions else ["Performance", "Scalability", "Limitations"]
        ))

        return ResearchPlan(
            original_question=user_question,
            primary_objective=f"Investigate and synthesize authoritative findings for: '{user_question}'",
            needs_web_search=True,
            target_entities=entities,
            evaluation_dimensions=dimensions,
            tasks=tasks,
            initial_rationale="Deterministic fallback plan generated covering all target entities and major dimensions."
        )
