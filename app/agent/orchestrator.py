"""
Agent Orchestrator: Coordinates the multi-step research workflow,
tool selection, live execution, source authority filtering,
evidence coverage validation, intelligent second-hop retrieval, and synthesis.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from app.agent.planner import ResearchPlanner
from app.agent.filter import InformationFilter
from app.agent.coverage import EvidenceCoverageValidator, EvidenceCoverageReport
from app.agent.confidence import ConfidenceAssessor
from app.agent.synthesizer import AnswerSynthesizer
from app.config import get_settings
from app.llm.base import BaseLLMClient
from app.llm.factory import get_llm_client
from app.llm.groq_client import GroqClient
from app.models.plan import ResearchPlan, ToolExecutionRecord
from app.models.response import ResearchResponse, ExecutionTrace, TokenUsageSummary
from app.tools.registry import ToolRegistry
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Coordinates the end-to-end agentic research process:
    Plan -> Select Tools -> Round 1 Search & Extract -> Coverage Validation
    -> Optional Round 2 Second-Hop -> Calibrate Confidence -> Grounded Synthesis
    """

    def __init__(
        self,
        llm_client: Optional[Union[BaseLLMClient, GroqClient, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        groq_client: Optional[Any] = None,
    ) -> None:
        self.settings = get_settings()
        self.llm = llm_client or groq_client or get_llm_client()
        self.tools = tool_registry or ToolRegistry()
        self.planner = ResearchPlanner(self.llm)
        self.filter = InformationFilter()
        self.synthesizer = AnswerSynthesizer(self.llm)

    async def run(
        self,
        user_question: str,
        max_search_results: Optional[int] = None,
        max_pages_to_extract: Optional[int] = None
    ) -> ResearchResponse:
        """
        Executes the full autonomous research workflow with multi-hop retrieval and budget guarding.
        """
        start_time = time.time()
        logger.info(f"Starting research orchestration for query: '{user_question}'")

        if hasattr(self.llm, "reset_request_usage"):
            self.llm.reset_request_usage()

        tools_executed: List[str] = []
        search_queries_used: List[str] = []
        round_1_queries: List[str] = []
        round_2_queries: List[str] = []
        raw_search_results: List[Dict[str, Any]] = []
        extracted_pages: List[Dict[str, Any]] = []
        extracted_urls_set = set()

        # ----------------------------------------------------------------------
        # Phase 1: Question Understanding & Planning
        # ----------------------------------------------------------------------
        plan: ResearchPlan = await self.planner.create_plan(user_question)
        entities, dimensions = EvidenceCoverageValidator.extract_entities_and_dimensions(
            user_question, plan.tasks
        )
        logger.info(
            f"Research plan established. Needs web search: {plan.needs_web_search} | "
            f"Entities: {entities} | Dimensions: {dimensions}"
        )

        search_limit = max_search_results or self.settings.MAX_SEARCH_RESULTS
        extract_limit = max_pages_to_extract or self.settings.MAX_EXTRACT_PAGES

        # ----------------------------------------------------------------------
        # Phase 2: Round 1 Discovery Search & Execution
        # ----------------------------------------------------------------------
        if plan.needs_web_search:
            tools_executed.append("web_search")

            queries_to_run: List[str] = []
            for task in plan.tasks:
                for q in task.search_queries:
                    if q not in queries_to_run:
                        queries_to_run.append(q)

            round_1_queries = queries_to_run[:3] if queries_to_run else [user_question]
            search_queries_used.extend(round_1_queries)

            logger.info(f"Executing Round 1 web searches for {len(round_1_queries)} queries...")
            for query in round_1_queries:
                t0 = time.time()
                res = await self.tools.execute("web_search", query=query, max_results=search_limit)
                dur = (time.time() - t0) * 1000

                if res.get("success"):
                    items = res.get("results", [])
                    raw_search_results.extend(items)
                    logger.debug(f"Search query '{query}' yielded {len(items)} results in {dur:.1f}ms")
                else:
                    logger.warning(f"Search query '{query}' failed: {res.get('error')}")
                await asyncio.sleep(0.2)

        # ----------------------------------------------------------------------
        # Phase 3: Authority Filtering and Deduplication
        # ----------------------------------------------------------------------
        filtered_sources = self.filter.filter_and_deduplicate_search_results(
            raw_search_results,
            user_query=user_question,
            target_entities=entities,
            max_to_keep=8
        )

        # ----------------------------------------------------------------------
        # Phase 4: Round 1 Webpage Extraction
        # ----------------------------------------------------------------------
        urls_to_extract = self.filter.select_urls_for_extraction(
            filtered_sources,
            max_extract=extract_limit
        )

        if urls_to_extract:
            if "webpage_extractor" not in tools_executed:
                tools_executed.append("webpage_extractor")
            logger.info(f"Round 1: Selected {len(urls_to_extract)} priority URLs for deep text extraction.")

            extract_tasks = [
                self.tools.execute("webpage_extractor", url=url)
                for url in urls_to_extract
            ]
            extraction_results = await asyncio.gather(*extract_tasks, return_exceptions=True)

            for url, res in zip(urls_to_extract, extraction_results):
                extracted_urls_set.add(url)
                if isinstance(res, Exception):
                    logger.error(f"Extraction task threw unhandled exception for {url}: {res}")
                    extracted_pages.append({"success": False, "url": url, "error": str(res), "content": ""})
                elif isinstance(res, dict):
                    extracted_pages.append(res)

        # ----------------------------------------------------------------------
        # Phase 5: Consolidate Round 1 Evidence
        # ----------------------------------------------------------------------
        evidence_chunks: List[str] = []
        verified_sources_list: List[Dict[str, Any]] = []

        for page in extracted_pages:
            if page.get("success") and page.get("content"):
                url = page.get("url")
                title = page.get("title") or "Web Document"
                sanitized = self.filter.sanitize_evidence_text(page["content"], url)
                evidence_chunks.append(sanitized)
                # Lookup classification metadata
                matched_meta = next((s for s in filtered_sources if s.get("url") == url), {})
                verified_sources_list.append({
                    "title": title,
                    "url": url,
                    "category": matched_meta.get("category", "WEB_PAGE"),
                    "is_official": matched_meta.get("is_official", False)
                })

        for src in filtered_sources:
            url = src.get("url", "")
            title = src.get("title", "")
            snippet = src.get("snippet", "")
            if not any(v.get("url") == url for v in verified_sources_list):
                if snippet:
                    evidence_chunks.append(
                        self.filter.sanitize_evidence_text(f"Snippet: {snippet}", url)
                    )
                verified_sources_list.append({
                    "title": title,
                    "url": url,
                    "category": src.get("category", "WEB_SNIPPET"),
                    "is_official": src.get("is_official", False)
                })

        # ----------------------------------------------------------------------
        # Phase 6: Evidence Coverage Assessment
        # ----------------------------------------------------------------------
        coverage_report = EvidenceCoverageValidator.evaluate(
            user_question=user_question,
            evidence_chunks=evidence_chunks,
            verified_sources=verified_sources_list,
            entities=entities,
            dimensions=dimensions
        )
        logger.info(
            f"Round 1 Evidence Coverage: {int(coverage_report.coverage_ratio*100)}% "
            f"(Covered: {coverage_report.covered_cells}/{coverage_report.total_cells}, "
            f"Missing pairs: {len(coverage_report.missing_pairs)})"
        )

        retrieval_rounds = 1

        # ----------------------------------------------------------------------
        # Phase 7: Intelligent Second-Hop Retrieval (Max 2 Rounds)
        # ----------------------------------------------------------------------
        if plan.needs_web_search and (not coverage_report.is_sufficient or coverage_report.missing_entities):
            target_second_hop_queries = EvidenceCoverageValidator.generate_second_hop_queries(
                missing_pairs=coverage_report.missing_pairs,
                existing_queries=search_queries_used,
                max_queries=2
            )

            if target_second_hop_queries:
                retrieval_rounds = 2
                round_2_queries = target_second_hop_queries
                search_queries_used.extend(round_2_queries)
                logger.info(f"Triggering Second-Hop Retrieval for {len(round_2_queries)} missing evidence queries: {round_2_queries}")

                round_2_raw_results: List[Dict[str, Any]] = []
                for query in round_2_queries:
                    res = await self.tools.execute("web_search", query=query, max_results=search_limit)
                    if res.get("success"):
                        round_2_raw_results.extend(res.get("results", []))
                    await asyncio.sleep(0.2)

                if round_2_raw_results:
                    raw_search_results.extend(round_2_raw_results)
                    filtered_sources = self.filter.filter_and_deduplicate_search_results(
                        raw_search_results,
                        user_query=user_question,
                        target_entities=entities,
                        max_to_keep=10
                    )

                    # Extract up to 2 additional new URLs discovered in Round 2
                    new_urls_to_extract = [
                        s.get("url") for s in filtered_sources
                        if s.get("url") and s.get("url") not in extracted_urls_set
                    ][:2]

                    if new_urls_to_extract:
                        logger.info(f"Round 2: Extracting {len(new_urls_to_extract)} additional pages for missing gaps.")
                        r2_tasks = [self.tools.execute("webpage_extractor", url=u) for u in new_urls_to_extract]
                        r2_results = await asyncio.gather(*r2_tasks, return_exceptions=True)

                        for u, res in zip(new_urls_to_extract, r2_results):
                            extracted_urls_set.add(u)
                            if isinstance(res, dict) and res.get("success") and res.get("content"):
                                extracted_pages.append(res)
                                sanitized = self.filter.sanitize_evidence_text(res["content"], u)
                                evidence_chunks.append(sanitized)
                                matched_m = next((s for s in filtered_sources if s.get("url") == u), {})
                                verified_sources_list.append({
                                    "title": res.get("title") or "Web Document",
                                    "url": u,
                                    "category": matched_m.get("category", "OFFICIAL_DOCS"),
                                    "is_official": matched_m.get("is_official", True)
                                })

                    # Re-evaluate evidence coverage post second-hop
                    coverage_report = EvidenceCoverageValidator.evaluate(
                        user_question=user_question,
                        evidence_chunks=evidence_chunks,
                        verified_sources=verified_sources_list,
                        entities=entities,
                        dimensions=dimensions
                    )
                    logger.info(f"Post Second-Hop Evidence Coverage: {int(coverage_report.coverage_ratio*100)}%")

        # ----------------------------------------------------------------------
        # Phase 8: Programmatic Confidence and Limitation Assessment
        # ----------------------------------------------------------------------
        confidence, assumptions, limitations, confidence_explanation = ConfidenceAssessor.evaluate(
            search_results=filtered_sources,
            extracted_pages=extracted_pages,
            evidence_snippets=evidence_chunks,
            coverage_report=coverage_report,
            return_explanation=True
        )

        # ----------------------------------------------------------------------
        # Phase 9: Answer Synthesis & Citation Whitelisting
        # ----------------------------------------------------------------------
        full_corpus = "\n\n".join(evidence_chunks)
        response = await self.synthesizer.synthesize(
            user_question=user_question,
            evidence_corpus=full_corpus,
            verified_sources=verified_sources_list,
            calculated_confidence=confidence,
            assumptions=assumptions,
            limitations=limitations,
            confidence_explanation=confidence_explanation,
            coverage_report=coverage_report
        )

        # ----------------------------------------------------------------------
        # Phase 10: Token Usage & Observability Trace Construction
        # ----------------------------------------------------------------------
        usage_summary = None
        if hasattr(self.llm, "get_request_usage"):
            raw_usage = self.llm.get_request_usage()
            usage_summary = TokenUsageSummary(
                provider=raw_usage.get("provider", "unknown"),
                model=raw_usage.get("model", "unknown"),
                llm_calls_count=raw_usage.get("llm_calls_count", 0),
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
                estimated_cost_usd=raw_usage.get("estimated_cost_usd"),
                retries_count=raw_usage.get("retries_count", 0),
                fallback_calls_count=raw_usage.get("fallback_calls_count", 0),
                budget_cap_usd=raw_usage.get("budget_cap_usd", self.settings.MAX_REQUEST_BUDGET_USD),
                budget_remaining_usd=raw_usage.get("budget_remaining_usd")
            )

        provider_name = getattr(self.llm, "provider_name", "unknown")
        active_model = getattr(self.llm, "active_model", "unknown")

        total_elapsed = round(time.time() - start_time, 2)
        response.execution_trace = ExecutionTrace(
            plan_objective=plan.primary_objective,
            tasks_planned=[t.objective for t in plan.tasks],
            tools_executed=tools_executed,
            search_queries_used=search_queries_used,
            pages_extracted_count=len([p for p in extracted_pages if p.get("success")]),
            total_execution_time_seconds=total_elapsed,
            llm_provider_used=provider_name,
            llm_model_used=active_model,
            retrieval_rounds=retrieval_rounds,
            round_1_queries=round_1_queries,
            round_2_queries=round_2_queries,
            token_usage=usage_summary
        )

        logger.info(
            f"Research orchestration finished in {total_elapsed}s. "
            f"Confidence: {response.confidence_level}, Sources: {len(response.sources_used)}, "
            f"Retrieval Rounds: {retrieval_rounds}"
        )
        return response
