"""
Comprehensive Test Suite for ResearchAI Production Upgrades.
Covers:
- Source Classification & Authority Scoring (Phase 1)
- Evidence Coverage Validation & Matrix (Phase 2)
- Second-Hop Retrieval Query Generation (Phase 3)
- Calibrated Programmatic Confidence Scoring (Phase 4)
- Claim-Level Citation Validation & Benchmark Qualification (Phase 5)
- Dynamic Planner (Phase 6)
- Pre-Call Budget Guard & Estimation (Phase 8)
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agent.source_classifier import SourceClassifier, SourceCategory
from app.agent.coverage import EvidenceCoverageValidator, EvidenceCoverageReport
from app.agent.citation import ClaimCitationValidator
from app.agent.confidence import ConfidenceAssessor
from app.agent.planner import ResearchPlanner
from app.models.response import ConfidenceLevel
from app.llm.base import BudgetExceededError
from app.llm.openai_client import OpenAIClient


# ==============================================================================
# PHASE 1: Source Classification & Authority Filtering Tests
# ==============================================================================

def test_source_classifier_official_documentation():
    entities = ["FastAPI", "Django"]
    
    # Official docs
    res1 = SourceClassifier.classify("https://fastapi.tiangolo.com/tutorial/", target_entities=entities)
    assert res1.category == SourceCategory.OFFICIAL_DOCS
    assert res1.is_official is True
    assert res1.authority_score >= 4.0

    res2 = SourceClassifier.classify("https://docs.djangoproject.com/en/5.0/topics/async/", target_entities=entities)
    assert res2.category == SourceCategory.OFFICIAL_DOCS
    assert res2.is_official is True


def test_source_classifier_academic_and_reputable():
    # Academic
    res_acad = SourceClassifier.classify("https://arxiv.org/abs/2301.00001")
    assert res_acad.category == SourceCategory.ACADEMIC_PAPER
    assert res_acad.authority_score >= 3.0

    # Reputable publication
    res_rep = SourceClassifier.classify("https://martinfowler.com/articles/microservices.html")
    assert res_rep.category == SourceCategory.REPUTABLE_PUBLICATION
    assert res_rep.authority_score >= 2.0


def test_source_classifier_low_quality_and_github():
    entities = ["FastAPI"]

    # Official repo
    res_repo = SourceClassifier.classify("https://github.com/tiangolo/fastapi", target_entities=entities)
    assert res_repo.category == SourceCategory.OFFICIAL_REPO
    assert res_repo.is_official is True

    # General / Unrelated repo
    res_gen = SourceClassifier.classify("https://github.com/randomuser/my-toy-script", target_entities=entities)
    assert res_gen.category == SourceCategory.GENERAL_REPO
    assert res_gen.is_official is False
    assert res_gen.authority_score <= 1.0

    # Low-quality spam / junk
    res_low = SourceClassifier.classify("https://quora.com/what-is-python", snippet="sign in to continue captcha")
    assert res_low.category == SourceCategory.LOW_QUALITY
    assert res_low.authority_score == 0.0


# ==============================================================================
# PHASE 2: Evidence Coverage Validator & Matrix Tests
# ==============================================================================

def test_evidence_coverage_matrix_evaluation():
    entities = ["FastAPI", "Django"]
    dimensions = ["Performance", "Async Support"]
    evidence = [
        "FastAPI exhibits high performance and throughput in standard benchmarks.",
        "FastAPI natively supports async and await coroutines with uvicorn event loop."
    ]
    sources = [{"url": "https://fastapi.tiangolo.com", "title": "FastAPI Docs"}]
    
    report = EvidenceCoverageValidator.evaluate(
        user_question="Compare FastAPI vs Django",
        evidence_chunks=evidence,
        verified_sources=sources,
        entities=entities,
        dimensions=dimensions
    )
    
    assert report.total_cells == 4
    assert (report.covered_cells + report.partial_cells) >= 2
    # Django has no evidence
    assert "Django" in report.missing_entities
    assert report.coverage_ratio < 1.0


def test_evidence_coverage_all_covered():
    entities = ["FastAPI"]
    dimensions = ["Performance"]
    evidence = [
        "FastAPI performance benchmark shows high throughput with Uvicorn async event loop."
    ]
    sources = [{"url": "https://fastapi.tiangolo.com", "title": "FastAPI Performance"}]
    
    report = EvidenceCoverageValidator.evaluate(
        user_question="FastAPI Performance",
        evidence_chunks=evidence,
        verified_sources=sources,
        entities=entities,
        dimensions=dimensions
    )
    
    assert report.coverage_ratio == 1.0
    assert len(report.missing_entities) == 0
    assert len(report.missing_dimensions) == 0


# ==============================================================================
# PHASE 3: Second-Hop Query Generation Tests
# ==============================================================================

def test_second_hop_query_generation():
    missing_pairs = [("Django", "Performance"), ("Django", "Deployment")]
    
    queries = EvidenceCoverageValidator.generate_second_hop_queries(
        missing_pairs=missing_pairs,
        existing_queries=["FastAPI vs Django overview"],
        max_queries=2
    )
    
    assert len(queries) > 0
    # Queries must target the missing entity and dimension
    assert any("Django" in q for q in queries)
    # Must not duplicate existing queries
    assert "FastAPI vs Django overview" not in queries


# ==============================================================================
# PHASE 4: Deterministic Programmatic Confidence Tests
# ==============================================================================

def test_confidence_all_four_levels():
    # 1. INSUFFICIENT_EVIDENCE (0 sources)
    conf_insuf, _, _, expl_insuf = ConfidenceAssessor.evaluate(
        search_results=[],
        extracted_pages=[],
        evidence_snippets=[],
        return_explanation=True
    )
    assert conf_insuf == ConfidenceLevel.INSUFFICIENT_EVIDENCE.value
    assert "INSUFFICIENT_EVIDENCE" in expl_insuf

    # 2. LOW (sparse evidence, missing entities)
    sparse_coverage = EvidenceCoverageReport(
        total_cells=4, covered_cells=1, coverage_ratio=0.25,
        missing_entities=["Qdrant"], missing_dimensions=["Scalability"]
    )
    conf_low, _, _, expl_low = ConfidenceAssessor.evaluate(
        search_results=[{"url": "https://randomblog.com/post", "category": "COMMUNITY_ARTICLE"}],
        extracted_pages=[],
        evidence_snippets=["Short mention"],
        coverage_report=sparse_coverage,
        return_explanation=True
    )
    assert conf_low == ConfidenceLevel.LOW.value
    assert "LOW" in expl_low

    # 3. MEDIUM (reasonable sources, some missing dimensions)
    med_coverage = EvidenceCoverageReport(
        total_cells=4, covered_cells=2, coverage_ratio=0.50,
        missing_entities=[], missing_dimensions=["Pricing"]
    )
    conf_med, _, _, expl_med = ConfidenceAssessor.evaluate(
        search_results=[
            {"url": "https://fastapi.tiangolo.com", "category": "OFFICIAL_DOCS", "is_official": True},
            {"url": "https://docs.djangoproject.com", "category": "OFFICIAL_DOCS", "is_official": True}
        ],
        extracted_pages=[{"success": True, "url": "https://fastapi.tiangolo.com"}],
        evidence_snippets=["Detailed evidence " * 80],
        coverage_report=med_coverage,
        return_explanation=True
    )
    assert conf_med == ConfidenceLevel.MEDIUM.value
    assert "MEDIUM" in expl_med

    # 4. HIGH (authoritative sources, deep extractions, high coverage, zero missing entities)
    high_coverage = EvidenceCoverageReport(
        total_cells=4, covered_cells=4, coverage_ratio=1.0,
        missing_entities=[], missing_dimensions=[]
    )
    conf_high, _, _, expl_high = ConfidenceAssessor.evaluate(
        search_results=[
            {"url": "https://fastapi.tiangolo.com", "category": "OFFICIAL_DOCS", "is_official": True},
            {"url": "https://docs.djangoproject.com", "category": "OFFICIAL_DOCS", "is_official": True},
            {"url": "https://flask.palletsprojects.com", "category": "OFFICIAL_DOCS", "is_official": True}
        ],
        extracted_pages=[
            {"success": True, "url": "https://fastapi.tiangolo.com"},
            {"success": True, "url": "https://docs.djangoproject.com"}
        ],
        evidence_snippets=["Comprehensive technical documentation and verified benchmark details " * 80],
        coverage_report=high_coverage,
        return_explanation=True
    )
    assert conf_high == ConfidenceLevel.HIGH.value
    assert "HIGH" in expl_high


def test_confidence_penalizes_contradictions():
    high_coverage = EvidenceCoverageReport(
        total_cells=4, covered_cells=4, coverage_ratio=1.0,
        missing_entities=[], missing_dimensions=[]
    )
    # With unresolved contradictions, confidence must be downgraded from HIGH
    conf, _, lims, _ = ConfidenceAssessor.evaluate(
        search_results=[
            {"url": "https://a.com", "category": "OFFICIAL_DOCS", "is_official": True},
            {"url": "https://b.com", "category": "OFFICIAL_DOCS", "is_official": True}
        ],
        extracted_pages=[{"success": True, "url": "https://a.com"}],
        evidence_snippets=["Evidence content " * 100],
        coverage_report=high_coverage,
        contradictions=["Source A claims throughput is 50k req/s while Source B reports 5k req/s."],
        return_explanation=True
    )
    assert conf != ConfidenceLevel.HIGH.value
    assert any("Contradictory" in l for l in lims)


# ==============================================================================
# PHASE 5: Claim-Level Citation Validation Tests
# ==============================================================================

def test_claim_citation_and_unverified_flagging():
    validator = ClaimCitationValidator()
    verified_sources = [
        {"title": "FastAPI Tutorial Documentation", "url": "https://fastapi.tiangolo.com/tutorial/"},
        {"title": "Django Async Support", "url": "https://docs.djangoproject.com/en/5.0/topics/async/"}
    ]
    corpus = "FastAPI is built on Starlette and Pydantic with asynchronous routing."

    findings = [
        "FastAPI is built on Starlette and Pydantic.",
        "Completely fabricated claim about quantum computing integration."
    ]

    sanitized, validated, contradictions = validator.validate_findings(
        findings=findings,
        verified_sources=verified_sources,
        evidence_corpus=corpus
    )

    assert len(validated) == 2
    # First claim is grounded
    assert validated[0].is_fully_grounded is True
    assert validated[0].claim_type == "FACT"

    # Second claim has low overlap in corpus -> flagged as unverified
    assert validated[1].is_fully_grounded is False
    assert "[Unverified Observation]" in sanitized[1]


def test_claim_category_and_numerical_benchmark_qualification():
    validator = ClaimCitationValidator()
    verified_sources = [{"title": "FastAPI Benchmarks", "url": "https://fastapi.tiangolo.com/benchmarks/"}]
    # Corpus lacks test hardware conditions
    corpus = "FastAPI achieves 80,000 req/s under peak load."

    findings = ["FastAPI achieves 80,000 req/s under peak load."]
    sanitized, validated, contradictions = validator.validate_findings(
        findings=findings,
        verified_sources=verified_sources,
        evidence_corpus=corpus
    )

    assert len(validated) == 1
    assert validated[0].claim_type == "BENCHMARK"
    assert validated[0].has_numerical_benchmark is True
    # Should qualify the benchmark with note
    assert "Benchmark figures reflect specific vendor or test conditions" in sanitized[0]


# ==============================================================================
# PHASE 6: Dynamic Planner Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_dynamic_planner_extracts_entities_and_current_year():
    mock_llm = MagicMock()
    mock_llm.generate_structured_json = AsyncMock(return_value={
        "original_question": "Compare FastAPI vs Django in 2026",
        "primary_objective": "Comparative benchmark study",
        "needs_web_search": True,
        "tasks": [
            {
                "task_id": 1,
                "objective": "Investigate FastAPI async performance",
                "search_queries": ["FastAPI async benchmarks 2026"],
                "aspects_to_investigate": ["performance", "async"]
            },
            {
                "task_id": 2,
                "objective": "Investigate Django async performance",
                "search_queries": ["Django async architecture benchmarks"],
                "aspects_to_investigate": ["performance", "architecture"]
            }
        ],
        "initial_rationale": "Plan covers both target frameworks across core dimensions."
    })

    planner = ResearchPlanner(llm_client=mock_llm)
    plan = await planner.create_plan("Compare FastAPI vs Django in 2026")

    assert len(plan.tasks) == 2
    assert plan.tasks[0].task_id == 1
    assert len(plan.tasks[0].search_queries) == 1


# ==============================================================================
# PHASE 8: Pre-Call Budget Guard Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_pre_call_budget_guard_blocks_when_cap_exceeded():
    client = OpenAIClient(api_key="sk-mock-key", model="gpt-4o-mini")
    # Simulate already accumulated $0.098 of cost ($0.10 cap)
    client.total_cost_usd = 0.098

    # Attempting a large call with 1.25x margin will exceed the $0.10 limit
    with pytest.raises(BudgetExceededError) as exc_info:
        await client.chat_completion(
            messages=[{"role": "user", "content": "Generate a comprehensive analysis " * 500}],
            max_tokens=2048
        )
    assert "Pre-call budget guard blocked completion" in str(exc_info.value)


def test_estimate_call_cost():
    client = OpenAIClient(api_key="sk-mock-key", model="gpt-4o-mini")
    # Test estimation returns non-zero positive float
    estimated = client.estimate_call_cost(prompt_chars=4000, max_tokens=1000)
    assert estimated is not None
    assert estimated > 0.0
