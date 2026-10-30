"""
Comprehensive Research-Quality Regression Test Suite.
Validates fixes for source relevance, entity grounding, second-hop prioritization,
confidence calibration, source-count consistency, and citation accuracy.
"""

import pytest
from app.agent.source_classifier import SourceClassifier, SourceCategory
from app.agent.filter import InformationFilter
from app.agent.coverage import EvidenceCoverageValidator, EvidenceCoverageReport
from app.agent.confidence import ConfidenceAssessor
from app.agent.citation import ClaimCitationValidator
from app.agent.planner import ResearchPlanner
from app.models.response import ConfidenceLevel, ResearchResponse, SourceItem, ExecutionTrace


# ==============================================================================
# 1. Source Relevance & Fake Official GitHub Filtering
# ==============================================================================

def test_unrelated_github_repository_not_classified_as_official():
    """
    Ensure third-party or arbitrary user GitHub projects mentioning a framework
    (such as DataBird-Savvy/AI_ChatbotSlackwith-N8N) are NOT classified as OFFICIAL_REPO.
    """
    classified = SourceClassifier.classify(
        url="https://github.com/DataBird-Savvy/AI_ChatbotSlackwith-N8N",
        title="AI Chatbot Slack with LangChain and N8N",
        snippet="A personal demo integrating LangChain with Slack and N8N workflow automations.",
        target_entities=["LangChain", "LlamaIndex", "Haystack"]
    )
    assert classified.category == SourceCategory.GENERAL_REPO
    assert classified.is_official is False

    # Also test arbitrary user repo
    classified_user = SourceClassifier.classify(
        url="https://github.com/johndoe/my-langchain-rag-app",
        title="My LangChain App",
        snippet="Testing LangChain retrieval pipelines.",
        target_entities=["LangChain"]
    )
    assert classified_user.category == SourceCategory.GENERAL_REPO
    assert classified_user.is_official is False


def test_verified_official_github_org_classified_as_official():
    """
    Ensure verified official organizations (e.g. langchain-ai, run-llama, deepset-ai)
    are correctly recognized as OFFICIAL_REPO with is_official=True.
    """
    classified_lc = SourceClassifier.classify(
        url="https://github.com/langchain-ai/langchain",
        title="langchain-ai/langchain: Build context-aware LLM applications",
        snippet="Official LangChain repository for Python and TypeScript.",
        target_entities=["LangChain", "LlamaIndex", "Haystack"]
    )
    assert classified_lc.category == SourceCategory.OFFICIAL_REPO
    assert classified_lc.is_official is True
    assert classified_lc.matched_entity == "LangChain"

    classified_llama = SourceClassifier.classify(
        url="https://github.com/run-llama/llama_index",
        title="run-llama/llama_index: Data framework for LLM applications",
        snippet="Official LlamaIndex repository.",
        target_entities=["LangChain", "LlamaIndex", "Haystack"]
    )
    assert classified_llama.category == SourceCategory.OFFICIAL_REPO
    assert classified_llama.is_official is True
    assert classified_llama.matched_entity == "LlamaIndex"

    classified_haystack = SourceClassifier.classify(
        url="https://github.com/deepset-ai/haystack",
        title="deepset-ai/haystack: LLM orchestration framework",
        snippet="Official Haystack repository by deepset.",
        target_entities=["LangChain", "LlamaIndex", "Haystack"]
    )
    assert classified_haystack.category == SourceCategory.OFFICIAL_REPO
    assert classified_haystack.is_official is True
    assert classified_haystack.matched_entity == "Haystack"


# ==============================================================================
# 2. Entity Grounding in Search Filtering (No Milvus in Haystack/LangChain)
# ==============================================================================

def test_unrelated_entity_source_filtered_out():
    """
    Verify that an unrelated entity article (e.g. Milvus Wikipedia page)
    is completely filtered out when researching LangChain, LlamaIndex, and Haystack.
    """
    raw_results = [
        {
            "url": "https://python.langchain.com/docs/introduction",
            "title": "LangChain Official Documentation",
            "snippet": "Architecture and RAG building guide for LangChain applications.",
            "domain": "python.langchain.com"
        },
        {
            "url": "https://en.wikipedia.org/wiki/Milvus_(vector_database)",
            "title": "Milvus - Wikipedia",
            "snippet": "Milvus is an open-source vector database built to power embedding similarity search and AI applications.",
            "domain": "en.wikipedia.org"
        },
        {
            "url": "https://docs.llamaindex.ai/en/stable/",
            "title": "LlamaIndex Official Documentation",
            "snippet": "Index structures and retrieval engines for LlamaIndex RAG systems.",
            "domain": "docs.llamaindex.ai"
        }
    ]

    filtered = InformationFilter.filter_and_deduplicate_search_results(
        results=raw_results,
        user_query="Compare LangChain, LlamaIndex, and Haystack for building production-ready RAG applications.",
        target_entities=["LangChain", "LlamaIndex", "Haystack"]
    )

    urls = [item["url"] for item in filtered]
    # LangChain and LlamaIndex must be preserved
    assert "https://python.langchain.com/docs/introduction" in urls
    assert "https://docs.llamaindex.ai/en/stable/" in urls
    # Milvus Wikipedia must be dropped because it mentions none of the 3 researched entities
    assert "https://en.wikipedia.org/wiki/Milvus_(vector_database)" not in urls


# ==============================================================================
# 3. Official Documentation Preference & Wikipedia Penalization
# ==============================================================================

def test_official_documentation_preferred_over_wikipedia():
    """
    Verify that when user specifies 'Prefer official documentation',
    official documentation domains receive a heavy boost and Wikipedia is penalized.
    """
    raw_results = [
        {
            "url": "https://en.wikipedia.org/wiki/LangChain",
            "title": "LangChain - Wikipedia",
            "snippet": "LangChain is a framework designed to simplify the creation of applications using LLMs.",
            "domain": "en.wikipedia.org"
        },
        {
            "url": "https://python.langchain.com/docs/concepts",
            "title": "LangChain Architecture Concepts",
            "snippet": "Official LangChain architectural overview and production deployment guides.",
            "domain": "python.langchain.com"
        }
    ]

    filtered = InformationFilter.filter_and_deduplicate_search_results(
        results=raw_results,
        user_query="Compare LangChain and Haystack. Prefer official documentation.",
        target_entities=["LangChain", "Haystack"]
    )

    assert len(filtered) == 2
    # Official doc must rank first with higher score
    assert filtered[0]["url"] == "https://python.langchain.com/docs/concepts"
    assert filtered[0]["is_official"] is True
    assert filtered[0]["score"] > filtered[1]["score"]


# ==============================================================================
# 4. Coverage-Driven Retrieval & Missing Entity Prioritization
# ==============================================================================

def test_second_hop_retrieval_prioritizes_completely_missing_entity():
    """
    When one entity (e.g. LlamaIndex) is completely missing across all dimensions
    while another entity (e.g. LangChain) is partially covered, second-hop retrieval
    MUST prioritize the missing entity and interleave queries rather than starving it.
    """
    missing_pairs = [
        ("LangChain", "Performance"),
        ("LangChain", "Limitations"),
        ("LlamaIndex", "Architecture"),
        ("LlamaIndex", "Integrations"),
        ("LlamaIndex", "Scalability"),
        ("LlamaIndex", "Limitations"),
    ]

    second_hop = EvidenceCoverageValidator.generate_second_hop_queries(
        missing_pairs=missing_pairs,
        existing_queries=["LangChain architecture overview", "Haystack pipeline benchmarks"],
        max_queries=3,
        missing_entities=["LlamaIndex"]
    )

    assert len(second_hop) == 3
    # Top priority must target LlamaIndex
    assert "LlamaIndex" in second_hop[0]
    # LlamaIndex must not be starved
    llamaindex_queries = [q for q in second_hop if "LlamaIndex" in q]
    assert len(llamaindex_queries) >= 2


def test_round_robin_url_extraction_balances_across_entities():
    """
    Verify select_urls_for_extraction uses round-robin to ensure each entity
    gets webpage extraction representation instead of monopolizing all slots.
    """
    filtered_results = [
        {"url": "https://python.langchain.com/docs/1", "matched_entity": "LangChain", "title": "LangChain 1", "snippet": "LangChain"},
        {"url": "https://python.langchain.com/docs/2", "matched_entity": "LangChain", "title": "LangChain 2", "snippet": "LangChain"},
        {"url": "https://docs.llamaindex.ai/1", "matched_entity": "LlamaIndex", "title": "LlamaIndex 1", "snippet": "LlamaIndex"},
        {"url": "https://haystack.deepset.ai/1", "matched_entity": "Haystack", "title": "Haystack 1", "snippet": "Haystack"},
    ]

    selected = InformationFilter.select_urls_for_extraction(
        filtered_search_results=filtered_results,
        max_extract=3,
        target_entities=["LangChain", "LlamaIndex", "Haystack"]
    )

    assert len(selected) == 3
    # Must contain one URL from each of the three entities
    assert "https://python.langchain.com/docs/1" in selected
    assert "https://docs.llamaindex.ai/1" in selected
    assert "https://haystack.deepset.ai/1" in selected


# ==============================================================================
# 5. Confidence Calibration & Threshold Consistency
# ==============================================================================

def test_confidence_calibration_score_cannot_return_low_without_cap():
    """
    Audit scoring threshold mapping:
    A score >= 45 cannot return LOW unless capped by a documented hard safety condition.
    When coverage is inadequate (< 40%) or an entity is missing, score is capped <= 44.0
    and the override is transparently explained.
    """
    # Case A: Severe coverage gap (30.6%) with missing entity LlamaIndex
    coverage_report = EvidenceCoverageReport(
        total_cells=18,
        covered_cells=5,
        partial_cells=1,
        coverage_ratio=0.306,
        missing_entities=["LlamaIndex"],
        missing_dimensions=["Limitations"]
    )

    conf, _, _, expl = ConfidenceAssessor.evaluate(
        search_results=[
            {"url": "https://python.langchain.com", "category": "OFFICIAL_DOCS", "is_official": True},
            {"url": "https://haystack.deepset.ai", "category": "OFFICIAL_DOCS", "is_official": True}
        ],
        extracted_pages=[{"success": True, "url": "https://python.langchain.com"}],
        evidence_snippets=["Evidence text corpus " * 50],
        coverage_report=coverage_report,
        return_explanation=True
    )

    assert conf == ConfidenceLevel.LOW.value
    # Explanation must state LOW, the score must be capped <= 44.0, and override must be explained
    assert "LOW" in expl
    assert "Hard Safety Override" in expl
    assert "LlamaIndex" in expl or "coverage" in expl


def test_confidence_medium_when_all_entities_covered_with_moderate_score():
    """
    Verify that when all entities are covered with coverage >= 40% and score >= 45,
    MEDIUM is consistently returned without erroneous downgrade.
    """
    coverage_report = EvidenceCoverageReport(
        total_cells=6,
        covered_cells=3,
        partial_cells=1,
        coverage_ratio=0.58,
        missing_entities=[],
        missing_dimensions=["Limitations"]
    )

    conf, _, _, expl = ConfidenceAssessor.evaluate(
        search_results=[
            {"url": "https://python.langchain.com", "category": "OFFICIAL_DOCS", "is_official": True},
            {"url": "https://docs.llamaindex.ai", "category": "OFFICIAL_DOCS", "is_official": True},
            {"url": "https://haystack.deepset.ai", "category": "OFFICIAL_DOCS", "is_official": True}
        ],
        extracted_pages=[{"success": True, "url": "https://python.langchain.com"}],
        evidence_snippets=["Comprehensive text " * 60],
        coverage_report=coverage_report,
        return_explanation=True
    )

    assert conf == ConfidenceLevel.MEDIUM.value
    assert "MEDIUM" in expl


# ==============================================================================
# 6. Source Count Consistency
# ==============================================================================

def test_source_count_consistency_between_verified_sources_and_explanation():
    """
    Verify that the verified sources count passed to evaluate matches the count
    reported in the confidence explanation and the UI metric.
    """
    verified_sources = [
        {"title": "LangChain Docs", "url": "https://python.langchain.com", "category": "OFFICIAL_DOCS", "is_official": True},
        {"title": "Haystack Docs", "url": "https://haystack.deepset.ai", "category": "OFFICIAL_DOCS", "is_official": True}
    ]

    conf, _, _, expl = ConfidenceAssessor.evaluate(
        search_results=verified_sources,
        extracted_pages=[{"success": True, "url": "https://python.langchain.com"}],
        evidence_snippets=["Text"],
        return_explanation=True,
        discovered_sources_count=8
    )

    # Explanation must report exactly 2 verified sources, not 8 discovered search results
    assert "2 verified source" in expl


# ==============================================================================
# 7. Citation Grounding: Claim Entity Relevance
# ==============================================================================

def test_citation_grounding_never_associates_unrelated_entity_with_framework_claim():
    """
    Verify that a claim specifically discussing Haystack is NEVER supported by
    a source about Milvus or LangChain if the source doesn't mention Haystack.
    """
    verified_sources = [
        {
            "title": "Haystack Official Pipeline Architecture Guide",
            "url": "https://haystack.deepset.ai/docs/pipelines",
            "snippet": "Haystack pipelines use explicit DAG routing for RAG applications."
        },
        {
            "title": "Milvus Vector Database for Production Applications - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Milvus",
            "snippet": "Milvus is a distributed vector database designed for high scale embedding similarity."
        }
    ]

    evidence_corpus = (
        "Haystack pipelines use explicit DAG routing for RAG applications. "
        "Milvus is a distributed vector database designed for high scale embedding similarity."
    )

    findings = [
        "Haystack provides an explicit DAG pipeline architecture for production RAG systems.",
        "Milvus scales to billions of vectors for high-throughput similarity indexing."
    ]

    _, validated_claims, _ = ClaimCitationValidator.validate_findings(
        findings=findings,
        verified_sources=verified_sources,
        evidence_corpus=evidence_corpus
    )

    haystack_claim = validated_claims[0]
    milvus_claim = validated_claims[1]

    # Haystack claim must ONLY cite Haystack URL, never Milvus Wikipedia
    assert "https://haystack.deepset.ai/docs/pipelines" in haystack_claim.supporting_urls
    assert "https://en.wikipedia.org/wiki/Milvus" not in haystack_claim.supporting_urls

    # Milvus claim must cite Milvus URL, never Haystack
    assert "https://en.wikipedia.org/wiki/Milvus" in milvus_claim.supporting_urls
    assert "https://haystack.deepset.ai/docs/pipelines" not in milvus_claim.supporting_urls


def test_three_entity_comparison_fallback_plan_does_not_drop_third_entity():
    """
    Verify fallback planner does not truncate the 3rd entity in multi-entity comparison.
    """
    planner = ResearchPlanner(llm_client=None)
    plan = planner._generate_fallback_plan(
        "Compare LangChain, LlamaIndex, and Haystack for building production-ready RAG applications."
    )

    task1 = plan.tasks[0]
    all_queries_text = " ".join(task1.search_queries)
    assert "LangChain" in all_queries_text
    assert "LlamaIndex" in all_queries_text
    assert "Haystack" in all_queries_text


# ==============================================================================
# REGRESSION SUITE: MULTI-AGENT COMPARISON & RESEARCH QUALITY
# Query: "Compare LangGraph, CrewAI, and Microsoft AutoGen for building
# production-ready multi-agent systems. Evaluate architecture, orchestration,
# tool integration, memory management, scalability, error handling, and limitations."
# ==============================================================================

TEST_QUERY = (
    "Compare LangGraph, CrewAI, and Microsoft AutoGen for building production-ready multi-agent systems. "
    "Evaluate architecture, orchestration, tool integration, memory management, scalability, error handling, and limitations."
)


def test_microsoft_autogen_entity_preservation():
    """
    REGRESSION TEST 1: Microsoft AutoGen entity preservation.
    'Microsoft AutoGen' must remain one entity and must not be split into 'Microsoft' and 'AutoGen'.
    Exactly 3 entities must be extracted: LangGraph, CrewAI, Microsoft AutoGen.
    """
    entities, _ = EvidenceCoverageValidator.extract_entities_and_dimensions(TEST_QUERY)

    assert "Microsoft AutoGen" in entities, "Microsoft AutoGen must be preserved as a single entity."
    assert "Microsoft" not in entities, "'Microsoft' must never be extracted as an isolated entity."
    assert "AutoGen" not in entities, "'AutoGen' must not be extracted separately when Microsoft AutoGen is present."
    assert "LangGraph" in entities
    assert "CrewAI" in entities
    assert len(entities) == 3, f"Expected exactly 3 entities, got {entities}"
    assert entities == ["LangGraph", "CrewAI", "Microsoft AutoGen"]


def test_evaluate_not_classified_as_an_entity():
    """
    REGRESSION TEST 2: Evaluate not classified as an entity.
    Instruction verbs ('Evaluate', 'Compare') must never become research entities.
    """
    entities, _ = EvidenceCoverageValidator.extract_entities_and_dimensions(TEST_QUERY)

    assert "Evaluate" not in entities, "'Evaluate' must never become a research entity."
    assert "Compare" not in entities, "'Compare' must never become a research entity."
    for e in entities:
        assert e.lower() not in {"evaluate", "compare", "building", "systems", "production-ready"}


def test_seven_requested_dimensions_retained():
    """
    REGRESSION TEST 3: Seven requested dimensions retained.
    All 7 requested dimensions must be extracted and preserved without omission:
    Architecture, Orchestration, Tool Integration, Memory Management, Scalability, Error Handling, Limitations.
    """
    _, dimensions = EvidenceCoverageValidator.extract_entities_and_dimensions(TEST_QUERY)

    expected_dimensions = [
        "Architecture",
        "Orchestration",
        "Tool Integration",
        "Memory Management",
        "Scalability",
        "Error Handling",
        "Limitations"
    ]

    for expected in expected_dimensions:
        assert expected in dimensions, f"Missing requested dimension: {expected}. Found: {dimensions}"

    assert len(dimensions) == 7, f"Expected exactly 7 dimensions, got {len(dimensions)}: {dimensions}"
    assert dimensions == expected_dimensions, f"Dimensions order mismatch. Got {dimensions}"


def test_irrelevant_microsoft_wikipedia_pages_rejected():
    """
    REGRESSION TEST 4: Irrelevant Microsoft Wikipedia pages rejected.
    Never treat Microsoft Flight Simulator Wikipedia pages (or other unrelated pages)
    as evidence for Microsoft AutoGen.
    """
    test_results = [
        {
            "title": "Microsoft Flight Simulator - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Microsoft_Flight_Simulator",
            "snippet": "Microsoft Flight Simulator is an amateur flight simulator program developed by Microsoft.",
            "domain": "en.wikipedia.org"
        },
        {
            "title": "Xbox Game Studios - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Xbox_Game_Studios",
            "snippet": "Xbox Game Studios is an American video game publisher and division of Microsoft Gaming.",
            "domain": "en.wikipedia.org"
        },
        {
            "title": "AutoGen - Multi-Agent Conversation Framework | Microsoft Learn",
            "url": "https://microsoft.github.io/autogen/",
            "snippet": "AutoGen is an open-source framework by Microsoft that enables multi-agent conversation systems.",
            "domain": "microsoft.github.io"
        },
        {
            "title": "microsoft/autogen: A programming framework for agentic AI - GitHub",
            "url": "https://github.com/microsoft/autogen",
            "snippet": "A programming framework for agentic AI. Developed by Microsoft Research.",
            "domain": "github.com"
        }
    ]

    filtered = InformationFilter.filter_and_deduplicate_search_results(
        results=test_results,
        user_query=TEST_QUERY,
        target_entities=["LangGraph", "CrewAI", "Microsoft AutoGen"],
        max_to_keep=10
    )

    filtered_urls = [r["url"] for r in filtered]
    assert "https://en.wikipedia.org/wiki/Microsoft_Flight_Simulator" not in filtered_urls, (
        "Microsoft Flight Simulator Wikipedia page must be rejected."
    )
    assert "https://en.wikipedia.org/wiki/Xbox_Game_Studios" not in filtered_urls, (
        "Xbox Game Studios Wikipedia page must be rejected."
    )
    assert "https://microsoft.github.io/autogen/" in filtered_urls
    assert "https://github.com/microsoft/autogen" in filtered_urls


def test_official_documentation_prioritized():
    """
    REGRESSION TEST 5: Official documentation prioritized.
    Prioritize:
    - LangGraph official documentation and langchain-ai GitHub.
    - CrewAI official documentation and crewAIInc GitHub.
    - Microsoft AutoGen official documentation and microsoft GitHub.
    """
    test_sources = [
        # Random community blogs
        {
            "title": "Tutorial on LangGraph - Medium",
            "url": "https://medium.com/@dev/langgraph-tutorial",
            "snippet": "Here is a quick tutorial on using LangGraph for multi-agent loops.",
            "domain": "medium.com"
        },
        {
            "title": "Building with CrewAI - Dev.to",
            "url": "https://dev.to/engineer/crewai-deep-dive",
            "snippet": "A walkthrough of building role-playing agents in CrewAI.",
            "domain": "dev.to"
        },
        # Official docs and repositories
        {
            "title": "LangGraph Documentation",
            "url": "https://langchain-ai.github.io/langgraph/",
            "snippet": "Official documentation for LangGraph: Build stateful multi-agent systems with LangChain.",
            "domain": "langchain-ai.github.io"
        },
        {
            "title": "langchain-ai/langgraph GitHub",
            "url": "https://github.com/langchain-ai/langgraph",
            "snippet": "Build resilient language agents as graphs. Official repository by LangChain.",
            "domain": "github.com"
        },
        {
            "title": "CrewAI Official Documentation",
            "url": "https://docs.crewai.com/",
            "snippet": "Official docs for CrewAI: Framework for orchestrating autonomous AI agents.",
            "domain": "docs.crewai.com"
        },
        {
            "title": "crewAIInc/crewAI GitHub",
            "url": "https://github.com/crewAIInc/crewAI",
            "snippet": "Official CrewAI repository. Framework for orchestrating role-playing, autonomous AI agents.",
            "domain": "github.com"
        },
        {
            "title": "AutoGen Official Documentation",
            "url": "https://microsoft.github.io/autogen/",
            "snippet": "Official documentation for Microsoft AutoGen: Multi-agent conversation framework.",
            "domain": "microsoft.github.io"
        },
        {
            "title": "microsoft/autogen GitHub",
            "url": "https://github.com/microsoft/autogen",
            "snippet": "Official Microsoft AutoGen repository. A programming framework for agentic AI.",
            "domain": "github.com"
        }
    ]

    # Verify classification of each official source
    lg_doc = SourceClassifier.classify("https://langchain-ai.github.io/langgraph/", target_entities=["LangGraph"])
    assert lg_doc.is_official is True
    assert lg_doc.category == SourceCategory.OFFICIAL_DOCS

    lg_repo = SourceClassifier.classify("https://github.com/langchain-ai/langgraph", target_entities=["LangGraph"])
    assert lg_repo.is_official is True
    assert lg_repo.category == SourceCategory.OFFICIAL_REPO

    crew_doc = SourceClassifier.classify("https://docs.crewai.com/", target_entities=["CrewAI"])
    assert crew_doc.is_official is True
    assert crew_doc.category == SourceCategory.OFFICIAL_DOCS

    crew_repo = SourceClassifier.classify("https://github.com/crewAIInc/crewAI", target_entities=["CrewAI"])
    assert crew_repo.is_official is True
    assert crew_repo.category == SourceCategory.OFFICIAL_REPO

    ag_doc = SourceClassifier.classify("https://microsoft.github.io/autogen/", target_entities=["Microsoft AutoGen"])
    assert ag_doc.is_official is True
    assert ag_doc.category == SourceCategory.OFFICIAL_DOCS

    ag_repo = SourceClassifier.classify("https://github.com/microsoft/autogen", target_entities=["Microsoft AutoGen"])
    assert ag_repo.is_official is True
    assert ag_repo.category == SourceCategory.OFFICIAL_REPO

    # Verify filtering ranks official sources at the top
    filtered = InformationFilter.filter_and_deduplicate_search_results(
        results=test_sources,
        user_query=TEST_QUERY,
        target_entities=["LangGraph", "CrewAI", "Microsoft AutoGen"],
        max_to_keep=6
    )

    # All top 6 results should be official docs/repos
    for item in filtered:
        assert item["is_official"] is True, f"Official source should be prioritized, but got {item['url']}"


def test_unsupported_or_weakly_supported_claims_not_marked_fully_grounded():
    """
    REGRESSION TEST 6: Unsupported claims not marked fully grounded.
    A source URL existing does not automatically mean a claim is fully grounded.
    If source content does not actually support the specific claim, mark it partially grounded or unsupported.
    """
    verified_sources = [
        {
            "title": "LangGraph Documentation",
            "url": "https://langchain-ai.github.io/langgraph/",
            "snippet": "LangGraph is a library for building stateful, multi-actor applications with LLMs."
        },
        {
            "title": "Microsoft AutoGen Overview",
            "url": "https://microsoft.github.io/autogen/",
            "snippet": "AutoGen enables building multi-agent conversational systems with asynchronous chat."
        }
    ]

    evidence_corpus = (
        "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
        "AutoGen enables building multi-agent conversational systems with asynchronous chat."
    )

    findings = [
        # Claim 1: Well-supported by AutoGen documentation
        "Microsoft AutoGen provides conversational multi-agent systems with asynchronous chat support.",
        # Claim 2: Hallucinated/fabricated technical assertion for LangGraph with no source support
        "LangGraph includes native quantum hardware compiler integration with sub-nanosecond qubit entanglement.",
        # Claim 3: Claim discussing CrewAI, which has NO verified sources retrieved at all
        "CrewAI provides hierarchical team structures with manager agents delegating subtasks."
    ]

    sanitized, validated_claims, _ = ClaimCitationValidator.validate_findings(
        findings=findings,
        verified_sources=verified_sources,
        evidence_corpus=evidence_corpus,
        target_entities=["LangGraph", "CrewAI", "Microsoft AutoGen"]
    )

    # Claim 1: Fully grounded
    assert validated_claims[0].is_fully_grounded is True
    assert validated_claims[0].grounding_status == "FULLY_GROUNDED"
    assert "https://microsoft.github.io/autogen/" in validated_claims[0].supporting_urls

    # Claim 2: Unsupported / weakly grounded despite LangGraph URL existing in verified sources
    assert validated_claims[1].is_fully_grounded is False
    assert validated_claims[1].grounding_status in ("UNSUPPORTED", "PARTIALLY_GROUNDED")
    # LangGraph URL must NOT be assigned as supporting URL for quantum compilation claim
    assert "https://langchain-ai.github.io/langgraph/" not in validated_claims[1].supporting_urls
    assert "[Unverified Observation]" in sanitized[1] or "[Partially Verified]" in sanitized[1]

    # Claim 3: Unsupported due to no CrewAI source available
    assert validated_claims[2].is_fully_grounded is False
    assert validated_claims[2].grounding_status == "UNSUPPORTED"
    assert len(validated_claims[2].supporting_urls) == 0
    assert "[Unverified Observation]" in sanitized[2]


# ==============================================================================
# 10. Topic-Independent Entity, Dimension, Coverage & Second-Hop Pipeline
# ==============================================================================

def test_healthcare_entity_and_dimension_extraction():
    """
    REGRESSION TEST: Healthcare Generative AI Query.
    Query: 'Conduct a comprehensive research report on how Generative AI is transforming
    the healthcare industry in 2025–2026. Analyze medical diagnosis, drug discovery,
    AI healthcare assistants, patient data privacy, real-world implementations,
    benefits, risks, and future challenges.'

    Verifies:
    1. Entities extracted are ['Generative AI', 'Healthcare Industry'].
    2. 'Conduct', 'Generative', 'Analyze', 'Use' are strictly excluded from entities.
    3. Dimensions extracted match all 8 requested topics:
       Medical Diagnosis, Drug Discovery, AI Healthcare Assistants, Patient Data Privacy,
       Real-World Implementations, Benefits, Risks, Future Challenges.
    4. Hardcoded software dimensions like 'Database Integration', 'Scalability', 'Features'
       are NOT incorrectly injected.
    """
    query = (
        "Conduct a comprehensive research report on how Generative AI is transforming "
        "the healthcare industry in 2025–2026. Analyze medical diagnosis, drug discovery, "
        "AI healthcare assistants, patient data privacy, real-world implementations, "
        "benefits, risks, and future challenges."
    )

    entities, dimensions = EvidenceCoverageValidator.extract_entities_and_dimensions(query)

    # 1. Meaningful entities
    assert "Generative AI" in entities, f"Expected 'Generative AI' in {entities}"
    assert "Healthcare Industry" in entities, f"Expected 'Healthcare Industry' in {entities}"

    # 2. Command/instruction words must NEVER be entities
    for forbidden in ["Conduct", "Generative", "Analyze", "Use", "Comprehensive", "Report"]:
        assert forbidden not in entities, f"Forbidden non-entity '{forbidden}' was extracted: {entities}"

    # 3. Exactly the requested dimensions
    expected_dimensions = [
        "Medical Diagnosis",
        "Drug Discovery",
        "AI Healthcare Assistants",
        "Patient Data Privacy",
        "Real-World Implementations",
        "Benefits",
        "Risks",
        "Future Challenges",
    ]
    for dim in expected_dimensions:
        assert dim in dimensions, f"Missing requested dimension '{dim}' in {dimensions}"

    # 4. Erroneous software defaults must NOT be present
    for bogus in ["Database Integration", "Features", "Scalability"]:
        assert bogus not in dimensions, f"Unrelated dimension '{bogus}' was incorrectly generated: {dimensions}"


def test_quantum_cryptography_unrelated_query():
    """
    REGRESSION TEST: Unrelated Research Query 1 - Quantum Computing & Post-Quantum Cryptography.
    Query: 'Analyze the impact of Quantum Computing on Post-Quantum Cryptography in 2026.
    Evaluate lattice-based encryption, stateful hash-based signatures, NIST standards,
    migration risks, and implementation overhead.'

    Verifies:
    - Entities are correctly extracted without hardcoding.
    - Dimensions reflect the specific cryptographic evaluation criteria.
    - Verbs like 'Analyze' and 'Evaluate' are not treated as entities.
    """
    query = (
        "Analyze the impact of Quantum Computing on Post-Quantum Cryptography in 2026. "
        "Evaluate lattice-based encryption, stateful hash-based signatures, NIST standards, "
        "migration risks, and implementation overhead."
    )

    entities, dimensions = EvidenceCoverageValidator.extract_entities_and_dimensions(query)

    assert "Quantum Computing" in entities
    assert "Post-Quantum Cryptography" in entities
    assert "Analyze" not in entities
    assert "Evaluate" not in entities

    expected_dims = [
        "Lattice-Based Encryption",
        "Stateful Hash-Based Signatures",
        "NIST Standards",
        "Migration Risks",
        "Implementation Overhead",
    ]
    for dim in expected_dims:
        assert dim in dimensions, f"Expected '{dim}' in {dimensions}"


def test_european_logistics_ev_unrelated_query():
    """
    REGRESSION TEST: Unrelated Research Query 2 - Electric Vehicles in European Logistics.
    Query: 'Evaluate autonomous electric vehicle adoption in European logistics.
    Assess battery swapping vs ultra-fast charging, total cost of ownership,
    grid capacity, regulatory hurdles, and safety standards.'

    Verifies:
    - Multi-word domain entities are extracted cleanly.
    - Custom evaluation dimensions are accurately captured.
    """
    query = (
        "Evaluate autonomous electric vehicle adoption in European logistics. "
        "Assess battery swapping vs ultra-fast charging, total cost of ownership, "
        "grid capacity, regulatory hurdles, and safety standards."
    )

    entities, dimensions = EvidenceCoverageValidator.extract_entities_and_dimensions(query)

    assert "Autonomous Electric Vehicle" in entities
    assert "European Logistics" in entities
    assert "Evaluate" not in entities
    assert "Assess" not in entities

    dims_lower = [d.lower() for d in dimensions]
    for expected in [
        "battery swapping vs ultra-fast charging",
        "total cost of ownership",
        "grid capacity",
        "regulatory hurdles",
        "safety standards",
    ]:
        assert expected in dims_lower, f"Expected '{expected}' in {dimensions}"


def test_healthcare_coverage_matrix_and_second_hop_relevance():
    """
    REGRESSION TEST: Coverage matrix evaluation and second-hop query generation
    for the healthcare research topic.

    Verifies:
    1. Coverage matrix correctly matches healthcare evidence against healthcare dimensions.
    2. Missing pairs trigger contextual second-hop queries.
    3. Generated second-hop queries do NOT contain 'official documentation guide'
       or irrelevant software boilerplate.
    """
    query = (
        "Conduct a comprehensive research report on how Generative AI is transforming "
        "the healthcare industry in 2025–2026. Analyze medical diagnosis, drug discovery, "
        "AI healthcare assistants, patient data privacy, real-world implementations, "
        "benefits, risks, and future challenges."
    )

    entities, dimensions = EvidenceCoverageValidator.extract_entities_and_dimensions(query)

    # Simulated verified sources with partial coverage (diagnosis and privacy covered; drug discovery missing)
    snippet = "Generative AI transforms medical diagnosis, radiology pathology detection, and patient data privacy under HIPAA compliance."
    verified_sources = [
        {
            "title": "Generative AI in Clinical Diagnostics 2025",
            "url": "https://example.org/genai-health-diagnosis",
            "snippet": snippet
        }
    ]

    coverage_report = EvidenceCoverageValidator.evaluate(
        user_question=query,
        evidence_chunks=[snippet],
        verified_sources=verified_sources,
        entities=entities,
        dimensions=dimensions
    )

    # Check matrix structure
    assert len(coverage_report.coverage_matrix) == len(entities)
    for ent in entities:
        assert ent in coverage_report.coverage_matrix
        for dim in dimensions:
            assert dim in coverage_report.coverage_matrix[ent]

    # Cell verification: Generative AI + Medical Diagnosis should have evidence
    assert coverage_report.coverage_matrix["Generative AI"]["Medical Diagnosis"] in ("COVERED", "PARTIAL")
    assert coverage_report.coverage_matrix["Generative AI"]["Patient Data Privacy"] in ("COVERED", "PARTIAL")

    # AI Healthcare Assistants was not covered in the snippet -> must be a missing pair
    assert ("Generative AI", "AI Healthcare Assistants") in coverage_report.missing_pairs
    assert coverage_report.coverage_matrix["Generative AI"]["AI Healthcare Assistants"] == "MISSING"

    # Generate second-hop queries targeting missing evidence
    second_hop = EvidenceCoverageValidator.generate_second_hop_queries(
        missing_pairs=coverage_report.missing_pairs,
        existing_queries=["Generative AI healthcare transformation"],
        max_queries=4,
        user_question=query
    )

    assert len(second_hop) > 0
    for sh_q in second_hop:
        # Crucial: Must be relevant to healthcare research, NOT software documentation
        assert "official documentation guide" not in sh_q
        assert "architecture benchmarks" not in sh_q
        # Must target missing entities or dimensions
        q_lower = sh_q.lower()
        assert any(
            term.lower() in q_lower
            for term in ["generative ai", "drug discovery", "healthcare", "assistants", "diagnosis"]
        ), f"Second hop query '{sh_q}' lacked context from missing evidence"


