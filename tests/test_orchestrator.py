"""
Unit tests for InformationFilter, ConfidenceAssessor, and Agent Data Models.
"""

import pytest
from app.agent.filter import InformationFilter
from app.agent.confidence import ConfidenceAssessor
from app.models.response import ResearchResponse, SourceItem, ConfidenceLevel


def test_url_normalization():
    dirty_url = "https://example.com/docs/vector-db/?utm_source=twitter&utm_medium=cpc#section2"
    clean_url = InformationFilter.normalize_url(dirty_url)
    assert clean_url == "https://example.com/docs/vector-db"


def test_filter_and_deduplicate_results():
    sample_results = [
        {"title": "Qdrant Docs", "url": "https://qdrant.tech/documentation/", "snippet": "Vector search engine documentation and Rust client.", "domain": "qdrant.tech"},
        {"title": "Qdrant Docs Duplicate", "url": "https://qdrant.tech/documentation/?utm_campaign=launch", "snippet": "Vector search engine documentation.", "domain": "qdrant.tech"},
        {"title": "Junk Result", "url": "https://badsite.com/blocked", "snippet": "Enable JavaScript to run this app.", "domain": "badsite.com"},
        {"title": "Milvus Vector DB", "url": "https://milvus.io/docs", "snippet": "Milvus is a distributed open-source vector database built for AI.", "domain": "milvus.io"}
    ]

    filtered = InformationFilter.filter_and_deduplicate_search_results(sample_results, max_to_keep=5)
    # Junk should be filtered out
    assert not any("badsite.com" in r["url"] for r in filtered)
    # Duplicates should be deduplicated
    assert len([r for r in filtered if "qdrant.tech" in r["url"]]) == 1
    # Trusted domains (qdrant, milvus) should be scored and kept
    assert len(filtered) == 2


def test_confidence_assessment_scenarios():
    # Scenario 1: Zero sources -> INSUFFICIENT_EVIDENCE
    conf1, _, lim1 = ConfidenceAssessor.evaluate([], [], [])
    assert conf1 == ConfidenceLevel.INSUFFICIENT_EVIDENCE.value
    assert len(lim1) >= 1

    # Scenario 2: 1 sparse source -> LOW
    conf2, _, lim2 = ConfidenceAssessor.evaluate(
        search_results=[{"url": "https://example.com"}],
        extracted_pages=[],
        evidence_snippets=["Short snippet"]
    )
    assert conf2 == ConfidenceLevel.LOW.value

    # Scenario 3: Multiple sources with successful extraction -> HIGH
    conf3, assumptions3, _ = ConfidenceAssessor.evaluate(
        search_results=[{"url": "https://a.com"}, {"url": "https://b.com"}],
        extracted_pages=[{"success": True, "url": "https://a.com"}],
        evidence_snippets=["Detailed evidence " * 150]
    )
    assert conf3 == ConfidenceLevel.HIGH.value


def test_research_response_schema_validation():
    valid_data = {
        "user_question": "Compare Qdrant vs Chroma",
        "short_answer": "Qdrant is optimized for production scale; Chroma is lightweight and embeddable.",
        "key_findings": [
            "Qdrant written in Rust provides higher throughput.",
            "Chroma provides seamless Python SQLite embedding."
        ],
        "sources_used": [
            {
                "title": "Qdrant Docs",
                "url": "https://qdrant.tech",
                "relevance": "Architectural benchmarks"
            }
        ],
        "confidence_level": "HIGH",
        "assumptions": ["Tested on standard hardware."],
        "limitations": ["Managed cloud costs not compared."],
        "suggested_next_steps": ["Run local latency benchmarks."]
    }

    response = ResearchResponse.model_validate(valid_data)
    assert response.confidence_level == "HIGH"
    assert len(response.sources_used) == 1
    assert response.sources_used[0].url == "https://qdrant.tech"
