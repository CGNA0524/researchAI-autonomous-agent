"""
Integration tests for FastAPI endpoints: /, /health, /research, /docs.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Predixion AI Research Agent"
    assert data["status"] == "online"
    assert "documentation" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "tools_ready" in data
    assert "web_search" in data["tools_ready"]
    assert "webpage_extractor" in data["tools_ready"]


def test_research_endpoint_empty_query_validation():
    response = client.post("/research", json={"query": ""})
    assert response.status_code == 422


def test_research_endpoint_whitespace_validation():
    response = client.post("/research", json={"query": "    "})
    assert response.status_code == 422


def test_swagger_docs_accessible():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
