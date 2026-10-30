"""
Unit tests for WebSearchTool and WebpageExtractorTool.
"""

import pytest
from app.tools.web_search import WebSearchTool
from app.tools.webpage_extractor import WebpageExtractorTool


@pytest.mark.asyncio
async def test_web_search_empty_query():
    tool = WebSearchTool()
    res = await tool.execute(query="   ")
    assert res["success"] is False
    assert "Search query cannot be empty" in res["error"]


def test_web_search_schema():
    tool = WebSearchTool()
    schema = tool.get_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert "query" in schema["function"]["parameters"]["required"]


@pytest.mark.asyncio
async def test_webpage_extractor_ssrf_blocked():
    extractor = WebpageExtractorTool()
    res = await extractor.execute(url="http://127.0.0.1:8000/internal")
    assert res["success"] is False
    assert "Security restriction" in res["error"]


@pytest.mark.asyncio
async def test_webpage_extractor_html_cleaning():
    extractor = WebpageExtractorTool()
    raw_html = """
    <html>
        <head><title>Vector DB Guide</title></head>
        <body>
            <nav><a href="/home">Home</a></nav>
            <script>alert('run');</script>
            <article>
                <h1>Comparing Vector Databases</h1>
                <p>Qdrant is written in Rust and provides high-performance vector search.</p>
                <p>Milvus provides distributed architecture for billion-scale vectors.</p>
            </article>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """
    title, clean_text = extractor._clean_html(raw_html)
    assert title == "Vector DB Guide"
    assert "Comparing Vector Databases" in clean_text
    assert "Qdrant is written in Rust" in clean_text
    assert "alert('run')" not in clean_text
    assert "Home" not in clean_text
    assert "Copyright 2026" not in clean_text
