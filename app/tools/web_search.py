"""
Web Search Tool implementation with DuckDuckGo (HTML & Lite) and Wikipedia Fallback.
Provides resilient, key-free web search with real URL unquoting and clean snippets.
"""

import asyncio
from typing import Any, Dict, List, Optional
import urllib.parse
from bs4 import BeautifulSoup
import httpx
from app.config import get_settings
from app.tools.base import BaseTool
from app.utils.logging import get_logger

logger = get_logger(__name__)

SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
WIKI_USER_AGENT = "PredixionResearchBot/1.0 (https://github.com/predixion-ai; research@predixion.ai)"


class WebSearchTool(BaseTool):
    """
    Modular Web Search Tool that queries search engines and returns
    structured results containing titles, verified URLs, snippets, and domains.
    """

    name = "web_search"
    description = (
        "Search the live web for authoritative, up-to-date information, technical articles, "
        "and documentation on a specified query."
    )

    def __init__(self) -> None:
        self.settings = get_settings()

    def get_schema(self) -> Dict[str, Any]:
        """Returns OpenAPI/JSON schema for Groq tool calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The specific search query to run on the web search engine."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results to retrieve (default is 5).",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    async def execute(self, query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes a web search query across primary and secondary engines with error isolation.
        """
        limit = max_results or self.settings.MAX_SEARCH_RESULTS
        cleaned_query = query.strip()

        if not cleaned_query:
            return {
                "success": False,
                "query": query,
                "results": [],
                "error": "Search query cannot be empty."
            }

        logger.info(f"Executing web search for query: '{cleaned_query}' (limit={limit})")
        combined_results: List[Dict[str, Any]] = []
        seen_urls = set()

        def add_unique(items: List[Dict[str, Any]]):
            for item in items:
                u = item.get("url")
                if u and u not in seen_urls:
                    seen_urls.add(u)
                    combined_results.append(item)

        # 1. Primary: DuckDuckGo HTML (short timeout)
        try:
            ddg_results = await self._run_html_search(cleaned_query, limit)
            add_unique(ddg_results)
        except Exception as e:
            logger.debug(f"DDG HTML search notice: {e}")

        # 2. Secondary: GitHub Technical Repositories API
        if len(combined_results) < limit:
            try:
                gh_results = await self._run_github_search(cleaned_query, limit - len(combined_results))
                add_unique(gh_results)
            except Exception as e:
                logger.debug(f"GitHub search notice: {e}")

        # 3. Tertiary: ArXiv Academic Research Papers API
        if len(combined_results) < limit:
            try:
                arxiv_results = await self._run_arxiv_search(cleaned_query, limit - len(combined_results))
                add_unique(arxiv_results)
            except Exception as e:
                logger.debug(f"ArXiv search notice: {e}")

        # 4. Quaternary: Wikipedia Technical Search API
        if len(combined_results) < limit:
            try:
                wiki_results = await self._run_wikipedia_search(cleaned_query, limit - len(combined_results))
                add_unique(wiki_results)
            except Exception as e:
                logger.debug(f"Wikipedia search notice: {e}")

        if combined_results:
            logger.info(f"Search successful ({len(combined_results)} verified results for '{cleaned_query}').")
            return {
                "success": True,
                "query": cleaned_query,
                "results": combined_results[:limit],
                "count": len(combined_results[:limit])
            }

        logger.warning(f"No search results returned from any provider for query: '{cleaned_query}'")
        return {
            "success": True,
            "query": cleaned_query,
            "results": [],
            "message": f"Zero search results found for query '{cleaned_query}'."
        }

    async def _run_html_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Direct HTTP post to DuckDuckGo HTML endpoint."""
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query}
        headers = {
            "User-Agent": SEARCH_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.post(url, data=data, headers=headers)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            formatted: List[Dict[str, Any]] = []

            for result in soup.select(".result"):
                title_elem = result.select_one(".result__title a")
                snippet_elem = result.select_one(".result__snippet")
                if not title_elem or not snippet_elem:
                    continue

                raw_href = title_elem.get("href", "")
                actual_url = raw_href
                if "uddg=" in raw_href:
                    query_params = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
                    if "uddg" in query_params:
                        actual_url = query_params["uddg"][0]

                if not actual_url.startswith("http"):
                    continue

                domain = urllib.parse.urlparse(actual_url).netloc
                formatted.append({
                    "title": title_elem.get_text(strip=True),
                    "url": actual_url,
                    "snippet": snippet_elem.get_text(strip=True),
                    "domain": domain
                })
                if len(formatted) >= max_results:
                    break

            return formatted

    async def _run_lite_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback to DuckDuckGo Lite endpoint."""
        url = "https://lite.duckduckgo.com/lite/"
        data = {"q": query}
        headers = {
            "User-Agent": SEARCH_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.post(url, data=data, headers=headers)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            formatted: List[Dict[str, Any]] = []
            link_elements = soup.select(".result-link")
            snippet_elements = soup.select(".result-snippet")

            for title_elem, snip_elem in zip(link_elements, snippet_elements):
                raw_href = title_elem.get("href", "")
                actual_url = raw_href
                if "uddg=" in raw_href:
                    query_params = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
                    if "uddg" in query_params:
                        actual_url = query_params["uddg"][0]

                if not actual_url.startswith("http"):
                    continue

                domain = urllib.parse.urlparse(actual_url).netloc
                formatted.append({
                    "title": title_elem.get_text(strip=True),
                    "url": actual_url,
                    "snippet": snip_elem.get_text(strip=True),
                    "domain": domain
                })
                if len(formatted) >= max_results:
                    break

            return formatted

    async def _run_wikipedia_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback querying Wikipedia's open search API for technical topics."""
        endpoint = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": "1"
        }
        headers = {"User-Agent": WIKI_USER_AGENT}

        async with httpx.AsyncClient(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(endpoint, params=params, headers=headers)
            if response.status_code != 200:
                return []

            data = response.json()
            search_items = data.get("query", {}).get("search", [])

            # If no items found, relax query by removing generic modifier words
            if not search_items:
                modifiers = {
                    "operational", "features", "capabilities", "overview", "comparison",
                    "benchmarks", "guide", "architecture", "documentation", "limitations",
                    "issues", "details", "analysis", "ranking", "popularity", "common",
                    "methods", "for", "the", "and", "top", "best"
                }
                relaxed = " ".join([w for w in query.replace(",", " ").replace('"', '').split() if w.lower() not in modifiers])
                if relaxed and relaxed.lower() != query.lower():
                    params["srsearch"] = relaxed
                    res2 = await client.get(endpoint, params=params, headers=headers)
                    if res2.status_code == 200:
                        search_items = res2.json().get("query", {}).get("search", [])

            formatted: List[Dict[str, Any]] = []

            for item in search_items:
                title = item.get("title", "")
                raw_snippet = item.get("snippet", "")
                # Clean html tags from snippet
                clean_snippet = BeautifulSoup(raw_snippet, "html.parser").get_text(strip=True)
                url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

                formatted.append({
                    "title": title,
                    "url": url,
                    "snippet": clean_snippet,
                    "domain": "en.wikipedia.org"
                })
                if len(formatted) >= max_results:
                    break

            return formatted

    async def _run_github_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Queries GitHub repository search API for technical code, tools, and docs."""
        endpoint = "https://api.github.com/search/repositories"
        clean_q = " ".join([w for w in query.replace('"', '').split() if not w.startswith("-")])
        if not clean_q:
            return []
        params = {"q": clean_q[:80], "sort": "stars", "per_page": min(max_results * 2, 10)}
        headers = {
            "User-Agent": "PredixionResearchBot/1.0",
            "Accept": "application/vnd.github.v3+json"
        }

        async with httpx.AsyncClient(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(endpoint, params=params, headers=headers)
            if response.status_code != 200:
                return []
            data = response.json()
            items = data.get("items", [])
            formatted: List[Dict[str, Any]] = []
            for item in items:
                desc = item.get("description") or f"Open-source repository for {item.get('name')}"
                formatted.append({
                    "title": f"{item.get('name')} - {item.get('full_name')}",
                    "url": item.get("html_url"),
                    "snippet": desc,
                    "domain": "github.com"
                })
                if len(formatted) >= max_results:
                    break
            return formatted

    async def _run_arxiv_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Queries ArXiv academic search API for technical papers and research."""
        endpoint = "https://export.arxiv.org/api/query"
        words = [w for w in query.replace('"', '').replace(":", " ").split() if len(w) > 2]
        if not words:
            return []
        search_query = " AND ".join([f"all:{w}" for w in words[:4]])
        params = {"search_query": search_query, "max_results": max_results, "sortBy": "relevance"}
        headers = {"User-Agent": "PredixionResearchBot/1.0"}

        async with httpx.AsyncClient(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(endpoint, params=params, headers=headers)
            if response.status_code != 200:
                return []
            soup = BeautifulSoup(response.text, "xml")
            entries = soup.find_all("entry")
            formatted: List[Dict[str, Any]] = []
            for entry in entries:
                title = entry.title.get_text(strip=True).replace("\n", " ") if entry.title else ""
                summary = entry.summary.get_text(strip=True).replace("\n", " ") if entry.summary else ""
                raw_id = entry.id.get_text(strip=True) if entry.id else ""
                if not title or not raw_id:
                    continue
                formatted.append({
                    "title": title,
                    "url": raw_id,
                    "snippet": summary[:300],
                    "domain": "arxiv.org"
                })
                if len(formatted) >= max_results:
                    break
            return formatted
