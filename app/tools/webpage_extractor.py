"""
Webpage Content Extraction Tool with SSRF Protection, Clean Parsing, and Error Resilience.
"""

from typing import Any, Dict
from bs4 import BeautifulSoup
import httpx
from app.config import get_settings
from app.tools.base import BaseTool
from app.utils.logging import get_logger
from app.utils.security import SecurityValidator

logger = get_logger(__name__)

# Standard browser User-Agent to avoid getting blocked by anti-bot headers
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class WebpageExtractorTool(BaseTool):
    """
    Extracts readable article and technical documentation text from given web URLs,
    stripping advertisements, navigations, scripts, and non-article clutter.
    Enforces SSRF prevention and timeout boundaries.
    """

    name = "webpage_extractor"
    description = (
        "Extract readable, sanitized text content from a web URL. "
        "Use this tool when search snippets are insufficient and detailed technical evidence is needed."
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
                        "url": {
                            "type": "string",
                            "description": "The exact HTTP or HTTPS URL of the webpage to extract."
                        }
                    },
                    "required": ["url"]
                }
            }
        }

    async def execute(self, url: str) -> Dict[str, Any]:
        """
        Safely fetches and extracts readable text from the provided URL.
        Guarantees that extraction failures are never treated as successes.
        """
        cleaned_url = url.strip()
        if not cleaned_url:
            return {
                "success": False,
                "url": url,
                "title": "",
                "content": "",
                "error": "URL cannot be empty."
            }

        # 1. SSRF Protection: verify target host is public and safe
        is_safe, reason = SecurityValidator.validate_url_safe_for_ssrf(cleaned_url)
        if not is_safe:
            logger.warning(f"Blocked potential SSRF attempt for URL '{cleaned_url}': {reason}")
            return {
                "success": False,
                "url": cleaned_url,
                "title": "",
                "content": "",
                "error": f"Security restriction: {reason}"
            }

        logger.info(f"Extracting webpage content from: {cleaned_url}")

        ua = "PredixionResearchBot/1.0 (https://github.com/predixion-ai; research@predixion.ai)" if "wikipedia.org" in cleaned_url.lower() else USER_AGENT
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.HTTP_TIMEOUT_SECONDS,
                follow_redirects=True,
                headers=headers,
                max_redirects=5
            ) as client:
                response = await client.get(cleaned_url)

                # Check final redirected URL against SSRF
                final_url = str(response.url)
                final_safe, final_reason = SecurityValidator.validate_url_safe_for_ssrf(final_url)
                if not final_safe:
                    logger.warning(f"Redirected to disallowed URL '{final_url}': {final_reason}")
                    return {
                        "success": False,
                        "url": final_url,
                        "title": "",
                        "content": "",
                        "error": f"Security restriction on redirect: {final_reason}"
                    }

                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {cleaned_url} - HTTP status {response.status_code}")
                    return {
                        "success": False,
                        "url": cleaned_url,
                        "title": "",
                        "content": "",
                        "error": f"HTTP status {response.status_code} ({response.reason_phrase})"
                    }

                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    logger.warning(f"Non-text content type '{content_type}' at {cleaned_url}")
                    return {
                        "success": False,
                        "url": cleaned_url,
                        "title": "",
                        "content": "",
                        "error": f"Unsupported content-type: '{content_type}'"
                    }

                title, clean_text = self._clean_html(response.text)

                if not clean_text or len(clean_text) < 50:
                    return {
                        "success": False,
                        "url": cleaned_url,
                        "title": title,
                        "content": "",
                        "error": "Webpage contained insufficient readable text."
                    }

                # Cap content length to prevent context explosion
                truncated_text = clean_text[: self.settings.MAX_PAGE_CHARS]
                if len(clean_text) > self.settings.MAX_PAGE_CHARS:
                    truncated_text += "\n[... Content truncated for length ...]"

                logger.info(
                    f"Successfully extracted {len(truncated_text)} characters from {cleaned_url} (Title: '{title}')"
                )
                return {
                    "success": True,
                    "url": cleaned_url,
                    "title": title,
                    "content": truncated_text,
                    "char_count": len(truncated_text)
                }

        except httpx.TimeoutException:
            logger.error(f"Timeout extracting URL: {cleaned_url}")
            return {
                "success": False,
                "url": cleaned_url,
                "title": "",
                "content": "",
                "error": f"Request timed out after {self.settings.HTTP_TIMEOUT_SECONDS}s."
            }
        except httpx.RequestError as e:
            logger.error(f"Network error extracting URL {cleaned_url}: {str(e)}")
            return {
                "success": False,
                "url": cleaned_url,
                "title": "",
                "content": "",
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error extracting URL {cleaned_url}: {str(e)}")
            return {
                "success": False,
                "url": cleaned_url,
                "title": "",
                "content": "",
                "error": f"Parsing failure: {str(e)}"
            }

    def _clean_html(self, html_text: str) -> tuple[str, str]:
        """Parses HTML, removes boilerplates and returns (title, extracted_text)."""
        soup = BeautifulSoup(html_text, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else "Untitled Webpage"

        # Remove irrelevant and clutter elements
        for element in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "svg", "iframe", "form"]):
            element.decompose()

        # Try to find main article or content body if present
        main_content = soup.find("article") or soup.find("main") or soup.find("body") or soup
        text = main_content.get_text(separator="\n", strip=True)

        # Collapse excessive newlines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        return title, clean_text
