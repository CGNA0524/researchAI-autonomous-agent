"""
Information Filtering, Deduplication, Authority Scoring, and Prompt Injection Sanitization.
"""

from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse, urlunparse
from app.utils.logging import get_logger
from app.utils.security import SecurityValidator

logger = get_logger(__name__)

# Established authoritative domains for engineering and academic research
TRUSTED_DOMAINS = {
    "github.com",
    "arxiv.org",
    "docs.python.org",
    "stackoverflow.com",
    "medium.com",
    "dev.to",
    "en.wikipedia.org",
    "huggingface.co",
    "pinecone.io",
    "qdrant.tech",
    "weaviate.io",
    "milvus.io",
    "chromadb.com",
    "mongodb.com",
    "postgresql.org",
    "redis.io",
    "towardsdatascience.com",
    "fastapi.tiangolo.com",
}

# Snippet patterns indicating junk, captcha, or unindexed pages
JUNK_PATTERNS = [
    "enable javascript to run this app",
    "access denied",
    "403 forbidden",
    "sign in to continue",
    "captcha verification",
    "cookies are disabled",
    "page not found",
    "404 not found",
]


from app.agent.source_classifier import SourceClassifier, SourceCategory

class InformationFilter:
    """Filters, deduplicates, ranks, and sanitizes gathered evidence."""

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalizes URLs by stripping tracking parameters, fragments, and trailing slashes."""
        try:
            parsed = urlparse(url.strip())
            # Strip query tracking parameters and fragment
            clean_path = parsed.path.rstrip("/")
            normalized = urlunparse((
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                clean_path,
                "",  # params
                "",  # query
                ""   # fragment
            ))
            return normalized
        except Exception:
            return url.strip().rstrip("/")

    @classmethod
    def filter_and_deduplicate_search_results(
        cls,
        results: List[Dict[str, Any]],
        user_query: str = "",
        target_entities: Optional[List[str]] = None,
        max_to_keep: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Deduplicates search results by normalized URL and content fingerprint,
        filters junk snippets, classifies sources by authority tier,
        and ranks authoritative/official sources higher.
        """
        seen_urls = set()
        seen_snippets = set()
        scored_results: List[Tuple[float, Dict[str, Any]]] = []

        # Check if user explicitly asked for official documentation
        query_lower = user_query.lower()
        official_requested = any(term in query_lower for term in [
            "official doc", "official documentation", "primary doc", "official source"
        ])

        # Extract meaningful query keywords (length >= 3, lowercase)
        stopwords = {"what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
                     "from", "with", "have", "been", "were", "year", "your", "about", "into", "over"}
        query_words = set()
        if user_query:
            query_words = {
                w.lower() for w in user_query.replace(".", " ").replace("?", " ").split()
                if len(w) >= 3 and w.lower() not in stopwords
            }

        for item in results:
            raw_url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            domain = item.get("domain", "")

            if not raw_url or not raw_url.startswith("http"):
                continue

            normalized = cls.normalize_url(raw_url)
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)

            # Content-level near-duplicate deduplication (using first 100 non-space characters)
            snippet_clean = "".join(snippet.lower().split())[:100]
            if snippet_clean and snippet_clean in seen_snippets:
                logger.debug(f"Filtering duplicate content snippet from {normalized}")
                continue
            if snippet_clean:
                seen_snippets.add(snippet_clean)

            # Classify source authority and quality tier
            classified = SourceClassifier.classify(
                url=normalized,
                title=title,
                snippet=snippet,
                target_entities=target_entities
            )

            # Discard junk / low quality sources
            if classified.category == SourceCategory.LOW_QUALITY:
                logger.debug(f"Filtering out low-quality/junk result from {normalized}")
                continue

            # Calculate keyword relevance
            text_corpus = f"{title.lower()} {snippet.lower()} {domain.lower()}"
            keyword_matches = sum(1 for kw in query_words if kw in text_corpus) if query_words else 1

            # If user provided specific keywords and none matched, penalize or drop
            if query_words and keyword_matches == 0 and not classified.is_official:
                logger.debug(f"Filtering irrelevant search result from {normalized} (zero keywords matched)")
                continue

            # Combined score: Authority weight + Keyword relevance + Official boosts
            base_score = classified.combined_score + (keyword_matches * 0.8)

            # If user explicitly requested official docs, boost official sources and penalize community
            if official_requested:
                if classified.category in (SourceCategory.OFFICIAL_DOCS, SourceCategory.OFFICIAL_REPO):
                    base_score += 4.0
                elif classified.category in (SourceCategory.COMMUNITY_ARTICLE, SourceCategory.GENERAL_REPO):
                    base_score -= 1.5

            item_copy = dict(item)
            item_copy["normalized_url"] = normalized
            item_copy["score"] = round(base_score, 3)
            item_copy["category"] = classified.category.value
            item_copy["authority_score"] = classified.authority_score
            item_copy["is_official"] = classified.is_official
            item_copy["matched_entity"] = classified.matched_entity
            item_copy["relevance_overlap"] = float(keyword_matches)
            scored_results.append((base_score, item_copy))

        # Sort descending by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        filtered = [item for _, item in scored_results[:max_to_keep]]
        logger.info(f"Information filter preserved {len(filtered)} high-quality results from {len(results)} inputs.")
        return filtered

    @classmethod
    def select_urls_for_extraction(
        cls,
        filtered_search_results: List[Dict[str, Any]],
        max_extract: int = 3
    ) -> List[str]:
        """Selects the top distinct URLs that warrant full webpage extraction."""
        selected: List[str] = []
        for item in filtered_search_results:
            url = item.get("url", "")
            if url and url not in selected:
                selected.append(url)
            if len(selected) >= max_extract:
                break
        return selected

    @classmethod
    def sanitize_evidence_text(cls, content: str, source_url: str) -> str:
        """Wraps untrusted content into defanged, isolated evidence containers."""
        return SecurityValidator.sanitize_untrusted_content(content, source_url)
