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

# Patterns for off-topic/unrelated results (e.g. flight simulators, video games when researching tech frameworks)
IRRELEVANT_PATTERNS = [
    "flight simulator",
    "flight-simulator",
    "flight simulation",
    "aviation",
    "aircraft",
    "video game",
    "xbox game studios",
    "xbox",
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
            "official doc", "official documentation", "primary doc", "official source", "prefer official"
        ])

        # Clean target entities
        clean_target_entities = [e.lower().strip() for e in (target_entities or []) if len(e.strip()) > 1]

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

            # Irrelevant topics guard (e.g. flight simulator, video games, xbox when researching AI frameworks)
            source_text_corpus = f"{title.lower()} {snippet.lower()} {domain.lower()} {normalized.lower()}"
            if any(p in source_text_corpus for p in IRRELEVANT_PATTERNS):
                logger.debug(f"Filtering off-topic result from {normalized} (matches irrelevant pattern)")
                continue

            # Entity Grounding: If target entities are defined, source MUST mention at least one target entity
            matched_target_entity = None
            if clean_target_entities:
                for e in clean_target_entities:
                    matches_e = (
                        ("autogen" in source_text_corpus) if e == "microsoft autogen" else (e in source_text_corpus)
                    )
                    if matches_e or (classified.matched_entity and (
                        e in classified.matched_entity.lower() or classified.matched_entity.lower() in e
                    )):
                        matched_target_entity = e
                        break
                if not matched_target_entity and not classified.is_official:
                    logger.debug(f"Filtering irrelevant search result from {normalized} (mentions none of {clean_target_entities})")
                    continue

            # Calculate keyword relevance
            keyword_matches = sum(1 for kw in query_words if kw in source_text_corpus) if query_words else 1

            # If user provided specific keywords and none matched, penalize or drop
            if query_words and keyword_matches == 0 and not classified.is_official:
                logger.debug(f"Filtering irrelevant search result from {normalized} (zero keywords matched)")
                continue

            # Combined score: Authority weight + Keyword relevance
            base_score = classified.combined_score + (keyword_matches * 0.8)

            # Boost verified official sources
            if classified.is_official:
                base_score += 2.0

            # Specific Entity Match Boost
            if matched_target_entity:
                base_score += 1.5

            # If user explicitly requested official docs, prioritize official docs and penalize encyclopedias/blogs
            if official_requested:
                if classified.category == SourceCategory.OFFICIAL_DOCS and classified.is_official:
                    base_score += 5.0
                elif classified.category == SourceCategory.OFFICIAL_REPO and classified.is_official:
                    base_score += 3.0
                elif any(wiki in domain for wiki in ["wikipedia.org", "en.wikipedia.org"]):
                    base_score -= 3.5  # Penalize Wikipedia when official docs are requested
                elif classified.category in (SourceCategory.COMMUNITY_ARTICLE, SourceCategory.GENERAL_REPO):
                    base_score -= 2.0
                elif classified.category == SourceCategory.ACADEMIC_PAPER:
                    base_score -= 1.5

            item_copy = dict(item)
            item_copy["normalized_url"] = normalized
            item_copy["score"] = round(base_score, 3)
            item_copy["category"] = classified.category.value
            item_copy["authority_score"] = classified.authority_score
            item_copy["is_official"] = classified.is_official
            item_copy["matched_entity"] = matched_target_entity or classified.matched_entity
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
        max_extract: int = 3,
        target_entities: Optional[List[str]] = None
    ) -> List[str]:
        """
        Selects the top distinct URLs that warrant full webpage extraction.
        When target_entities are provided, performs round-robin interleaving
        to ensure balanced evidentiary coverage across all researched entities.
        """
        clean_entities = [e.lower().strip() for e in (target_entities or []) if len(e.strip()) > 1]
        if not clean_entities or len(clean_entities) <= 1:
            selected: List[str] = []
            for item in filtered_search_results:
                url = item.get("url", "")
                if url and url not in selected:
                    selected.append(url)
                if len(selected) >= max_extract:
                    break
            return selected

        # Group by target entity
        entity_groups: Dict[str, List[str]] = {e: [] for e in clean_entities}
        general_urls: List[str] = []

        for item in filtered_search_results:
            url = item.get("url", "")
            if not url:
                continue
            matched = item.get("matched_entity", "")
            text_context = f"{item.get('title', '')} {item.get('snippet', '')} {url}".lower()

            found_entity = None
            if matched and matched.lower() in entity_groups:
                found_entity = matched.lower()
            else:
                for e in clean_entities:
                    if e in text_context:
                        found_entity = e
                        break

            if found_entity:
                if url not in entity_groups[found_entity]:
                    entity_groups[found_entity].append(url)
            else:
                if url not in general_urls:
                    general_urls.append(url)

        # Interleave round-robin across entities
        selected: List[str] = []
        max_rounds = max((len(urls) for urls in entity_groups.values()), default=0)
        for r in range(max_rounds):
            for e in clean_entities:
                urls = entity_groups.get(e, [])
                if r < len(urls):
                    u = urls[r]
                    if u not in selected:
                        selected.append(u)
                    if len(selected) >= max_extract:
                        return selected

        # Fill any remaining slots with general URLs
        for u in general_urls:
            if u not in selected:
                selected.append(u)
            if len(selected) >= max_extract:
                break

        return selected

    @classmethod
    def sanitize_evidence_text(cls, content: str, source_url: str) -> str:
        """Wraps untrusted content into defanged, isolated evidence containers."""
        return SecurityValidator.sanitize_untrusted_content(content, source_url)
