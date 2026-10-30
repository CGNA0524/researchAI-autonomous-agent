"""
Source Quality Classifier and Authority Ranking Component.
Classifies web sources into distinct credibility tiers and ranks them
based on authoritative domain matching, technical relevance, and evidence depth.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Set
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SourceCategory(str, Enum):
    """Hierarchical authority categories for evidence sources."""
    OFFICIAL_DOCS = "OFFICIAL_DOCS"
    OFFICIAL_REPO = "OFFICIAL_REPO"
    ACADEMIC_PAPER = "ACADEMIC_PAPER"
    REPUTABLE_PUBLICATION = "REPUTABLE_PUBLICATION"
    COMMUNITY_ARTICLE = "COMMUNITY_ARTICLE"
    GENERAL_REPO = "GENERAL_REPO"
    LOW_QUALITY = "LOW_QUALITY"


class ClassifiedSource(BaseModel):
    """Detailed classification and scoring breakdown of an individual source."""
    url: str
    normalized_url: str
    domain: str
    category: SourceCategory
    authority_score: float
    relevance_score: float
    combined_score: float
    is_official: bool = False
    matched_entity: Optional[str] = None
    title: str = ""
    snippet: str = ""


# Authoritative academic publishers
ACADEMIC_DOMAINS = {
    "arxiv.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "semanticscholar.org",
    "researchgate.net",
    "openreview.net",
    "papers.nips.cc",
    "biorxiv.org",
}

# Reputable engineering, industry and architectural publications
REPUTABLE_TECH_DOMAINS = {
    "martinfowler.com",
    "lwn.net",
    "infoq.com",
    "queue.acm.org",
    "aws.amazon.com",
    "cloud.google.com",
    "netflixtechblog.com",
    "blog.cloudflare.com",
    "engineering.fb.com",
    "github.blog",
    "stackoverflow.blog",
    "zdnet.com",
    "theverge.com",
    "arstechnica.com",
}

# Community-driven platforms
COMMUNITY_DOMAINS = {
    "medium.com",
    "dev.to",
    "towardsdatascience.com",
    "substack.com",
    "hashnode.dev",
    "hackernoon.com",
    "dzone.com",
    "freecodecamp.org",
    "geeksforgeeks.org",
    "baeldung.com",
    "reddit.com",
}

# Standard library and foundation official domains
KNOWN_FOUNDATION_DOMAINS = {
    "python.org": "Python",
    "docs.python.org": "Python",
    "pypi.org": "PyPI",
    "fastapi.tiangolo.com": "FastAPI",
    "djangoproject.com": "Django",
    "docs.djangoproject.com": "Django",
    "palletsprojects.com": "Flask",
    "flask.palletsprojects.com": "Flask",
    "qdrant.tech": "Qdrant",
    "weaviate.io": "Weaviate",
    "milvus.io": "Milvus",
    "chromadb.com": "Chroma",
    "pinecone.io": "Pinecone",
    "redis.io": "Redis",
    "postgresql.org": "PostgreSQL",
    "sqlite.org": "SQLite",
    "duckdb.org": "DuckDB",
    "mongodb.com": "MongoDB",
    "langchain.com": "LangChain",
    "docs.langchain.com": "LangChain",
    "python.langchain.com": "LangChain",
    "js.langchain.com": "LangChain",
    "langchain-ai.github.io/langgraph": "LangGraph",
    "langchain-ai.github.io": "LangGraph",
    "docs.crewai.com": "CrewAI",
    "crewai.com": "CrewAI",
    "microsoft.github.io/autogen": "Microsoft AutoGen",
    "microsoft.github.io": "Microsoft AutoGen",
    "llamaindex.ai": "LlamaIndex",
    "docs.llamaindex.ai": "LlamaIndex",
    "haystack.deepset.ai": "Haystack",
    "docs.haystack.deepset.ai": "Haystack",
    "deepset.ai": "Haystack",
    "huggingface.co": "Hugging Face",
    "kubernetes.io": "Kubernetes",
    "docker.com": "Docker",
}

# Verified official GitHub organizations for authoritative repositories
KNOWN_OFFICIAL_GITHUB_ORGS: Dict[str, Set[str]] = {
    "langchain": {"langchain-ai", "langchain"},
    "langgraph": {"langchain-ai", "langchain"},
    "crewai": {"crewaiinc", "joaomdmoura", "crewai"},
    "autogen": {"microsoft"},
    "microsoft autogen": {"microsoft"},
    "llamaindex": {"run-llama", "llamaindex"},
    "haystack": {"deepset-ai", "deepset", "haystack"},
    "fastapi": {"tiangolo"},
    "django": {"django"},
    "flask": {"pallets"},
    "qdrant": {"qdrant"},
    "weaviate": {"weaviate"},
    "milvus": {"milvus-io", "milvus"},
    "chroma": {"chroma-core", "chromadb"},
    "docker": {"docker"},
    "kubernetes": {"kubernetes"},
    "redis": {"redis", "redis-io"},
    "celery": {"celery"},
    "postgresql": {"postgres"},
}

ENCYCLOPEDIA_DOMAINS = {
    "wikipedia.org",
    "en.wikipedia.org",
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
    "security check to continue",
    "cloudflare ray id",
]


class SourceClassifier:
    """Classifies sources and assigns calibrated authority weights."""

    # Base authority scores by tier
    CATEGORY_WEIGHTS: Dict[SourceCategory, float] = {
        SourceCategory.OFFICIAL_DOCS: 4.5,
        SourceCategory.OFFICIAL_REPO: 3.5,
        SourceCategory.ACADEMIC_PAPER: 3.0,
        SourceCategory.REPUTABLE_PUBLICATION: 2.2,
        SourceCategory.COMMUNITY_ARTICLE: 1.0,
        SourceCategory.GENERAL_REPO: 0.5,
        SourceCategory.LOW_QUALITY: 0.0,
    }

    @classmethod
    def classify(
        cls,
        url: str,
        title: str = "",
        snippet: str = "",
        target_entities: Optional[List[str]] = None
    ) -> ClassifiedSource:
        """Classifies a URL into a source category based on domain patterns and target entities."""
        parsed = urlparse(url.lower().strip())
        domain = parsed.netloc or ""
        path = parsed.path or ""
        title_lower = title.lower()
        snippet_lower = snippet.lower()
        entity_canonical_map = {e.lower().strip(): e.strip() for e in (target_entities or []) if len(e.strip()) > 1}
        entities = list(entity_canonical_map.keys())

        # 1. Check for Low Quality / Junk
        if any(junk in snippet_lower for junk in JUNK_PATTERNS) or not domain:
            return cls._build_record(
                url, domain, SourceCategory.LOW_QUALITY, 0.0, 0.0, title, snippet
            )

        # 2. Check for Academic Domains
        if any(acad in domain for acad in ACADEMIC_DOMAINS) or "arxiv" in path:
            return cls._build_record(
                url, domain, SourceCategory.ACADEMIC_PAPER, cls.CATEGORY_WEIGHTS[SourceCategory.ACADEMIC_PAPER],
                cls._calc_relevance(title_lower, snippet_lower, entities), title, snippet
            )

        # 3. Check for Official Documentation
        # A. Known foundation domain match (prioritizing longer, specific host/path patterns)
        url_host_and_path = f"{domain}{path}".rstrip("/")
        for known_domain, entity_name in sorted(KNOWN_FOUNDATION_DOMAINS.items(), key=lambda x: len(x[0]), reverse=True):
            if known_domain in domain or known_domain in url_host_and_path:
                is_target_entity = not entities or any(
                    e == entity_name.lower() or e in entity_name.lower() or entity_name.lower() in e or
                    ("autogen" in e and "autogen" in entity_name.lower())
                    for e in entities
                )
                canonical_name = entity_canonical_map.get(entity_name.lower(), entity_name)
                return cls._build_record(
                    url, domain, SourceCategory.OFFICIAL_DOCS, cls.CATEGORY_WEIGHTS[SourceCategory.OFFICIAL_DOCS],
                    cls._calc_relevance(title_lower, snippet_lower, entities) + (1.5 if is_target_entity else 0.0),
                    title, snippet,
                    is_official=is_target_entity, matched_entity=canonical_name
                )

        # B. Dynamic official domain matching for target entities (e.g. docs.entity.io, entity.org/docs)
        for entity in entities:
            entity_slug = entity.replace(" ", "").replace("-", "").replace(".", "").lower()
            domain_slug = domain.replace("-", "").replace(".", "").lower()

            if entity_slug and (
                domain == f"{entity_slug}.com" or
                domain == f"{entity_slug}.org" or
                domain == f"{entity_slug}.io" or
                domain == f"{entity_slug}.ai" or
                domain == f"{entity_slug}.tech" or
                domain == f"docs.{entity_slug}.com" or
                domain == f"docs.{entity_slug}.org" or
                domain == f"docs.{entity_slug}.io" or
                domain == f"docs.{entity_slug}.ai" or
                domain == f"docs.{entity_slug}.tech" or
                f"{entity_slug}.readthedocs.io" in domain or
                f"docs.{entity_slug}" in domain_slug
            ):
                is_docs = "doc" in domain or "doc" in path or "wiki" in path or "manual" in path or "guide" in path
                cat = SourceCategory.OFFICIAL_DOCS if is_docs else SourceCategory.OFFICIAL_REPO
                canonical_name = entity_canonical_map.get(entity, entity)
                return cls._build_record(
                    url, domain, cat, cls.CATEGORY_WEIGHTS[cat],
                    cls._calc_relevance(title_lower, snippet_lower, entities) + 1.0, title, snippet,
                    is_official=True, matched_entity=canonical_name
                )

        # C. Generic documentation domains
        if "docs." in domain or ".readthedocs.io" in domain or "/docs/" in path or "/documentation/" in path:
            return cls._build_record(
                url, domain, SourceCategory.OFFICIAL_DOCS, cls.CATEGORY_WEIGHTS[SourceCategory.OFFICIAL_DOCS] - 0.5,
                cls._calc_relevance(title_lower, snippet_lower, entities), title, snippet,
                is_official=False
            )

        # 4. Check GitHub Repositories
        if "github.com" in domain:
            # Check if it looks like an official organization repo for an entity
            path_parts = [p for p in path.strip("/").split("/") if p]
            if len(path_parts) >= 2:
                owner, repo = path_parts[0].lower(), path_parts[1].lower()
                repo_slug = repo.replace("_", "").replace("-", "").lower()
                for entity in entities:
                    e_slug = entity.replace(" ", "").replace("-", "").lower()
                    verified_orgs = (
                        KNOWN_OFFICIAL_GITHUB_ORGS.get(entity.lower(), set()) |
                        KNOWN_OFFICIAL_GITHUB_ORGS.get(e_slug, set())
                    )
                    if "autogen" in entity.lower():
                        verified_orgs |= KNOWN_OFFICIAL_GITHUB_ORGS.get("autogen", set())

                    # Check if owner is an official organization for this entity
                    is_official_owner = (
                        owner in verified_orgs or
                        owner == e_slug or
                        owner == f"{e_slug}-ai" or
                        owner == f"{e_slug}-io" or
                        owner == f"{e_slug}-project" or
                        owner == f"{e_slug}project" or
                        owner == f"{e_slug}-core"
                    )
                    # Check if repo is the main framework repository, docs, or official core
                    e_clean = e_slug.replace("_", "").replace("-", "")
                    is_core_repo = (
                        repo_slug == e_clean or
                        repo_slug == "autogen" or
                        repo_slug.startswith(e_clean) or
                        ("autogen" in entity.lower() and repo_slug.startswith("autogen")) or
                        repo in {e_slug, "autogen", "docs", "documentation", "core", f"{e_slug}-python", f"{e_slug}-core", "autogen-core", "autogen-ext", "autogen-agentchat"}
                    )
                    if is_official_owner and is_core_repo:
                        canonical_name = entity_canonical_map.get(entity, entity)
                        if "autogen" in entity.lower():
                            canonical_name = "Microsoft AutoGen"
                        return cls._build_record(
                            url, domain, SourceCategory.OFFICIAL_REPO,
                            cls.CATEGORY_WEIGHTS[SourceCategory.OFFICIAL_REPO],
                            cls._calc_relevance(title_lower, snippet_lower, entities) + 1.0,
                            title, snippet, is_official=True, matched_entity=canonical_name
                        )
            # Otherwise, general user GitHub repo (personal tutorials, forks, third-party projects)
            return cls._build_record(
                url, domain, SourceCategory.GENERAL_REPO,
                cls.CATEGORY_WEIGHTS[SourceCategory.GENERAL_REPO],
                cls._calc_relevance(title_lower, snippet_lower, entities),
                title, snippet, is_official=False
            )

        # 5. Check Encyclopedias (e.g. Wikipedia)
        if any(wiki in domain for wiki in ENCYCLOPEDIA_DOMAINS):
            text_to_check = f"{title_lower} {snippet_lower} {path}".lower()
            if any(p in text_to_check for p in ["flight_simulator", "flight simulator", "video_game", "video game", "aircraft", "aviation", "xbox"]):
                return cls._build_record(
                    url, domain, SourceCategory.LOW_QUALITY, 0.0, 0.0, title, snippet
                )
            return cls._build_record(
                url, domain, SourceCategory.COMMUNITY_ARTICLE,
                1.0, cls._calc_relevance(title_lower, snippet_lower, entities),
                title, snippet, is_official=False
            )

        # 6. Check Reputable Technical Publications
        if any(rep in domain for rep in REPUTABLE_TECH_DOMAINS):
            return cls._build_record(
                url, domain, SourceCategory.REPUTABLE_PUBLICATION,
                cls.CATEGORY_WEIGHTS[SourceCategory.REPUTABLE_PUBLICATION],
                cls._calc_relevance(title_lower, snippet_lower, entities),
                title, snippet
            )

        # 7. Check Community Blogs & Platforms
        if any(comm in domain for comm in COMMUNITY_DOMAINS) or "blog" in domain or "tutorial" in domain:
            return cls._build_record(
                url, domain, SourceCategory.COMMUNITY_ARTICLE,
                cls.CATEGORY_WEIGHTS[SourceCategory.COMMUNITY_ARTICLE],
                cls._calc_relevance(title_lower, snippet_lower, entities),
                title, snippet
            )

        # 8. Default Web Publication
        return cls._build_record(
            url, domain, SourceCategory.COMMUNITY_ARTICLE,
            1.2, cls._calc_relevance(title_lower, snippet_lower, entities),
            title, snippet
        )

    @classmethod
    def _calc_relevance(cls, title: str, snippet: str, entities: List[str]) -> float:
        """Calculates textual keyword relevance against researched entities."""
        if not entities:
            return 1.0
        matches = 0
        text = f"{title} {snippet}"
        for e in entities:
            if e in text:
                matches += 1
        return 1.0 + (matches * 0.75)

    @classmethod
    def _build_record(
        cls,
        url: str,
        domain: str,
        category: SourceCategory,
        auth_score: float,
        rel_score: float,
        title: str,
        snippet: str,
        is_official: bool = False,
        matched_entity: Optional[str] = None
    ) -> ClassifiedSource:
        """Constructs a ClassifiedSource instance with combined ranking score."""
        combined = round(auth_score * 0.6 + rel_score * 0.4, 3)
        return ClassifiedSource(
            url=url,
            normalized_url=url.rstrip("/"),
            domain=domain,
            category=category,
            authority_score=auth_score,
            relevance_score=rel_score,
            combined_score=combined,
            is_official=is_official,
            matched_entity=matched_entity,
            title=title,
            snippet=snippet
        )
