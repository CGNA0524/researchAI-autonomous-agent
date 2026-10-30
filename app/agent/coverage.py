"""
Evidence Coverage Validator and Matrix Analyzer.
Tracks entity and evaluation dimension coverage across retrieved evidence,
identifies missing research dimensions, and generates targeted second-hop queries.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
from pydantic import BaseModel, Field
from app.utils.logging import get_logger

logger = get_logger(__name__)


class CoverageCell(BaseModel):
    """Evidence status for a specific (entity, dimension) intersection."""
    entity: str
    dimension: str
    status: str = "MISSING"  # "COVERED", "PARTIAL", "MISSING"
    evidence_strength: str = "NONE"  # "STRONG", "MODERATE", "WEAK", "NONE"
    supporting_sources: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)
    is_directly_supported: bool = False


class EvidenceCoverageReport(BaseModel):
    """Comprehensive evidence coverage matrix and completeness assessment."""
    entities: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    total_cells: int = 0
    covered_cells: int = 0
    partial_cells: int = 0
    missing_cells: int = 0
    coverage_ratio: float = 0.0
    is_sufficient: bool = False
    missing_entities: List[str] = Field(default_factory=list)
    missing_dimensions: List[str] = Field(default_factory=list)
    missing_pairs: List[Tuple[str, str]] = Field(default_factory=list)
    coverage_matrix: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    limitations_identified: List[str] = Field(default_factory=list)


# Dimension keyword synonyms for semantic matching
DIMENSION_KEYWORDS: Dict[str, List[str]] = {
    "Performance": ["performance", "latency", "throughput", "speed", "qps", "benchmark", "fast", "overhead"],
    "Architecture": ["architecture", "design", "internals", "engine", "distributed", "monolithic", "storage", "rust", "c++", "python"],
    "Async Support": ["async", "asynchronous", "asyncio", "concurrency", "uvicorn", "event loop", "non-blocking"],
    "Scalability": ["scalability", "scale", "clustering", "sharding", "horizontal", "billion", "cluster", "distributed"],
    "Deployment": ["deployment", "deploy", "docker", "kubernetes", "setup", "self-hosted", "cloud", "managed", "install"],
    "Authentication": ["auth", "authentication", "authorization", "jwt", "oauth", "security", "permissions"],
    "Database Integration": ["database", "orm", "sql", "nosql", "postgres", "sqlite", "migrations", "model"],
    "Ecosystem": ["ecosystem", "community", "libraries", "plugins", "integrations", "github stars", "adoption"],
    "Limitations": ["limitations", "drawbacks", "disadvantages", "bottleneck", "issues", "weakness", "tradeoffs"],
    "Features": ["features", "capabilities", "functionality", "support", "filtering", "payload"],
}


class EvidenceCoverageValidator:
    """Evaluates whether all requested comparison entities and dimensions have sufficient evidence."""

    @classmethod
    def extract_entities_and_dimensions(
        cls,
        user_question: str,
        tasks_planned: Optional[List[Any]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Extracts researched entities and relevant evaluation dimensions from the query and plan.
        """
        question_clean = user_question.replace(",", " ").replace(" vs ", " ").replace(" versus ", " ")
        words = question_clean.split()

        # Known common tech entities
        known_techs = [
            "fastapi", "django", "flask", "qdrant", "milvus", "weaviate", "chroma", "chromadb",
            "pinecone", "redis", "postgresql", "sqlite", "duckdb", "mongodb", "langchain",
            "llamaindex", "haystack", "docker", "kubernetes", "celery", "kafka", "rabbitmq"
        ]

        entities: List[str] = []
        for word in words:
            clean_word = word.strip("?:;.,'\"").lower()
            for tech in known_techs:
                if tech == clean_word and tech.capitalize() not in entities:
                    # Map canonical proper capitalization
                    canonical = {
                        "fastapi": "FastAPI", "django": "Django", "flask": "Flask",
                        "qdrant": "Qdrant", "milvus": "Milvus", "weaviate": "Weaviate",
                        "chroma": "Chroma", "chromadb": "Chroma", "pinecone": "Pinecone",
                        "redis": "Redis", "postgresql": "PostgreSQL", "sqlite": "SQLite",
                        "duckdb": "DuckDB", "mongodb": "MongoDB", "langchain": "LangChain",
                        "llamaindex": "LlamaIndex", "haystack": "Haystack",
                    }.get(tech, tech.capitalize())
                    if canonical not in entities:
                        entities.append(canonical)

        # If entities weren't in known dictionary, check capitalized words or keywords
        if len(entities) < 2:
            candidate_words = [w.strip("?:;.,'\"") for w in user_question.split() if w and w[0].isupper()]
            stopwords = {"Compare", "What", "Which", "How", "Why", "The", "For", "Top", "Explain", "Recommend"}
            for cw in candidate_words:
                if cw not in stopwords and len(cw) >= 3 and cw not in entities:
                    entities.append(cw)

        # Ensure fallback entities if query is a general tech topic
        if not entities:
            entities = ["Primary Topic"]

        # Extract dimensions
        dimensions: List[str] = []
        question_lower = user_question.lower()
        for dim, kws in DIMENSION_KEYWORDS.items():
            if any(kw in question_lower for kw in kws):
                dimensions.append(dim)

        # Incorporate aspects from plan tasks if available
        if tasks_planned:
            for task in tasks_planned:
                aspects = getattr(task, "aspects_to_investigate", [])
                for asp in aspects:
                    for dim, kws in DIMENSION_KEYWORDS.items():
                        if any(kw in asp.lower() for kw in kws) and dim not in dimensions:
                            dimensions.append(dim)

        # Ensure default core dimensions for comparative technical studies
        default_core = ["Features", "Architecture", "Performance", "Deployment", "Limitations"]
        for d in default_core:
            if d not in dimensions and len(dimensions) < 4:
                dimensions.append(d)

        return entities, dimensions

    @classmethod
    def evaluate(
        cls,
        user_question: str,
        evidence_chunks: List[str],
        verified_sources: List[Dict[str, Any]],
        entities: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None,
    ) -> EvidenceCoverageReport:
        """
        Constructs the evidence coverage matrix and computes statistical coverage metrics.
        """
        resolved_entities = entities or []
        resolved_dimensions = dimensions or []

        if not resolved_entities or not resolved_dimensions:
            e, d = cls.extract_entities_and_dimensions(user_question)
            resolved_entities = resolved_entities or e
            resolved_dimensions = resolved_dimensions or d

        total_cells = len(resolved_entities) * len(resolved_dimensions)
        covered_count = 0
        partial_count = 0
        missing_count = 0
        missing_pairs: List[Tuple[str, str]] = []
        coverage_matrix: Dict[str, Dict[str, str]] = {}
        limitations: List[str] = []

        # Combine evidence texts
        combined_evidence = " ".join(evidence_chunks).lower()

        for entity in resolved_entities:
            coverage_matrix[entity] = {}
            entity_lower = entity.lower()

            for dimension in resolved_dimensions:
                dim_keywords = DIMENSION_KEYWORDS.get(dimension, [dimension.lower()])

                # Count distinct sources discussing both entity and dimension
                supporting_sources: List[str] = []
                for src in verified_sources:
                    src_text = f"{src.get('title', '')} {src.get('snippet', '')}".lower()
                    if entity_lower in src_text and any(dkw in src_text for dkw in dim_keywords):
                        url = src.get("url", "")
                        if url and url not in supporting_sources:
                            supporting_sources.append(url)

                # Also inspect extracted evidence chunks
                chunks_mentioning = sum(
                    1 for c in evidence_chunks
                    if entity_lower in c.lower() and any(dkw in c.lower() for dkw in dim_keywords)
                )

                if len(supporting_sources) >= 2 or (len(supporting_sources) >= 1 and chunks_mentioning >= 1):
                    status = "COVERED"
                    covered_count += 1
                elif len(supporting_sources) == 1 or chunks_mentioning >= 1:
                    status = "PARTIAL"
                    partial_count += 1
                else:
                    status = "MISSING"
                    missing_count += 1
                    missing_pairs.append((entity, dimension))

                coverage_matrix[entity][dimension] = status

        # Coverage ratio formula: Covered counts as 1.0, Partial counts as 0.5
        effective_covered = covered_count + (partial_count * 0.5)
        coverage_ratio = round(effective_covered / max(total_cells, 1), 3)
        is_sufficient = coverage_ratio >= 0.70 and missing_count <= (total_cells // 3)

        # Identify missing entities and dimensions
        missing_entities: List[str] = []
        for entity in resolved_entities:
            entity_statuses = list(coverage_matrix[entity].values())
            if entity_statuses.count("MISSING") >= (len(resolved_dimensions) * 0.6):
                missing_entities.append(entity)
                limitations.append(f"Insufficient independent evidence gathered for entity: '{entity}'.")

        missing_dimensions: List[str] = []
        for dimension in resolved_dimensions:
            dim_statuses = [coverage_matrix[e].get(dimension, "MISSING") for e in resolved_entities]
            if dim_statuses.count("MISSING") >= (len(resolved_entities) * 0.6):
                missing_dimensions.append(dimension)
                limitations.append(f"Comparative evidence for evaluation dimension '{dimension}' is incomplete.")

        return EvidenceCoverageReport(
            entities=resolved_entities,
            dimensions=resolved_dimensions,
            total_cells=total_cells,
            covered_cells=covered_count,
            partial_cells=partial_count,
            missing_cells=missing_count,
            coverage_ratio=coverage_ratio,
            is_sufficient=is_sufficient,
            missing_entities=missing_entities,
            missing_dimensions=missing_dimensions,
            missing_pairs=missing_pairs,
            coverage_matrix=coverage_matrix,
            limitations_identified=limitations
        )

    @classmethod
    def generate_second_hop_queries(
        cls,
        missing_pairs: List[Tuple[str, str]],
        existing_queries: List[str],
        max_queries: int = 3
    ) -> List[str]:
        """
        Generates highly focused, targeted queries exclusively for missing (entity, dimension) gaps.
        """
        queries: List[str] = []
        existing_lowered = {q.lower().strip() for q in existing_queries}

        for entity, dimension in missing_pairs:
            candidate_query = f"{entity} {dimension.lower()} documentation benchmarks"
            if candidate_query.lower() not in existing_lowered and candidate_query not in queries:
                queries.append(candidate_query)
            if len(queries) >= max_queries:
                break

        # Fallback if specific pairs didn't produce enough
        if len(queries) < max_queries and missing_pairs:
            for entity, dimension in missing_pairs:
                alt_query = f"{entity} official guide {dimension.lower()}"
                if alt_query.lower() not in existing_lowered and alt_query not in queries:
                    queries.append(alt_query)
                if len(queries) >= max_queries:
                    break

        return queries
