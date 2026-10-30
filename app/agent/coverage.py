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
    "Architecture": ["architecture", "design", "internals", "engine", "distributed", "monolithic", "storage", "rust", "c++", "python", "graph", "state machine"],
    "Orchestration": ["orchestration", "multi-agent", "agent coordination", "workflow", "supervisor", "hierarchical", "sequential", "graph execution", "routing"],
    "Tool Integration": ["tool integration", "tools", "tool calling", "function calling", "external tools", "api tools", "custom tools", "tool use"],
    "Memory Management": ["memory management", "memory", "short-term memory", "long-term memory", "state persistence", "session memory", "checkpointer", "context window", "message history"],
    "Error Handling": ["error handling", "exception handling", "retry", "fallback", "recovery", "fault tolerance", "resilience", "human-in-the-loop"],
    "Async Support": ["async", "asynchronous", "asyncio", "concurrency", "uvicorn", "event loop", "non-blocking"],
    "Scalability": ["scalability", "scale", "clustering", "sharding", "horizontal", "billion", "cluster", "distributed", "parallel"],
    "Deployment": ["deployment", "deploy", "operational", "maintenance", "operations", "docker", "kubernetes", "setup", "self-hosted", "cloud", "managed", "install"],
    "Authentication": ["auth", "authentication", "authorization", "jwt", "oauth", "security", "permissions"],
    "Database Integration": ["database", "orm", "sql", "nosql", "postgres", "sqlite", "migrations", "model"],
    "Integrations": ["integration", "integrations", "vector stores", "connectors", "loaders", "plugins"],
    "Ecosystem": ["ecosystem", "community", "libraries", "plugins", "integrations", "github stars", "adoption"],
    "Developer Experience": ["developer experience", "dx", "api design", "learning curve", "usability", "debugging", "ease of use"],
    "Limitations": ["limitations", "drawbacks", "disadvantages", "bottleneck", "issues", "weakness", "tradeoffs"],
    "Features": ["features", "capabilities", "functionality", "support", "filtering", "payload"],
}


class EvidenceCoverageValidator:
    """Evaluates whether all requested comparison entities and dimensions have sufficient evidence."""

    # Canonical knowledge dictionary mapping common framework/product expressions to canonical names
    CANONICAL_ENTITIES: Dict[str, str] = {
        # Multi-word entities (must be prioritized to avoid breaking into bare words)
        "microsoft autogen": "Microsoft AutoGen",
        "autogen": "Microsoft AutoGen",
        "langgraph": "LangGraph",
        "lang graph": "LangGraph",
        "crewai": "CrewAI",
        "crew ai": "CrewAI",
        "langchain": "LangChain",
        "llamaindex": "LlamaIndex",
        "llama index": "LlamaIndex",
        "deepset haystack": "Haystack",
        "haystack": "Haystack",
        "fastapi": "FastAPI",
        "django": "Django",
        "flask": "Flask",
        "qdrant": "Qdrant",
        "milvus": "Milvus",
        "weaviate": "Weaviate",
        "chromadb": "Chroma",
        "chroma": "Chroma",
        "pinecone": "Pinecone",
        "redis": "Redis",
        "postgresql": "PostgreSQL",
        "postgres": "PostgreSQL",
        "sqlite": "SQLite",
        "duckdb": "DuckDB",
        "mongodb": "MongoDB",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "celery": "Celery",
        "apache kafka": "Kafka",
        "kafka": "Kafka",
        "apache spark": "Spark",
        "spark": "Spark",
        "rabbitmq": "RabbitMQ",
        "hugging face": "Hugging Face",
    }

    # Strict non-entity stopwords to prevent imperative instruction verbs or generic terms becoming entities
    NON_ENTITY_WORDS = {
        "compare", "comparing", "comparison", "evaluate", "evaluating", "evaluation",
        "analyze", "analyzing", "analysis", "assess", "assessing", "assessment",
        "investigate", "investigating", "review", "benchmark", "benchmarking",
        "what", "which", "how", "why", "who", "whom", "where", "when",
        "the", "for", "top", "explain", "recommend", "recommendation",
        "building", "build", "production", "production-ready", "multi-agent",
        "systems", "system", "applications", "application", "frameworks", "framework",
        "tools", "tool", "versus", "across", "between", "with", "from", "into",
        "microsoft", "google", "amazon", "apache"  # Prevent bare company prefixes without product
    }

    @classmethod
    def extract_entities_and_dimensions(
        cls,
        user_question: str,
        tasks_planned: Optional[List[Any]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Extracts researched entities and relevant evaluation dimensions from the query and plan.
        Preserves multi-word entity names (e.g. 'Microsoft AutoGen') and prevents instruction verbs ('Evaluate')
        from becoming entities.
        """
        question_lower = user_question.lower()
        matched_items: List[Tuple[int, int, str]] = []

        # 1. Search for canonical entities (longest multi-word matches first)
        sorted_canonical = sorted(cls.CANONICAL_ENTITIES.keys(), key=len, reverse=True)
        for key in sorted_canonical:
            idx = 0
            while True:
                pos = question_lower.find(key, idx)
                if pos == -1:
                    break
                end_pos = pos + len(key)
                # Ensure word boundary checks
                is_start_boundary = (pos == 0 or not question_lower[pos - 1].isalnum())
                is_end_boundary = (end_pos == len(question_lower) or not question_lower[end_pos].isalnum())

                if is_start_boundary and is_end_boundary:
                    # Check if already covered by an earlier match span
                    if not any(s <= pos and end_pos <= e for s, e, _ in matched_items):
                        canonical = cls.CANONICAL_ENTITIES[key]
                        if not any(m[2] == canonical for m in matched_items):
                            matched_items.append((pos, end_pos, canonical))
                idx = end_pos

        # Sort matches by position in question
        matched_items.sort(key=lambda x: x[0])
        entities: List[str] = [m[2] for m in matched_items]

        # 2. If fewer than 2 entities found, look for unknown capitalized candidate terms
        if len(entities) < 2:
            candidate_words = [w.strip("?:;.,'\"") for w in user_question.split() if w and w[0].isupper()]
            for cw in candidate_words:
                clean_cw = cw.lower()
                if clean_cw not in cls.NON_ENTITY_WORDS and len(cw) >= 3 and cw not in entities:
                    entities.append(cw)

        # Fallback if no entity was detected
        if not entities:
            entities = ["Primary Topic"]

        # 3. Extract dimensions
        matched_dims: List[Tuple[int, str]] = []
        for dim, kws in DIMENSION_KEYWORDS.items():
            positions = [question_lower.find(kw) for kw in kws if kw in question_lower]
            if positions:
                first_pos = min(p for p in positions if p != -1)
                matched_dims.append((first_pos, dim))

        # Sort dimensions by their appearance order in the question
        matched_dims.sort(key=lambda x: x[0])
        dimensions: List[str] = [d[1] for d in matched_dims]

        # Disambiguate: If 'Tool Integration' was requested, avoid redundant generic 'Integrations'
        if "Tool Integration" in dimensions and "Integrations" in dimensions:
            dimensions.remove("Integrations")

        # Incorporate aspects from plan tasks if available
        if tasks_planned:
            for task in tasks_planned:
                aspects = getattr(task, "aspects_to_investigate", [])
                for asp in aspects:
                    for dim, kws in DIMENSION_KEYWORDS.items():
                        if any(kw in asp.lower() for kw in kws) and dim not in dimensions:
                            dimensions.append(dim)

        # Ensure default core dimensions only if user did not provide sufficient dimensions
        default_core = ["Features", "Architecture", "Performance", "Deployment", "Limitations"]
        for d in default_core:
            if d not in dimensions and len(dimensions) < 4:
                dimensions.append(d)

        return entities, dimensions

    @classmethod
    def _entity_matches_text(cls, entity: str, text: str) -> bool:
        """Determines if an entity is referenced in text, supporting multi-word variants."""
        e_low = entity.lower().strip()
        if e_low == "microsoft autogen":
            return "microsoft autogen" in text or "autogen" in text
        return e_low in text

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
                    if cls._entity_matches_text(entity, src_text) and any(dkw in src_text for dkw in dim_keywords):
                        url = src.get("url", "")
                        if url and url not in supporting_sources:
                            supporting_sources.append(url)

                # Also inspect extracted evidence chunks
                chunks_mentioning = sum(
                    1 for c in evidence_chunks
                    if cls._entity_matches_text(entity, c.lower()) and any(dkw in c.lower() for dkw in dim_keywords)
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
        max_queries: int = 3,
        missing_entities: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generates highly focused, targeted queries exclusively for missing (entity, dimension) gaps.
        Prioritizes completely missing entities and interleaves across entities to prevent starvation.
        """
        queries: List[str] = []
        existing_lowered = {q.lower().strip() for q in existing_queries}

        if not missing_pairs:
            return queries

        # Group missing pairs by entity
        entity_dims: Dict[str, List[str]] = {}
        for entity, dimension in missing_pairs:
            entity_dims.setdefault(entity, []).append(dimension)

        # Prioritize entities: completely missing entities first, then by number of missing dimensions
        missing_ent_set = set(missing_entities or [])
        sorted_entities = sorted(
            entity_dims.keys(),
            key=lambda e: (e not in missing_ent_set, -len(entity_dims[e]))
        )

        # Interleave across entities (round-robin)
        max_depth = max(len(dims) for dims in entity_dims.values()) if entity_dims else 0
        for depth in range(max_depth):
            for entity in sorted_entities:
                dims = entity_dims[entity]
                if depth < len(dims):
                    dim = dims[depth]
                    candidate = f"{entity} {dim.lower()} official documentation guide"
                    if candidate.lower() not in existing_lowered and candidate not in queries:
                        queries.append(candidate)
                    if len(queries) >= max_queries:
                        return queries

        # Fallback with benchmark queries if slots remain
        for depth in range(max_depth):
            for entity in sorted_entities:
                dims = entity_dims[entity]
                if depth < len(dims):
                    dim = dims[depth]
                    alt = f"{entity} {dim.lower()} architecture benchmarks"
                    if alt.lower() not in existing_lowered and alt not in queries:
                        queries.append(alt)
                    if len(queries) >= max_queries:
                        return queries

        return queries
