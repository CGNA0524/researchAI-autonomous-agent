"""
Evidence Coverage Validator and Matrix Analyzer.
Tracks entity and evaluation dimension coverage across retrieved evidence,
identifies missing research dimensions, and generates targeted second-hop queries.
"""

import re
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

    DIMENSION_KEYWORDS = DIMENSION_KEYWORDS

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
        "investigate", "investigating", "review", "reviewing", "benchmark", "benchmarking",
        "conduct", "conducting", "report", "reporting", "research", "researching",
        "what", "which", "how", "why", "who", "whom", "where", "when",
        "the", "for", "top", "explain", "recommend", "recommendation",
        "building", "build", "production", "production-ready", "multi-agent",
        "systems", "system", "applications", "application", "frameworks", "framework",
        "tools", "tool", "versus", "across", "between", "with", "from", "into",
        "microsoft", "google", "amazon", "apache", "use", "using", "comprehensive",
        "detailed", "in-depth", "overview", "study", "summary", "examine", "exploring", "explore",
        "please", "provide", "providing", "describe", "discuss", "identify", "impact", "role"
    }

    @classmethod
    def _clean_entity_name(cls, s: str) -> str:
        """Sanitizes extracted entity strings, stripping years, leading verbs, and prepositions while preserving acronyms."""
        s = re.sub(r'\b(?:in\s+)?(?:20\d\d[–\-—]?20\d\d|20\d\d)\b', '', s)
        s = re.sub(r'^(?:evaluate|evaluating|assess|assessing|analyze|analyzing|investigate|examine|conduct|study|review)\s+(?:a\s+|an\s+|the\s+)?', '', s, flags=re.IGNORECASE)
        s = re.sub(r'^(?:the|a|an|for|in|on|about)\s+', '', s, flags=re.IGNORECASE)
        s = s.strip(" .,;:'\"()?!")
        words = s.split()
        res = []
        for w in words:
            w_clean = w.strip(" .,;:'\"()")
            if not w_clean:
                continue
            if w_clean.upper() in ["AI", "ML", "API", "RAG", "LLM", "NIST", "SAAS", "IOT", "SDK", "PQC", "EV", "EVS"]:
                res.append(w_clean.upper())
            elif w_clean.lower() in ["and", "or", "of", "in", "for", "on", "vs", "versus"]:
                res.append(w_clean.lower())
            elif "-" in w_clean:
                res.append("-".join(part.capitalize() for part in w_clean.split("-")))
            else:
                res.append(w_clean.capitalize())
        return " ".join(res)

    @classmethod
    def get_dimension_keywords(cls, dimension: str) -> List[str]:
        """
        Returns search and match keywords for a dimension.
        Supports both predefined technical dimensions and dynamically extracted topic dimensions.
        """
        if dimension in cls.DIMENSION_KEYWORDS:
            return cls.DIMENSION_KEYWORDS[dimension]

        for d, kws in cls.DIMENSION_KEYWORDS.items():
            if d.lower() == dimension.lower():
                return kws

        keywords = [dimension.lower()]
        stopwords = {
            "and", "the", "for", "with", "from", "into", "over", "under", "about",
            "data", "systems", "system", "management", "across", "between"
        }
        words = [w.lower().strip(".,;:\"()") for w in dimension.split()]
        sig_words = [w for w in words if len(w) >= 4 and w not in stopwords]
        keywords.extend(sig_words)

        dim_lower = dimension.lower()
        synonym_map = {
            "diagnosis": ["diagnostic", "diagnostics", "detection", "screening", "pathology", "imaging", "radiology"],
            "discovery": ["drug discovery", "drug development", "molecules", "clinical", "pharmaceutical", "therapeutic", "compound"],
            "privacy": ["patient privacy", "hipaa", "confidentiality", "gdpr", "consent", "protected health"],
            "assistant": ["copilot", "chatbot", "virtual assistant", "agent", "clinician assistant"],
            "benefit": ["advantage", "advantages", "value", "positive", "efficiency", "improvement", "saving"],
            "risk": ["hazard", "hazards", "threat", "danger", "harm", "adverse", "liability", "safety"],
            "challenge": ["hurdle", "obstacle", "barrier", "difficulty", "bottleneck", "concern", "limitation"],
            "implementation": ["real-world", "clinical practice", "deployed", "adoption", "in practice", "pilot", "hospital"],
            "encryption": ["cryptography", "cryptographic", "cipher", "lattice", "key exchange"],
            "signature": ["digital signature", "signing", "hash-based", "verification"],
            "standard": ["nist", "compliance", "specification", "benchmark", "guideline", "iso"],
            "cost": ["tco", "pricing", "expense", "capex", "opex", "economic", "budget"],
            "charging": ["charger", "battery swapping", "supercharger", "kwh", "grid load", "charging speed"]
        }
        for trigger, syns in synonym_map.items():
            if trigger in dim_lower:
                keywords.extend(syns)

        return list(dict.fromkeys(keywords))

    @classmethod
    def extract_dimensions_from_prompt(cls, user_question: str) -> List[str]:
        """
        Extracts requested evaluation and research dimensions directly from prompt instruction clauses.
        Identifies lists introduced by 'Analyze', 'Evaluate', 'Assess', 'Investigate', etc.
        """
        sentences = re.split(r'[.!?\n]+', user_question)
        dim_candidates: List[str] = []

        dim_pattern = re.compile(
            r'(?:analyze|evaluate|assess|investigate|examine|explore|review|focus(?:ing)?\s+on|covering|include|including|consider(?:ing)?)\s+([^.]+)',
            re.IGNORECASE
        )

        for s in sentences:
            s = s.strip()
            if not s:
                continue
            m = dim_pattern.search(s)
            if m:
                clause = m.group(1).strip()
                # Skip if clause is an entity relationship or overarching topic statement rather than an aspect list
                if re.search(r'\b(?:impact|role|effect|how|adoption|transformation|evolution)\b', clause, re.IGNORECASE):
                    continue
                # If clause has no commas and no conjunctions, verify if it is an explicit predefined dimension
                if ',' not in clause and ' and ' not in clause.lower() and '&' not in clause:
                    matched_known = any(
                        clause.lower() == d.lower() or any(clause.lower() == kw for kw in kws[:2])
                        for d, kws in cls.DIMENSION_KEYWORDS.items()
                    )
                    if not matched_known:
                        continue
                parts = re.split(r',|\band\b|&', clause)
                for p in parts:
                    cleaned = re.sub(r'^(?:and|or|as\s+well\s+as)\s+', '', p.strip(), flags=re.IGNORECASE).strip()
                    cleaned = cleaned.strip(".,;:'\"()")
                    if cleaned and len(cleaned) >= 3 and cleaned.lower() not in {'etc', 'more'}:
                        # Map to predefined canonical dimension if exact match exists
                        matched_predefined = None
                        for d, kws in cls.DIMENSION_KEYWORDS.items():
                            if cleaned.lower() == d.lower() or any(cleaned.lower() == kw for kw in kws[:3]):
                                matched_predefined = d
                                break
                        if matched_predefined:
                            if matched_predefined not in dim_candidates:
                                dim_candidates.append(matched_predefined)
                        else:
                            title_dim = cleaned.title()
                            for acr in ['Ai', 'Api', 'Rag', 'Nist', 'Llm', 'Saas', 'Iot']:
                                title_dim = re.sub(rf'\b{acr}\b', acr.upper(), title_dim)
                            if title_dim not in dim_candidates:
                                dim_candidates.append(title_dim)

        return dim_candidates

    @classmethod
    def extract_entities_and_dimensions(
        cls,
        user_question: str,
        tasks_planned: Optional[List[Any]] = None,
        plan_entities: Optional[List[str]] = None,
        plan_dimensions: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Extracts researched entities and relevant evaluation dimensions from the query and plan.
        Preserves multi-word entity names (e.g. 'Microsoft AutoGen') and prevents instruction verbs ('Evaluate')
        from becoming entities. Supports both software framework comparisons and general domain research queries.
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
                is_start_boundary = (pos == 0 or not question_lower[pos - 1].isalnum())
                is_end_boundary = (end_pos == len(question_lower) or not question_lower[end_pos].isalnum())

                if is_start_boundary and is_end_boundary:
                    if not any(s <= pos and end_pos <= e for s, e, _ in matched_items):
                        canonical = cls.CANONICAL_ENTITIES[key]
                        if not any(m[2] == canonical for m in matched_items):
                            matched_items.append((pos, end_pos, canonical))
                idx = end_pos

        matched_items.sort(key=lambda x: x[0])
        entities: List[str] = [m[2] for m in matched_items]

        # 2. Extract dimensions first so dimension clauses are not mistaken for entities
        explicit_dims = cls.extract_dimensions_from_prompt(user_question)
        dimensions: List[str] = explicit_dims.copy()

        # If plan provided explicit dimensions, incorporate them
        if plan_dimensions:
            for pd in plan_dimensions:
                if pd not in dimensions:
                    dimensions.append(pd)

        # 3. If canonical entities not found or fewer than 2, search syntactic relationship patterns
        if len(entities) < 2:
            # Pattern A: 'how <X> is transforming/impacting/changing/revolutionizing <Y>'
            m_how = re.search(
                r'how\s+([A-Za-z0-9\s\-]+?)\s+is\s+(?:transforming|impacting|changing|disrupting|revolutionizing|reshaping|affecting|enhancing|driving)\s+(?:the\s+)?([A-Za-z0-9\s\-]+?)(?:\s+in\s+\d{4}|\s+in\s+the|\.|\?|;|$)',
                user_question,
                re.IGNORECASE
            )
            if m_how:
                e1 = cls._clean_entity_name(m_how.group(1))
                e2 = cls._clean_entity_name(m_how.group(2))
                if e1 and e1.lower() not in cls.NON_ENTITY_WORDS and e1 not in entities:
                    entities.append(e1)
                if e2 and e2.lower() not in cls.NON_ENTITY_WORDS and e2 not in entities:
                    entities.append(e2)

            # Pattern B: 'impact/role/effect/influence/application(s) of <X> on/in <Y>'
            if len(entities) < 2:
                m_impact = re.search(
                    r'(?:impact|role|effect|influence|application(?:s)?)\s+of\s+([A-Za-z0-9\s\-]+?)\s+(?:on|in)\s+(?:the\s+)?([A-Za-z0-9\s\-]+?)(?:\s+in\s+\d{4}|\s+in\s+the|\.|\?|;|$)',
                    user_question,
                    re.IGNORECASE
                )
                if m_impact:
                    e1 = cls._clean_entity_name(m_impact.group(1))
                    e2 = cls._clean_entity_name(m_impact.group(2))
                    if e1 and e1.lower() not in cls.NON_ENTITY_WORDS and e1 not in entities:
                        entities.append(e1)
                    if e2 and e2.lower() not in cls.NON_ENTITY_WORDS and e2 not in entities:
                        entities.append(e2)

            # Pattern C: 'adoption/integration/implementation/use of <X> in/across <Y>'
            # OR '<X> adoption/integration/implementation/deployment in/across <Y>'
            if len(entities) < 2:
                m_adopt = re.search(
                    r'(?:(?:adoption|integration|implementation|deployment|use)\s+of\s+([A-Za-z0-9\s\-]+?)\s+(?:in|across|for)\s+([A-Za-z0-9\s\-]+?)|([A-Za-z0-9\s\-]+?)\s+(?:adoption|integration|implementation|deployment)\s+(?:in|across|for)\s+([A-Za-z0-9\s\-]+?))(?:\.|\?|;|$)',
                    user_question,
                    re.IGNORECASE
                )
                if m_adopt:
                    raw_e1 = m_adopt.group(1) or m_adopt.group(3)
                    raw_e2 = m_adopt.group(2) or m_adopt.group(4)
                    e1 = cls._clean_entity_name(raw_e1) if raw_e1 else ""
                    e2 = cls._clean_entity_name(raw_e2) if raw_e2 else ""
                    if e1 and e1.lower() not in cls.NON_ENTITY_WORDS and e1 not in entities:
                        entities.append(e1)
                    if e2 and e2.lower() not in cls.NON_ENTITY_WORDS and e2 not in entities:
                        entities.append(e2)

            # Pattern D: 'compare/versus/between <X>, <Y>, and <Z>'
            if len(entities) < 2:
                m_compare = re.search(
                    r'(?:compare|comparison\s+of|versus|vs\.?|between)\s+([^.]+?)(?:\s+for\b|\s+in\b|\.|\?|;|$)',
                    user_question,
                    re.IGNORECASE
                )
                if m_compare:
                    clause = m_compare.group(1)
                    parts = re.split(r',|\band\b|&|\bvs\.?\b', clause)
                    for p in parts:
                        cleaned = cls._clean_entity_name(p)
                        if cleaned and cleaned.lower() not in cls.NON_ENTITY_WORDS and cleaned not in entities:
                            entities.append(cleaned)

        # 4. If plan provided entities, incorporate valid ones
        if plan_entities:
            for pe in plan_entities:
                cleaned_pe = cls._clean_entity_name(pe)
                if cleaned_pe and cleaned_pe.lower() not in cls.NON_ENTITY_WORDS and cleaned_pe not in entities:
                    entities.append(cleaned_pe)

        # 5. Capitalized multi-word noun phrase extraction fallback
        if len(entities) < 2:
            noun_phrases = re.findall(
                r'\b(?:[A-Z][a-zA-Z0-9]*(?:-[A-Za-z0-9]+)?(?:\s+(?:[A-Z][a-zA-Z0-9]*(?:-[A-Za-z0-9]+)?|[A-Z]{2,}))+)\b',
                user_question
            )
            for np in noun_phrases:
                cleaned = cls._clean_entity_name(np)
                first_word = cleaned.split()[0].lower() if cleaned else ""
                if cleaned and first_word not in cls.NON_ENTITY_WORDS and cleaned not in entities:
                    if not any(cleaned.lower() == d.lower() for d in dimensions):
                        entities.append(cleaned)

        # Fallback if no entity was detected
        if not entities:
            entities = ["Primary Research Topic"]

        # Incorporate aspects from plan tasks if available and dimensions are sparse
        if tasks_planned and len(dimensions) < 4:
            for task in tasks_planned:
                aspects = getattr(task, "aspects_to_investigate", [])
                for asp in aspects:
                    if asp not in dimensions and len(dimensions) < 6:
                        dimensions.append(asp)

        # Disambiguate: If 'Tool Integration' was requested, avoid redundant generic 'Integrations'
        if "Tool Integration" in dimensions and "Integrations" in dimensions:
            dimensions.remove("Integrations")

        # If user did NOT specify explicit dimensions, supply contextual defaults
        if len(dimensions) < 3:
            is_software = any(w in question_lower for w in [
                "framework", "frameworks", "library", "libraries", "api", "backend",
                "fastapi", "django", "flask", "langchain", "llamaindex", "haystack",
                "langgraph", "crewai", "autogen", "database", "database integration"
            ])
            if is_software:
                default_dims = ["Architecture", "Features", "Performance", "Deployment", "Limitations"]
            else:
                default_dims = ["Current State", "Key Applications", "Benefits", "Risks & Challenges"]
            for d in default_dims:
                if d not in dimensions and len(dimensions) < 5:
                    dimensions.append(d)

        return entities, dimensions

    @classmethod
    def _entity_matches_text(cls, entity: str, text: str) -> bool:
        """Determines if an entity is referenced in text, supporting multi-word variants and aliases."""
        e_low = entity.lower().strip()
        if not e_low:
            return False

        if e_low in text:
            return True

        if e_low == "microsoft autogen":
            return "microsoft autogen" in text or "autogen" in text

        if e_low in ("generative ai", "genai", "gen ai"):
            return "generative ai" in text or "genai" in text or "gen ai" in text or ("generative" in text and "ai" in text)

        if e_low == "healthcare industry":
            return "healthcare" in text or "health care" in text or "medical" in text or "clinical" in text

        # Multi-word entity matching: all significant words present in text
        words = [w for w in re.findall(r'[a-zA-Z0-9]+', e_low) if len(w) >= 3 and w not in {"the", "and", "for", "with"}]
        if len(words) >= 2 and all(w in text for w in words):
            return True

        return False

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
                dim_keywords = cls.get_dimension_keywords(dimension)

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
        missing_entities: Optional[List[str]] = None,
        user_question: str = ""
    ) -> List[str]:
        """
        Generates highly focused, targeted queries exclusively for missing (entity, dimension) gaps.
        Prioritizes completely missing entities and interleaves across entities to prevent starvation.
        Produces context-aware, topic-independent queries rather than generic boilerplates.
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

        q_low = user_question.lower()
        is_software = any(re.search(rf'\b{re.escape(w)}\b', q_low) for w in [
            "framework", "frameworks", "library", "libraries", "api", "backend", "saas",
            "sdk", "codebase", "official documentation", "database", "orm"
        ])
        software_known = {
            "fastapi", "django", "flask", "langchain", "llamaindex", "haystack",
            "langgraph", "crewai", "autogen", "microsoft autogen", "qdrant", "milvus", "chroma"
        }

        # Extract domain context if present in user_question
        domain_terms = [
            "healthcare", "cryptography", "logistics", "finance", "automotive",
            "security", "energy", "medicine", "robotics", "telecom", "pharma"
        ]
        domain_hint = ""
        for dt in domain_terms:
            if dt in q_low:
                domain_hint = dt
                break

        # Interleave across entities (round-robin)
        max_depth = max(len(dims) for dims in entity_dims.values()) if entity_dims else 0
        for depth in range(max_depth):
            for entity in sorted_entities:
                dims = entity_dims[entity]
                if depth < len(dims):
                    dim = dims[depth]
                    # Formulate context-aware query
                    if is_software or entity.lower() in software_known:
                        candidate = f"{entity} {dim.lower()} official documentation guide"
                    elif domain_hint and domain_hint not in entity.lower() and domain_hint not in dim.lower():
                        candidate = f"{entity} {dim.lower()} {domain_hint} research analysis"
                    else:
                        candidate = f"{entity} {dim.lower()} research findings"

                    if candidate.lower() not in existing_lowered and candidate not in queries:
                        queries.append(candidate)
                    if len(queries) >= max_queries:
                        return queries

        # Fallback with alternative formulations if slots remain
        for depth in range(max_depth):
            for entity in sorted_entities:
                dims = entity_dims[entity]
                if depth < len(dims):
                    dim = dims[depth]
                    if is_software or entity.lower() in software_known:
                        alt = f"{entity} {dim.lower()} architecture benchmarks"
                    elif domain_hint and domain_hint not in entity.lower() and domain_hint not in dim.lower():
                        alt = f"{entity} {dim.lower()} {domain_hint} real world implementations"
                    else:
                        alt = f"{entity} {dim.lower()} analysis and challenges"

                    if alt.lower() not in existing_lowered and alt not in queries:
                        queries.append(alt)
                    if len(queries) >= max_queries:
                        return queries

        return queries
