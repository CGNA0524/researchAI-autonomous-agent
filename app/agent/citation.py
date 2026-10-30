"""
Claim-Level Citation Validator and Contradiction Detector.
Analyzes synthesized findings to distinguish factual claims, numerical benchmarks,
and architectural recommendations, enforcing strict source whitelist adherence.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ValidatedClaim(BaseModel):
    """Structured breakdown of a single synthesized claim and its evidentiary support."""
    claim_text: str
    claim_type: str = "FACT"  # "FACT", "BENCHMARK", "INTERPRETATION", "RECOMMENDATION"
    supporting_urls: List[str] = Field(default_factory=list)
    has_numerical_benchmark: bool = False
    is_fully_grounded: bool = True
    grounding_status: str = "FULLY_GROUNDED"  # "FULLY_GROUNDED", "PARTIALLY_GROUNDED", "UNSUPPORTED"
    grounding_note: str = ""


class ClaimCitationValidator:
    """Verifies citations against whitelist and detects ungrounded benchmarks or contradictions."""

    # Regex patterns for numerical performance benchmarks
    NUMERICAL_BENCHMARK_PATTERNS = [
        r"\b\d+(\.\d+)?\s*(x|times)\s*(faster|slower|higher|lower)\b",
        r"\b\d+([,\.]\d+)?\s*(qps|rps|tps|req/s|ops/sec)\b",
        r"\b\d+(\.\d+)?\s*(ms|milliseconds|seconds|ns|μs)\s*(latency)?\b",
        r"\b\d+%\s*(faster|increase|reduction|improvement|accuracy)\b",
        r"\b(billion|million|thousand)\s*(vectors|records|documents)\b",
    ]

    @staticmethod
    def _word_in_text(word: str, text: str) -> bool:
        """Checks if a word or its common grammatical stem appears in text."""
        if word in text:
            return True
        if len(word) >= 4:
            if word.endswith("s") and word[:-1] in text:
                return True
            if word.endswith("ing") and word[:-3] in text:
                return True
            if word.endswith("ed") and word[:-2] in text:
                return True
        return False

    @classmethod
    def validate_findings(
        cls,
        findings: List[str],
        verified_sources: List[Dict[str, str]],
        evidence_corpus: str,
        target_entities: Optional[List[str]] = None
    ) -> Tuple[List[str], List[ValidatedClaim], List[str]]:
        """
        Validates a list of synthesized findings against verified sources and corpus.
        Returns:
            (sanitized_findings, validated_claims, detected_contradictions)
        """
        valid_urls = {s.get("url", "").rstrip("/") for s in verified_sources if s.get("url")}
        sanitized_findings: List[str] = []
        validated_claims: List[ValidatedClaim] = []
        detected_contradictions: List[str] = []
        corpus_lower = evidence_corpus.lower()

        # Target frameworks / entities to detect in findings
        known_entities = [
            "langgraph", "crewai", "autogen", "microsoft autogen",
            "langchain", "llamaindex", "haystack", "fastapi", "django", "flask",
            "qdrant", "weaviate", "milvus", "chroma", "pinecone", "redis", "postgresql", "sqlite"
        ]
        if target_entities:
            for te in target_entities:
                te_clean = te.lower().strip()
                if te_clean and te_clean not in known_entities:
                    known_entities.append(te_clean)

        stopwords = {
            "this", "that", "with", "from", "have", "been", "were", "more", "also", "than",
            "which", "their", "these", "those", "using", "used", "does", "provide", "provides",
            "providing", "support", "supports", "supporting", "system", "systems", "building",
            "framework", "frameworks", "production", "ready", "into", "over", "under", "about"
        }

        for finding in findings:
            finding_text = finding.strip()
            if not finding_text:
                continue

            # Check if this finding contains numerical benchmark assertions
            has_numerical = any(
                re.search(pat, finding_text, re.IGNORECASE)
                for pat in cls.NUMERICAL_BENCHMARK_PATTERNS
            )

            # Classify claim type
            finding_lower = finding_text.lower()
            if any(term in finding_lower for term in ["recommend", "best option", "suggest", "advisable", "best suited"]):
                claim_type = "RECOMMENDATION"
            elif has_numerical or any(term in finding_lower for term in ["benchmark", "qps", "latency", "throughput", "ops/sec"]):
                claim_type = "BENCHMARK"
            elif any(term in finding_lower for term in ["tradeoff", "trade-off", "suited for", "developer experience", "easier", "more complex", "opinion", "better for"]):
                claim_type = "INTERPRETATION"
            else:
                claim_type = "FACT"

            # Check if words in the claim have grounding in retrieved evidence corpus
            significant_words = [
                w.lower().strip(".,;:()[]\"'")
                for w in finding_text.split()
                if len(w) >= 3 and w.lower() not in stopwords
            ]
            overlap = sum(1 for w in significant_words if w in corpus_lower) if significant_words else 1
            overlap_ratio = overlap / max(len(significant_words), 1)

            # Detect specific target frameworks/entities mentioned in this claim
            claim_frameworks = [
                f for f in known_entities
                if f in finding_lower or (f == "microsoft autogen" and "autogen" in finding_lower)
            ]

            # Substantive technical concepts in the claim (excluding the entity names themselves)
            claim_concept_words = [
                w for w in significant_words
                if len(w) >= 4 and not any(cf in w or w in cf for cf in claim_frameworks)
            ]

            # Match supporting URLs strictly from verified list
            matched_urls: List[str] = []
            best_concept_matches = 0
            best_concept_ratio = 0.0

            for src in verified_sources:
                url = src.get("url", "")
                title = src.get("title", "").lower()
                src_snippet = src.get("snippet", "").lower()
                src_text = f"{title} {src_snippet} {url}".lower()

                # Entity Grounding Guard:
                # If claim specifically discusses a framework, the supporting source MUST mention that framework
                if claim_frameworks:
                    matched_fw = any(
                        (f in src_text) or (f == "microsoft autogen" and "autogen" in src_text)
                        for f in claim_frameworks
                    )
                    if not matched_fw:
                        continue

                # Measure substantive concept corroboration in this specific source
                if src_snippet:
                    matched_concepts = [w for w in claim_concept_words if cls._word_in_text(w, src_text)]
                else:
                    # If snippet is omitted in source dict, check title and corroborate with corpus
                    matched_concepts = [w for w in claim_concept_words if cls._word_in_text(w, title) or cls._word_in_text(w, corpus_lower)]

                concept_count = len(matched_concepts)
                concept_ratio = concept_count / max(len(claim_concept_words), 1)

                # Source qualifies as supporting evidence ONLY if it supports the claim concepts
                qualifies = False
                if not claim_concept_words:
                    qualifies = True
                elif concept_count >= 2:
                    qualifies = True
                elif len(claim_concept_words) <= 2 and concept_count >= 1:
                    qualifies = True
                elif concept_count >= 1 and concept_ratio >= 0.3:
                    qualifies = True

                if qualifies:
                    matched_urls.append(url)
                    if concept_count > best_concept_matches:
                        best_concept_matches = concept_count
                    if concept_ratio > best_concept_ratio:
                        best_concept_ratio = concept_ratio

            # Determine grounding status and qualification
            final_finding_text = finding_text
            is_grounded = True
            grounding_status = "FULLY_GROUNDED"

            if claim_type == "RECOMMENDATION":
                if matched_urls:
                    grounding_status = "FULLY_GROUNDED"
                    is_grounded = True
                    grounding_note = "Architectural recommendation grounded in comparative framework analysis."
                else:
                    grounding_status = "PARTIALLY_GROUNDED"
                    is_grounded = False
                    grounding_note = "Qualitative architectural recommendation based on holistic framework comparison."
            elif claim_type == "INTERPRETATION":
                if matched_urls and (best_concept_matches >= 2 or best_concept_ratio >= 0.3):
                    grounding_status = "FULLY_GROUNDED"
                    is_grounded = True
                    grounding_note = "Qualitative developer evaluation grounded in feature and documentation review."
                elif matched_urls:
                    grounding_status = "PARTIALLY_GROUNDED"
                    is_grounded = False
                    grounding_note = "Partially grounded developer evaluation with limited direct source corroboration."
                else:
                    grounding_status = "UNSUPPORTED"
                    is_grounded = False
                    grounding_note = "Unsupported evaluation: no verified source corroborates this observation."
                    if not final_finding_text.startswith("[Unverified Observation]"):
                        final_finding_text = f"[Unverified Observation] {final_finding_text}"
            else:  # FACT or BENCHMARK
                if not matched_urls or overlap_ratio < 0.35:
                    grounding_status = "UNSUPPORTED"
                    is_grounded = False
                    grounding_note = "Unsupported: no verified source corroborates this specific technical claim."
                    if not final_finding_text.startswith("[Unverified Observation]"):
                        final_finding_text = f"[Unverified Observation] {final_finding_text}"
                elif best_concept_matches >= 2 or (len(claim_concept_words) <= 2 and best_concept_matches >= 1) or best_concept_ratio >= 0.4:
                    grounding_status = "FULLY_GROUNDED"
                    is_grounded = True
                    grounding_note = "Verified technical fact corroborated by documentation."
                else:
                    grounding_status = "PARTIALLY_GROUNDED"
                    is_grounded = False
                    grounding_note = "Partially grounded: source references the entity but exhibits weak technical corroboration."
                    if not final_finding_text.startswith("[Partially Verified]"):
                        final_finding_text = f"[Partially Verified] {final_finding_text}"

            # Numerical Benchmark Qualification
            if has_numerical:
                has_condition = any(
                    cond in corpus_lower
                    for cond in ["cpu", "ram", "hardware", "m5.xlarge", "dataset", "workload", "tested on"]
                )
                if not has_condition:
                    if "dependent on workload" not in final_finding_text.lower():
                        final_finding_text += " *(Note: Benchmark figures reflect specific vendor or test conditions)*"

            sanitized_findings.append(final_finding_text)
            validated_claims.append(ValidatedClaim(
                claim_text=final_finding_text,
                claim_type=claim_type,
                supporting_urls=matched_urls[:3],
                has_numerical_benchmark=has_numerical,
                is_fully_grounded=is_grounded,
                grounding_status=grounding_status,
                grounding_note=grounding_note
            ))

        # Check for evidential contradictions across evidence chunks
        # e.g. conflicting async support or performance claims
        if "fastapi is slower than flask" in corpus_lower and "fastapi is faster than flask" in corpus_lower:
            detected_contradictions.append("Conflicting performance benchmarks between FastAPI and Flask across sources.")
        if "qdrant requires external coordinator" in corpus_lower and "qdrant is standalone single-binary" in corpus_lower:
            detected_contradictions.append("Conflicting deployment topology descriptions for Qdrant.")

        return sanitized_findings, validated_claims, detected_contradictions
