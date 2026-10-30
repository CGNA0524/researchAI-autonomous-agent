"""
Confidence and Limitation Assessment Handler.
Computes calibrated, deterministic confidence ratings based on mathematical evidence metrics,
source authority tiers, entity/dimension coverage ratios, and contradiction penalties.
The LLM does NOT independently decide confidence.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from app.models.response import ConfidenceLevel
from app.agent.coverage import EvidenceCoverageReport
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ConfidenceAssessor:
    """Evaluates evidence completeness, source credibility, and sets calibrated confidence levels."""

    @classmethod
    def evaluate(
        cls,
        search_results: List[Dict[str, Any]],
        extracted_pages: List[Dict[str, Any]],
        evidence_snippets: List[str],
        coverage_report: Optional[EvidenceCoverageReport] = None,
        contradictions: Optional[List[str]] = None,
        return_explanation: bool = False
    ) -> Union[Tuple[str, List[str], List[str]], Tuple[str, List[str], List[str], str]]:
        """
        Deterministically calculates confidence level, assumptions, and limitations.
        If return_explanation is True, returns 4-tuple including confidence_explanation.
        Otherwise returns backward-compatible 3-tuple (confidence, assumptions, limitations).
        """
        valid_sources = len(search_results)
        successful_extractions = len([p for p in extracted_pages if p.get("success")])
        total_evidence_volume = sum(len(s) for s in evidence_snippets)
        unresolved_contradictions = contradictions or []

        assumptions: List[str] = [
            "Information is current based on publicly available documentation and technical publications indexed up to present.",
            "Benchmark and performance assertions assume standard production hardware environments unless otherwise noted."
        ]
        limitations: List[str] = []

        # ----------------------------------------------------------------------
        # 1. Zero Sources or Empty Evidence -> INSUFFICIENT_EVIDENCE
        # ----------------------------------------------------------------------
        if valid_sources == 0 or total_evidence_volume == 0:
            confidence = ConfidenceLevel.INSUFFICIENT_EVIDENCE.value
            limitations.append("No active web sources or search results could be retrieved for this query.")
            limitations.append("Conclusions cannot be substantiated without external evidence.")
            explanation = (
                "INSUFFICIENT_EVIDENCE: Zero external sources were retrieved. "
                "The agent refuses to fabricate findings without primary evidence."
            )
            if return_explanation:
                return confidence, assumptions, limitations, explanation
            return confidence, assumptions, limitations

        # ----------------------------------------------------------------------
        # 2. Extract Authority Breakdown
        # ----------------------------------------------------------------------
        official_count = sum(1 for s in search_results if s.get("is_official") or s.get("category") == "OFFICIAL_DOCS")
        academic_count = sum(1 for s in search_results if s.get("category") == "ACADEMIC_PAPER")
        reputable_count = sum(1 for s in search_results if s.get("category") == "REPUTABLE_PUBLICATION")
        authoritative_sources = official_count + academic_count + reputable_count

        # If categories were not annotated (e.g. legacy search results), treat multi-source as having baseline authority
        has_category_annotations = any("category" in s or "is_official" in s for s in search_results)
        if not has_category_annotations and valid_sources >= 2:
            authoritative_sources = max(1, authoritative_sources)

        # ----------------------------------------------------------------------
        # 3. Coverage Metrics
        # ----------------------------------------------------------------------
        coverage_ratio = coverage_report.coverage_ratio if coverage_report else (0.85 if valid_sources >= 2 else 0.30)
        missing_entities = coverage_report.missing_entities if coverage_report else []
        missing_dimensions = coverage_report.missing_dimensions if coverage_report else []

        if coverage_report and coverage_report.limitations_identified:
            limitations.extend(coverage_report.limitations_identified)

        # ----------------------------------------------------------------------
        # 4. Deterministic Point Scoring System (0 to 100)
        # ----------------------------------------------------------------------
        # A. Source Count & Diversity (max 25 pts)
        if valid_sources >= 4:
            source_pts = 25.0
        elif valid_sources >= 2:
            source_pts = 18.0
        else:
            source_pts = 8.0

        # B. Authority Tier (max 25 pts)
        if authoritative_sources >= 2:
            authority_pts = 25.0
        elif authoritative_sources == 1:
            authority_pts = 16.0
        else:
            authority_pts = 8.0  # purely community blogs

        # C. Evidence Coverage Ratio (max 30 pts)
        coverage_pts = coverage_ratio * 30.0

        # D. Extraction Depth & Text Volume (max 20 pts)
        if successful_extractions >= 2 and total_evidence_volume >= 2000:
            extraction_pts = 20.0
        elif successful_extractions >= 1 and total_evidence_volume >= 1000:
            extraction_pts = 20.0
        elif successful_extractions >= 1 and total_evidence_volume >= 400:
            extraction_pts = 12.0
        else:
            extraction_pts = 5.0
            limitations.append("Synthesis relies primarily on search snippets rather than verified full-text page extractions.")

        # Total Raw Score
        raw_score = source_pts + authority_pts + coverage_pts + extraction_pts

        # Penalties
        if missing_entities:
            raw_score -= (len(missing_entities) * 8.0)
        if unresolved_contradictions:
            raw_score -= 20.0
            limitations.append(f"Contradictory claims detected across sources: {'; '.join(unresolved_contradictions)}")

        final_score = max(0.0, min(100.0, raw_score))

        # ----------------------------------------------------------------------
        # 5. Calibrated Threshold Assignment
        # ----------------------------------------------------------------------
        # HIGH requires:
        # - final_score >= 75
        # - coverage_ratio >= 0.70
        # - at least 1 authoritative source
        # - at least 1 successful deep extraction
        # - NO missing entities
        # - NO unresolved contradictions
        if (
            final_score >= 75.0
            and coverage_ratio >= 0.70
            and authoritative_sources >= 1
            and successful_extractions >= 1
            and not missing_entities
            and not unresolved_contradictions
        ):
            confidence = ConfidenceLevel.HIGH.value
            assumptions.append("Synthesized consensus across multiple authoritative technical publications and documentation.")
            explanation = (
                f"HIGH (Score: {final_score:.1f}/100): Strong evidence coverage ({int(coverage_ratio*100)}%), "
                f"{authoritative_sources} authoritative sources, {successful_extractions} deep page extractions, "
                f"and zero critical entity gaps."
            )

        elif final_score >= 45.0 and coverage_ratio >= 0.40:
            confidence = ConfidenceLevel.MEDIUM.value
            if missing_dimensions:
                limitations.append(f"Moderate coverage: evidence incomplete for dimensions {missing_dimensions}.")
            explanation = (
                f"MEDIUM (Score: {final_score:.1f}/100): Moderate coverage ({int(coverage_ratio*100)}%), "
                f"{valid_sources} sources retrieved ({authoritative_sources} authoritative). "
                f"Evidence is informative but exhibits minor dimension gaps."
            )

        elif final_score >= 15.0:
            confidence = ConfidenceLevel.LOW.value
            limitations.append("Limited or fragmented supporting evidence; cross-verification across independent sources was weak.")
            explanation = (
                f"LOW (Score: {final_score:.1f}/100): Limited coverage ({int(coverage_ratio*100)}%), "
                f"{valid_sources} source(s) with partial verification."
            )

        else:
            confidence = ConfidenceLevel.INSUFFICIENT_EVIDENCE.value
            limitations.append("Retrieved information lacks substantive technical depth to support definitive comparative conclusions.")
            explanation = (
                f"INSUFFICIENT_EVIDENCE (Score: {final_score:.1f}/100): Critical evidence is missing or unverified. "
                f"Confidence downgraded to safeguard against ungrounded speculation."
            )

        logger.info(f"Programmatic confidence evaluated: {confidence} | Score: {final_score:.1f} | Coverage: {coverage_ratio:.2f}")
        if return_explanation:
            return confidence, assumptions, limitations, explanation
        return confidence, assumptions, limitations
