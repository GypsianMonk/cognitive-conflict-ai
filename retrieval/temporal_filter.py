"""
Temporal Awareness — Recency-Aware Retrieval
RAG results are filtered and penalized by staleness.
Agents know when their sources were published and flag outdated knowledge.
Critical for finance, medicine, and technology domains.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from retrieval.hybrid_retriever import Document


@dataclass
class TemporalScore:
    document_id: str
    published_date: Optional[str]
    age_days: Optional[float]
    freshness_score: float          # 1.0 = very fresh, 0.0 = very stale
    is_flagged: bool
    flag_reason: str


DOMAIN_STALENESS_THRESHOLDS = {
    "finance":     30,      # Stale after 30 days
    "technology":  90,      # Stale after 3 months
    "medicine":    180,     # Stale after 6 months
    "law":         365,     # Stale after 1 year
    "science":     730,     # Stale after 2 years
    "general":     365,     # Stale after 1 year
}

DATE_PATTERNS = [
    r"(\d{4})-(\d{2})-(\d{2})",                    # 2024-01-15
    r"(\w+ \d{1,2},? \d{4})",                       # January 15, 2024
    r"(\d{1,2}/\d{1,2}/\d{4})",                     # 01/15/2024
    r"(\d{4})",                                      # Just a year: 2024
]


class TemporalFilter:
    """
    Analyzes document freshness and applies staleness penalties.
    """

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.stale_threshold_days = DOMAIN_STALENESS_THRESHOLDS.get(domain, 365)
        self.now = datetime.now(timezone.utc)

    def score_document(self, doc: Document) -> TemporalScore:
        published = doc.published_date or self._extract_date_from_content(doc.content)
        age_days = self._calculate_age(published)
        freshness = self._freshness_score(age_days)
        is_flagged, reason = self._should_flag(age_days, freshness)

        return TemporalScore(
            document_id=doc.id,
            published_date=published,
            age_days=age_days,
            freshness_score=freshness,
            is_flagged=is_flagged,
            flag_reason=reason,
        )

    def filter_and_rank(self, docs: list[Document]) -> list[tuple[Document, TemporalScore]]:
        """Score all docs by freshness and return sorted (freshest first)."""
        scored = [(doc, self.score_document(doc)) for doc in docs]
        scored.sort(key=lambda x: x[1].freshness_score, reverse=True)
        return scored

    def build_staleness_warning(self, scores: list[TemporalScore]) -> str:
        flagged = [s for s in scores if s.is_flagged]
        if not flagged:
            return ""
        parts = [f"⚠ TEMPORAL WARNING: {len(flagged)} source(s) may be outdated."]
        for s in flagged:
            parts.append(f"  • {s.document_id}: {s.flag_reason}")
        return "\n".join(parts)

    def _extract_date_from_content(self, content: str) -> Optional[str]:
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, content)
            if match:
                return match.group(0)
        return None

    def _calculate_age(self, date_str: Optional[str]) -> Optional[float]:
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%B %d %Y", "%m/%d/%Y", "%Y"):
            try:
                parsed = datetime.strptime(date_str.strip(), fmt)
                parsed = parsed.replace(tzinfo=timezone.utc)
                return (self.now - parsed).days
            except ValueError:
                continue
        return None

    def _freshness_score(self, age_days: Optional[float]) -> float:
        if age_days is None:
            return 0.5     # Unknown date — neutral score
        if age_days <= 0:
            return 1.0     # Future date (just published)
        # Exponential decay: half-life = threshold
        import math
        half_life = self.stale_threshold_days
        return round(math.exp(-0.693 * age_days / half_life), 4)

    def _should_flag(self, age_days: Optional[float], freshness: float) -> tuple[bool, str]:
        if age_days is None:
            return False, ""
        if age_days > self.stale_threshold_days * 2:
            return True, (
                f"Source is {int(age_days)} days old — "
                f"very stale for {self.domain} (threshold: {self.stale_threshold_days} days)."
            )
        if freshness < 0.3:
            return True, f"Low freshness score ({freshness:.2f}) for {self.domain} domain."
        return False, ""
