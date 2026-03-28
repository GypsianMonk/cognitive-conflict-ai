"""
Query Understanding Layer — Analyzes incoming queries to determine
complexity, domain, and optimal reasoning configuration.
"""

from dataclasses import dataclass
from typing import Optional
import re


DOMAIN_KEYWORDS = {
    "finance": ["stock", "investment", "portfolio", "revenue", "profit", "trading", "market", "crypto", "finance"],
    "medicine": ["disease", "symptom", "treatment", "drug", "clinical", "health", "patient", "diagnosis", "medical"],
    "law": ["legal", "law", "contract", "regulation", "court", "compliance", "liability", "statute", "rights"],
    "technology": ["code", "software", "algorithm", "api", "database", "cloud", "machine learning", "ai", "system"],
    "science": ["hypothesis", "experiment", "research", "data", "theory", "study", "evidence", "scientific"],
    "philosophy": ["ethics", "moral", "consciousness", "existence", "truth", "reality", "belief", "knowledge"],
}

COMPLEXITY_INDICATORS = {
    "simple": ["what is", "define", "who is", "when did", "where is"],
    "medium": ["how does", "explain", "compare", "why does", "what are the"],
    "complex": ["analyze", "evaluate", "should i", "design", "what is the best", "trade-off", "implications of"],
}


@dataclass
class QueryAnalysis:
    query: str
    complexity: str          # simple | medium | complex
    domain: Optional[str]    # detected domain or None
    requires_expert: bool
    debate_rounds: int
    reasoning_depth: str     # shallow | standard | deep


class QueryUnderstanding:
    """
    Analyzes a query and returns a structured analysis that drives
    how many agents are used, how many debate rounds to run, etc.
    """

    def analyze(self, query: str) -> QueryAnalysis:
        query_lower = query.lower()

        complexity = self._detect_complexity(query_lower)
        domain = self._detect_domain(query_lower)
        requires_expert = domain is not None and complexity in ("medium", "complex")
        debate_rounds = self._determine_debate_rounds(complexity)
        reasoning_depth = self._determine_reasoning_depth(complexity)

        return QueryAnalysis(
            query=query,
            complexity=complexity,
            domain=domain,
            requires_expert=requires_expert,
            debate_rounds=debate_rounds,
            reasoning_depth=reasoning_depth,
        )

    def _detect_complexity(self, query: str) -> str:
        for level, indicators in COMPLEXITY_INDICATORS.items():
            if any(ind in query for ind in indicators):
                return level
        # Fallback: use query length as proxy
        word_count = len(query.split())
        if word_count < 8:
            return "simple"
        elif word_count < 20:
            return "medium"
        return "complex"

    def _detect_domain(self, query: str) -> Optional[str]:
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                return domain
        return None

    def _determine_debate_rounds(self, complexity: str) -> int:
        return {"simple": 1, "medium": 2, "complex": 3}.get(complexity, 2)

    def _determine_reasoning_depth(self, complexity: str) -> str:
        return {"simple": "shallow", "medium": "standard", "complex": "deep"}.get(complexity, "standard")
