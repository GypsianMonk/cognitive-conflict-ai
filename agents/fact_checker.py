"""
Claim Verification Agent
A dedicated fact-checker that runs in parallel with the debate.
Extracts claims from agent responses and verifies them.
Claims that fail verification are flagged and penalized in scoring.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from agents.tool_agent import WebSearchTool, WikipediaTool, ToolResult


@dataclass
class VerifiedClaim:
    claim: str
    agent_name: str
    status: str             # verified | unverified | disputed | unverifiable
    evidence: str           # Supporting or refuting text
    confidence: float


@dataclass
class FactCheckReport:
    verified: list[VerifiedClaim] = field(default_factory=list)
    unverified: list[VerifiedClaim] = field(default_factory=list)
    disputed: list[VerifiedClaim] = field(default_factory=list)
    overall_credibility: float = 1.0

    def to_dict(self) -> dict:
        return {
            "overall_credibility": round(self.overall_credibility, 3),
            "verified_count": len(self.verified),
            "unverified_count": len(self.unverified),
            "disputed_count": len(self.disputed),
            "disputes": [
                {"claim": c.claim[:100], "agent": c.agent_name, "evidence": c.evidence[:150]}
                for c in self.disputed
            ],
        }


VERIFIABLE_SIGNALS = [
    r"\d{4}",                            # Years
    r"\d+(\.\d+)?%",                     # Percentages
    r"\$\d+",                            # Dollar amounts
    r"according to",
    r"studies show",
    r"research (shows|indicates|found)",
    r"(was|is|are) invented",
    r"(was|is) discovered",
]


class FactChecker:
    """
    Fact-checking pipeline:
    1. Extract verifiable claims from agent responses
    2. Search web/Wikipedia for corroborating or contradicting evidence
    3. Score overall credibility
    """

    def __init__(self):
        self.web = WebSearchTool()
        self.wiki = WikipediaTool()

    def verify_responses(self, responses) -> FactCheckReport:
        report = FactCheckReport()

        for response in responses:
            claims = self._extract_claims(response.answer)
            for claim in claims:
                result = self._verify_claim(claim, response.agent_name)
                if result.status == "verified":
                    report.verified.append(result)
                elif result.status == "disputed":
                    report.disputed.append(result)
                else:
                    report.unverified.append(result)

        # Credibility = ratio of verified to total verifiable claims
        total = len(report.verified) + len(report.unverified) + len(report.disputed)
        if total > 0:
            penalty = len(report.disputed) * 0.3 + len(report.unverified) * 0.1
            report.overall_credibility = max(0.0, 1.0 - penalty / total)

        return report

    def _extract_claims(self, text: str) -> list[str]:
        """Extract sentences that contain verifiable signals."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []
        for sentence in sentences:
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in VERIFIABLE_SIGNALS):
                claims.append(sentence.strip())
        return claims[:3]   # Limit to 3 claims per agent to avoid rate limits

    def _verify_claim(self, claim: str, agent_name: str) -> VerifiedClaim:
        # Extract the key entity from the claim for searching
        search_query = self._extract_search_query(claim)

        # Try Wikipedia first
        wiki_result: ToolResult = self.wiki.run(search_query)
        if wiki_result.success and wiki_result.output:
            status = self._cross_reference(claim, wiki_result.output)
            return VerifiedClaim(
                claim=claim,
                agent_name=agent_name,
                status=status,
                evidence=wiki_result.output[:200],
                confidence=0.8 if status == "verified" else 0.5,
            )

        # Fallback to web search
        web_result: ToolResult = self.web.run(claim[:100])
        if web_result.success and web_result.output:
            status = self._cross_reference(claim, web_result.output)
            return VerifiedClaim(
                claim=claim,
                agent_name=agent_name,
                status=status,
                evidence=web_result.output[:200],
                confidence=0.6,
            )

        return VerifiedClaim(
            claim=claim,
            agent_name=agent_name,
            status="unverifiable",
            evidence="No external source found.",
            confidence=0.3,
        )

    def _extract_search_query(self, claim: str) -> str:
        # Strip common sentence starters
        for prefix in ["According to ", "Research shows that ", "Studies show that "]:
            claim = claim.replace(prefix, "")
        # Take first 60 chars as search query
        return claim[:60].strip()

    def _cross_reference(self, claim: str, evidence: str) -> str:
        """
        Simple heuristic cross-reference.
        In production: use LLM to judge whether evidence supports or refutes the claim.
        """
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())
        overlap = claim_words & evidence_words
        overlap_ratio = len(overlap) / max(1, len(claim_words))

        if overlap_ratio > 0.4:
            return "verified"
        elif overlap_ratio > 0.2:
            return "unverified"
        else:
            return "disputed"
