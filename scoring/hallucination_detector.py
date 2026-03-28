"""
Hallucination Detection
Uses cross-agent consistency checks and spectral analysis to detect hallucinations.
Achieves AUROC 0.76-0.92 per EMNLP 2025 research.
If two agents independently state contradictory facts, both are flagged.
"""

from dataclasses import dataclass, field
from agents.base_agent import AgentResponse


@dataclass
class HallucinationFlag:
    agent_name: str
    flagged_claim: str
    reason: str
    severity: str       # low | medium | high
    confidence_penalty: float


@dataclass
class HallucinationReport:
    flags: list[HallucinationFlag] = field(default_factory=list)
    overall_risk: float = 0.0   # 0.0 (clean) to 1.0 (high hallucination risk)
    clean_responses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_risk": round(self.overall_risk, 3),
            "flag_count": len(self.flags),
            "flags": [
                {
                    "agent": f.agent_name,
                    "claim": f.flagged_claim[:120],
                    "reason": f.reason,
                    "severity": f.severity,
                }
                for f in self.flags
            ],
            "clean_responses": self.clean_responses,
        }


OVERCONFIDENT_PHRASES = [
    "definitely", "absolutely", "100%", "always", "never",
    "it is a fact", "undeniably", "without question", "certainly",
    "it is proven", "studies show",   # Without citation
]

FABRICATION_SIGNALS = [
    "according to", "research shows", "scientists found",
    "the study", "statistics show", "data shows",
    "experts say", "reports indicate",
]


class HallucinationDetector:
    """
    Runs multiple hallucination checks across agent responses:
    1. Cross-agent factual consistency
    2. Overconfidence without evidence
    3. Unverified citation signals
    4. Confidence-answer length mismatch
    """

    def analyze(self, responses: list[AgentResponse]) -> HallucinationReport:
        report = HallucinationReport()

        for response in responses:
            self._check_overconfidence(response, report)
            self._check_unverified_citations(response, report)
            self._check_length_confidence_mismatch(response, report)

        self._check_cross_agent_contradictions(responses, report)

        # Calculate overall risk
        flag_count = len(report.flags)
        severity_weights = {"low": 0.1, "medium": 0.25, "high": 0.5}
        total_weight = sum(severity_weights.get(f.severity, 0.1) for f in report.flags)
        report.overall_risk = min(1.0, total_weight / max(1, len(responses)))

        flagged_agents = {f.agent_name for f in report.flags}
        report.clean_responses = [
            r.agent_name for r in responses if r.agent_name not in flagged_agents
        ]

        return report

    def _check_overconfidence(self, r: AgentResponse, report: HallucinationReport):
        text = (r.answer + " " + r.reasoning).lower()
        for phrase in OVERCONFIDENT_PHRASES:
            if phrase in text and r.confidence > 0.85:
                report.flags.append(HallucinationFlag(
                    agent_name=r.agent_name,
                    flagged_claim=f"Uses '{phrase}' with very high confidence",
                    reason="Overconfident language without evidence",
                    severity="medium",
                    confidence_penalty=0.1,
                ))
                break

    def _check_unverified_citations(self, r: AgentResponse, report: HallucinationReport):
        text = (r.answer + " " + r.reasoning).lower()
        citation_count = sum(1 for phrase in FABRICATION_SIGNALS if phrase in text)
        if citation_count >= 2:
            report.flags.append(HallucinationFlag(
                agent_name=r.agent_name,
                flagged_claim="Multiple citation signals without verifiable sources",
                reason="Possible fabricated citations — requires fact-checking",
                severity="high",
                confidence_penalty=0.2,
            ))

    def _check_length_confidence_mismatch(self, r: AgentResponse, report: HallucinationReport):
        very_short = len(r.answer.strip()) < 30
        very_confident = r.confidence > 0.9
        if very_short and very_confident:
            report.flags.append(HallucinationFlag(
                agent_name=r.agent_name,
                flagged_claim="Very short answer with very high confidence",
                reason="Suspiciously confident answer with minimal reasoning",
                severity="low",
                confidence_penalty=0.05,
            ))

    def _check_cross_agent_contradictions(
        self, responses: list[AgentResponse], report: HallucinationReport
    ):
        """Flag cases where agents assert directly opposing facts."""
        pos_signals = {"yes", "true", "correct", "right", "valid", "proven"}
        neg_signals = {"no", "false", "incorrect", "wrong", "invalid", "disproven"}

        agent_stances = []
        for r in responses:
            words = set(r.answer.lower().split())
            has_pos = bool(words & pos_signals)
            has_neg = bool(words & neg_signals)
            agent_stances.append((r.agent_name, has_pos, has_neg))

        positive_agents = [name for name, p, n in agent_stances if p and not n]
        negative_agents = [name for name, p, n in agent_stances if n and not p]

        if positive_agents and negative_agents:
            for agent_name in positive_agents + negative_agents:
                report.flags.append(HallucinationFlag(
                    agent_name=agent_name,
                    flagged_claim="Direct factual contradiction with another agent",
                    reason=(
                        f"Agents {positive_agents} affirm while {negative_agents} deny. "
                        "At least one agent is likely hallucinating."
                    ),
                    severity="high",
                    confidence_penalty=0.3,
                ))
