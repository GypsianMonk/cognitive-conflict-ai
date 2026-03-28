"""
Uncertainty Decomposition
Splits uncertainty into:
  - EPISTEMIC: lack of knowledge (can be reduced with more data/retrieval)
  - ALEATORIC: inherent randomness (irreducible — must flag to user)
Based on arXiv 2025 AAMAS paper on uncertainty-aware multi-agent systems.
"""

from dataclasses import dataclass
from agents.base_agent import AgentResponse


@dataclass
class UncertaintyBreakdown:
    total_uncertainty: float       # 0.0 (certain) to 1.0 (fully uncertain)
    epistemic: float               # Knowledge gap — can be fixed with retrieval
    aleatoric: float               # Inherent randomness — cannot be fixed
    dominant_type: str             # 'epistemic' | 'aleatoric' | 'low'
    recommendation: str            # What to do about it
    agent_agreement: float         # Agreement across agents (0-1)

    def to_dict(self) -> dict:
        return {
            "total_uncertainty": round(self.total_uncertainty, 3),
            "epistemic": round(self.epistemic, 3),
            "aleatoric": round(self.aleatoric, 3),
            "dominant_type": self.dominant_type,
            "recommendation": self.recommendation,
            "agent_agreement": round(self.agent_agreement, 3),
        }


HEDGING_WORDS = [
    "might", "may", "could", "possibly", "perhaps", "uncertain",
    "unclear", "unknown", "not sure", "I think", "I believe",
    "it depends", "varies", "debated", "controversial",
]

KNOWLEDGE_GAP_WORDS = [
    "don't know", "no data", "no evidence", "not found",
    "insufficient", "missing", "unavailable", "cannot confirm",
]

INHERENT_RANDOMNESS_WORDS = [
    "random", "stochastic", "unpredictable", "chaotic",
    "probabilistic", "chance", "luck", "coincidence",
]


class UncertaintyDecomposer:
    """
    Analyzes agent responses to decompose uncertainty into epistemic and aleatoric components.
    """

    def decompose(self, responses: list[AgentResponse]) -> UncertaintyBreakdown:
        if not responses:
            return UncertaintyBreakdown(1.0, 0.5, 0.5, "epistemic", "No responses to analyze.", 0.0)

        epistemic = self._score_epistemic(responses)
        aleatoric = self._score_aleatoric(responses)
        agreement = self._score_agreement(responses)

        # Low agreement amplifies epistemic uncertainty
        epistemic = min(1.0, epistemic + (1.0 - agreement) * 0.3)
        total = min(1.0, (epistemic + aleatoric) / 2)

        dominant = self._dominant_type(epistemic, aleatoric, total)
        recommendation = self._recommend(dominant, epistemic, aleatoric)

        return UncertaintyBreakdown(
            total_uncertainty=round(total, 3),
            epistemic=round(epistemic, 3),
            aleatoric=round(aleatoric, 3),
            dominant_type=dominant,
            recommendation=recommendation,
            agent_agreement=round(agreement, 3),
        )

    def _score_epistemic(self, responses: list[AgentResponse]) -> float:
        all_text = " ".join(r.answer + " " + r.reasoning for r in responses).lower()
        matches = sum(1 for w in HEDGING_WORDS + KNOWLEDGE_GAP_WORDS if w in all_text)
        return min(1.0, matches * 0.06)

    def _score_aleatoric(self, responses: list[AgentResponse]) -> float:
        all_text = " ".join(r.answer + " " + r.reasoning for r in responses).lower()
        matches = sum(1 for w in INHERENT_RANDOMNESS_WORDS if w in all_text)
        return min(1.0, matches * 0.12)

    def _score_agreement(self, responses: list[AgentResponse]) -> float:
        if len(responses) < 2:
            return 1.0
        confidences = [r.confidence for r in responses]
        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
        return max(0.0, 1.0 - variance * 4)

    def _dominant_type(self, epistemic: float, aleatoric: float, total: float) -> str:
        if total < 0.2:
            return "low"
        if epistemic >= aleatoric:
            return "epistemic"
        return "aleatoric"

    def _recommend(self, dominant: str, epistemic: float, aleatoric: float) -> str:
        if dominant == "low":
            return "Confidence is high. Proceed with the answer."
        if dominant == "epistemic":
            return (
                f"Knowledge gap detected (epistemic={epistemic:.2f}). "
                "Recommend: retrieve additional sources, query a domain expert, or request clarification."
            )
        return (
            f"Inherent randomness detected (aleatoric={aleatoric:.2f}). "
            "This uncertainty cannot be reduced with more data — flag this to the user explicitly."
        )
