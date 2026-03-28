"""
Anti-Herding Protocol — Blind First Round
Research shows naive debate degenerates into majority dynamics where
the first plausible answer gains momentum and others follow.
Solution: agents generate initial responses WITHOUT seeing each other's output.
Only from round 2 onward do they see each other's responses.
"""

from dataclasses import dataclass
from typing import Optional
from agents.base_agent import AgentResponse


@dataclass
class BlindRoundResult:
    round_number: int
    blind: bool
    responses: list[AgentResponse]
    diversity_score: float      # 0 = all same, 1 = maximally diverse

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "blind": self.blind,
            "diversity_score": round(self.diversity_score, 3),
            "responses": [r.to_dict() for r in self.responses],
        }


class BlindFirstRound:
    """
    Implements independent-then-debate protocol:
    Round 1: All agents generate answers blindly (no cross-agent context)
    Round 2+: Agents see prior round's responses and can critique/refine
    """

    def run_blind_round(
        self, query: str, agents: list, round_number: int,
        prior_context: Optional[str] = None
    ) -> BlindRoundResult:
        is_blind = round_number == 1

        responses: list[AgentResponse] = []
        for agent in agents:
            # In round 1: no context (blind). In subsequent rounds: provide context.
            context = None if is_blind else prior_context
            response = agent.generate(query, context=context)
            responses.append(response)

        diversity = self._calculate_diversity(responses)

        return BlindRoundResult(
            round_number=round_number,
            blind=is_blind,
            responses=responses,
            diversity_score=diversity,
        )

    def _calculate_diversity(self, responses: list[AgentResponse]) -> float:
        """
        Estimate diversity: high variance in confidence + different answer lengths
        = more diverse (agents genuinely disagree).
        """
        if len(responses) < 2:
            return 0.0

        confidences = [r.confidence for r in responses]
        lengths = [len(r.answer) for r in responses]

        conf_mean = sum(confidences) / len(confidences)
        conf_var = sum((c - conf_mean) ** 2 for c in confidences) / len(confidences)

        len_mean = sum(lengths) / len(lengths)
        len_var = sum((l - len_mean) ** 2 for l in lengths) / len(lengths)
        len_var_norm = min(1.0, len_var / 10000)

        diversity = min(1.0, (conf_var * 4) * 0.4 + len_var_norm * 0.6)
        return round(diversity, 3)

    def build_context_from_round(self, result: BlindRoundResult) -> str:
        """Format this round's responses as context for the next round."""
        parts = [f"Previous round responses (use these to refine your answer):"]
        for r in result.responses:
            parts.append(f"\n[{r.agent_name}]\nAnswer: {r.answer}\nReasoning: {r.reasoning}")
        return "\n".join(parts)
