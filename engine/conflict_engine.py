"""
Conflict Engine — The heart of Cognitive Conflict AI.
Orchestrates multi-round debates between agents and detects contradictions.
"""

from dataclasses import dataclass, field
from typing import Optional
import time

from agents.base_agent import AgentResponse


@dataclass
class DebateRound:
    round_number: int
    responses: list[AgentResponse]
    contradictions: list[str]
    convergence_score: float   # 0.0 = total disagreement, 1.0 = full agreement


@dataclass
class DebateResult:
    query: str
    rounds: list[DebateRound]
    total_rounds: int
    final_responses: list[AgentResponse]
    contradiction_summary: list[str]
    convergence_achieved: bool
    duration_seconds: float


class ConflictEngine:
    """
    Manages the structured debate loop between agents.
    Detects contradictions, scores convergence, and decides when to stop.
    """

    CONVERGENCE_THRESHOLD = 0.75

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds

    def run_debate(
        self,
        query: str,
        agents: list,
        retrieval_context: Optional[str] = None,
    ) -> DebateResult:
        start = time.time()
        rounds: list[DebateRound] = []
        context = retrieval_context

        for round_num in range(1, self.max_rounds + 1):
            responses = self._run_round(query, agents, round_num, context)
            contradictions = self._detect_contradictions(responses)
            convergence = self._calculate_convergence(responses)

            debate_round = DebateRound(
                round_number=round_num,
                responses=responses,
                contradictions=contradictions,
                convergence_score=convergence,
            )
            rounds.append(debate_round)

            # Build context for next round from this round's answers
            context = self._build_round_context(responses)

            if convergence >= self.CONVERGENCE_THRESHOLD:
                break  # Agents have converged — no need for more rounds

        final_responses = rounds[-1].responses
        all_contradictions = [c for r in rounds for c in r.contradictions]
        convergence_achieved = rounds[-1].convergence_score >= self.CONVERGENCE_THRESHOLD

        return DebateResult(
            query=query,
            rounds=rounds,
            total_rounds=len(rounds),
            final_responses=final_responses,
            contradiction_summary=list(set(all_contradictions)),
            convergence_achieved=convergence_achieved,
            duration_seconds=time.time() - start,
        )

    def _run_round(
        self,
        query: str,
        agents: list,
        round_num: int,
        context: Optional[str],
    ) -> list[AgentResponse]:
        """Generate responses from all agents for this round."""
        responses = []
        for agent in agents:
            # In round 1, no cross-agent context; from round 2+, agents see prior round
            agent_context = context if round_num > 1 else None
            response = agent.generate(query, context=agent_context)
            responses.append(response)
        return responses

    def _detect_contradictions(self, responses: list[AgentResponse]) -> list[str]:
        """
        Simple heuristic contradiction detection.
        In production, replace with an LLM-based contradiction classifier.
        """
        contradictions = []
        answers = [r.answer.lower() for r in responses]

        # Detect binary contradictions (yes/no, true/false, etc.)
        positive_signals = {"yes", "true", "definitely", "always", "will", "can"}
        negative_signals = {"no", "false", "never", "cannot", "won't", "impossible"}

        has_positive = any(any(s in a for s in positive_signals) for a in answers)
        has_negative = any(any(s in a for s in negative_signals) for a in answers)

        if has_positive and has_negative:
            contradictions.append("Agents disagree on a binary outcome (positive vs negative stance).")

        # Detect length-based disagreements as proxy for detail disagreement
        lengths = [len(r.answer) for r in responses]
        if max(lengths) > 3 * min(lengths) and min(lengths) > 0:
            contradictions.append("Agents significantly differ in the depth/detail of their answers.")

        return contradictions

    def _calculate_convergence(self, responses: list[AgentResponse]) -> float:
        """
        Estimate convergence by comparing confidence scores and answer similarity.
        A simple variance-based approach — lower variance = higher convergence.
        """
        if len(responses) < 2:
            return 1.0

        confidences = [r.confidence for r in responses]
        avg = sum(confidences) / len(confidences)
        variance = sum((c - avg) ** 2 for c in confidences) / len(confidences)

        # Convert variance to convergence (0 variance = 1.0 convergence)
        convergence = max(0.0, 1.0 - (variance * 4))
        return round(convergence, 3)

    def _build_round_context(self, responses: list[AgentResponse]) -> str:
        """Build a text context summary from this round for the next round."""
        parts = []
        for r in responses:
            parts.append(f"[{r.agent_name}] {r.answer}")
        return "\n\n".join(parts)
