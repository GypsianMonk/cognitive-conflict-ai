"""
Scorer — Evaluates and ranks agent responses using weighted metrics.
"""

from dataclasses import dataclass
from agents.base_agent import AgentResponse


@dataclass
class ScoredResponse:
    agent_name: str
    role: str
    answer: str
    reasoning: str
    relevance_score: float
    coherence_score: float
    contradiction_score: float   # Lower is better (0 = no contradiction)
    total_score: float

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "answer": self.answer,
            "relevance_score": round(self.relevance_score, 3),
            "coherence_score": round(self.coherence_score, 3),
            "contradiction_score": round(self.contradiction_score, 3),
            "total_score": round(self.total_score, 3),
        }


class Scorer:
    """
    Scores agent responses on multiple quality dimensions.

    Formula:
        total = (relevance × w1) + (coherence × w2) - (contradiction × w3)

    Default weights:
        relevance    = 0.4
        coherence    = 0.4
        contradiction = 0.2
    """

    DEFAULT_WEIGHTS = {
        "relevance": 0.4,
        "coherence": 0.4,
        "contradiction": 0.2,
    }

    def __init__(self, scoring_config: dict):
        self.weights = {
            "relevance": scoring_config.get("relevance_weight", self.DEFAULT_WEIGHTS["relevance"]),
            "coherence": scoring_config.get("coherence_weight", self.DEFAULT_WEIGHTS["coherence"]),
            "contradiction": scoring_config.get("contradiction_penalty", self.DEFAULT_WEIGHTS["contradiction"]),
        }

    def score_all(self, responses: list[AgentResponse]) -> list[ScoredResponse]:
        """Score all responses and return sorted by total score (descending)."""
        scored = [self._score_one(r) for r in responses]
        return sorted(scored, key=lambda x: x.total_score, reverse=True)

    def _score_one(self, response: AgentResponse) -> ScoredResponse:
        relevance = self._score_relevance(response)
        coherence = self._score_coherence(response)
        contradiction = self._score_contradiction(response)

        total = (
            relevance * self.weights["relevance"]
            + coherence * self.weights["coherence"]
            - contradiction * self.weights["contradiction"]
        )
        total = round(max(0.0, min(1.0, total)), 4)

        return ScoredResponse(
            agent_name=response.agent_name,
            role=response.role,
            answer=response.answer,
            reasoning=response.reasoning,
            relevance_score=relevance,
            coherence_score=coherence,
            contradiction_score=contradiction,
            total_score=total,
        )

    def _score_relevance(self, response: AgentResponse) -> float:
        """
        Heuristic relevance: use agent confidence + answer length as proxy.
        In production: embed query + answer and compute cosine similarity.
        """
        length_score = min(1.0, len(response.answer) / 300)
        return round((response.confidence * 0.6) + (length_score * 0.4), 4)

    def _score_coherence(self, response: AgentResponse) -> float:
        """
        Heuristic coherence: check if answer and reasoning are both non-empty.
        In production: use a coherence classifier or LLM judge.
        """
        has_answer = 1.0 if response.answer.strip() else 0.0
        has_reasoning = 0.5 if response.reasoning.strip() else 0.0
        return round(has_answer * 0.5 + has_reasoning, 4)

    def _score_contradiction(self, response: AgentResponse) -> float:
        """
        Heuristic contradiction: look for self-contradicting hedge words.
        Lower score = less contradiction = better.
        """
        contradictory_phrases = [
            "however", "but also", "on the other hand", "yet simultaneously",
            "contradicts", "conflicts with",
        ]
        text = (response.answer + " " + response.reasoning).lower()
        count = sum(1 for phrase in contradictory_phrases if phrase in text)
        return round(min(1.0, count * 0.2), 4)
