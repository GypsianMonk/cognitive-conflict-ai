"""
Tests for the Scorer module.
"""

import pytest
from agents.base_agent import AgentResponse
from scoring.scorer import Scorer, ScoredResponse


SCORING_CONFIG = {
    "relevance_weight": 0.4,
    "coherence_weight": 0.4,
    "contradiction_penalty": 0.2,
}


def make_response(answer: str, reasoning: str, confidence: float, name: str = "Test Agent") -> AgentResponse:
    return AgentResponse(
        agent_name=name,
        role="Test",
        answer=answer,
        reasoning=reasoning,
        confidence=confidence,
    )


class TestScorer:
    def setup_method(self):
        self.scorer = Scorer(SCORING_CONFIG)

    def test_score_single_response(self):
        r = make_response("This is a solid answer.", "Here is why.", 0.85)
        scored = self.scorer.score_all([r])
        assert len(scored) == 1
        assert isinstance(scored[0], ScoredResponse)
        assert 0.0 <= scored[0].total_score <= 1.0

    def test_higher_confidence_scores_higher(self):
        low = make_response("Answer", "Reasoning", 0.3, "Low")
        high = make_response("Answer", "Reasoning", 0.9, "High")
        scored = self.scorer.score_all([low, high])
        high_scored = next(s for s in scored if s.agent_name == "High")
        low_scored = next(s for s in scored if s.agent_name == "Low")
        assert high_scored.total_score >= low_scored.total_score

    def test_empty_answer_penalized(self):
        empty = make_response("", "", 0.8)
        full = make_response("A detailed and informative answer here.", "Clear reasoning.", 0.8)
        scored_empty = self.scorer.score_all([empty])[0]
        scored_full = self.scorer.score_all([full])[0]
        assert scored_full.total_score > scored_empty.total_score

    def test_sorted_descending(self):
        responses = [
            make_response("Short.", ".", 0.4, "C"),
            make_response("Longer answer with more detail.", "Good reasoning.", 0.9, "A"),
            make_response("Medium length answer here.", "Some reasoning.", 0.7, "B"),
        ]
        scored = self.scorer.score_all(responses)
        scores = [s.total_score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_to_dict_structure(self):
        r = make_response("Answer", "Reasoning", 0.75)
        scored = self.scorer.score_all([r])[0]
        d = scored.to_dict()
        assert "agent_name" in d
        assert "total_score" in d
        assert "relevance_score" in d
        assert "coherence_score" in d
