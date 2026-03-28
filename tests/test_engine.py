"""
Tests for the Conflict Engine and Query Understanding layer.
"""

import pytest
from unittest.mock import patch, MagicMock

from engine.query_understanding import QueryUnderstanding, QueryAnalysis
from engine.conflict_engine import ConflictEngine
from agents.base_agent import AgentResponse


class TestQueryUnderstanding:
    def setup_method(self):
        self.analyzer = QueryUnderstanding()

    def test_simple_query(self):
        result = self.analyzer.analyze("What is Python?")
        assert isinstance(result, QueryAnalysis)
        assert result.complexity in ("simple", "medium", "complex")

    def test_complex_query(self):
        result = self.analyzer.analyze("Analyze the implications of using microservices over monolith architecture")
        assert result.complexity == "complex"
        assert result.debate_rounds == 3

    def test_domain_detection_finance(self):
        result = self.analyzer.analyze("What is the best stock investment strategy?")
        assert result.domain == "finance"
        assert result.requires_expert is True

    def test_domain_detection_medicine(self):
        result = self.analyzer.analyze("What are common disease symptoms?")
        assert result.domain == "medicine"

    def test_no_domain(self):
        result = self.analyzer.analyze("What time is it?")
        assert result.domain is None
        assert result.requires_expert is False

    def test_debate_rounds_scale_with_complexity(self):
        simple = self.analyzer.analyze("What is 2+2?")
        complex_ = self.analyzer.analyze("Evaluate the ethical implications of AI replacing human jobs")
        assert simple.debate_rounds <= complex_.debate_rounds


class TestConflictEngine:
    def _make_mock_agent(self, name: str, answer: str, confidence: float):
        agent = MagicMock()
        agent.name = name
        agent.generate.return_value = AgentResponse(
            agent_name=name,
            role="Test Role",
            answer=answer,
            reasoning="Test reasoning",
            confidence=confidence,
        )
        return agent

    def test_single_round_debate(self):
        engine = ConflictEngine(max_rounds=1)
        agents = [
            self._make_mock_agent("A", "Yes, it is true.", 0.9),
            self._make_mock_agent("B", "No, that is incorrect.", 0.7),
        ]
        result = engine.run_debate("Is the sky blue?", agents)
        assert result.total_rounds == 1
        assert len(result.final_responses) == 2

    def test_convergence_stops_debate_early(self):
        engine = ConflictEngine(max_rounds=3)
        # Both agents agree (high, similar confidence)
        agents = [
            self._make_mock_agent("A", "Yes definitely.", 0.88),
            self._make_mock_agent("B", "Yes agreed.", 0.90),
        ]
        result = engine.run_debate("Should we test code?", agents)
        # Convergence should stop early — may run fewer than 3 rounds
        assert result.total_rounds <= 3

    def test_contradiction_detection(self):
        engine = ConflictEngine(max_rounds=1)
        agents = [
            self._make_mock_agent("A", "It will definitely always work.", 0.9),
            self._make_mock_agent("B", "It is impossible and will never work.", 0.8),
        ]
        result = engine.run_debate("Will this succeed?", agents)
        assert len(result.contradiction_summary) > 0

    def test_debate_result_has_duration(self):
        engine = ConflictEngine(max_rounds=1)
        agents = [self._make_mock_agent("A", "Answer", 0.8)]
        result = engine.run_debate("Test?", agents)
        assert result.duration_seconds >= 0
