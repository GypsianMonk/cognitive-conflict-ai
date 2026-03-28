"""
Tests for the agent layer.
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.optimist_agent import OptimistAgent
from agents.skeptic_agent import SkepticAgent
from agents.alternative_agent import AlternativeAgent, ExpertAgent
from agents.base_agent import AgentResponse


MODEL_CONFIG = {"provider": "ollama", "name": "mistral", "temperature": 0.7}

MOCK_RESPONSE = (
    "ANSWER: This is a test answer.\n"
    "REASONING: This is the reasoning.\n"
    "CONFIDENCE: 0.85"
)


@pytest.fixture
def mock_llm():
    with patch.object(OptimistAgent, "_call_llm", return_value=MOCK_RESPONSE):
        with patch.object(SkepticAgent, "_call_llm", return_value=MOCK_RESPONSE):
            with patch.object(AlternativeAgent, "_call_llm", return_value=MOCK_RESPONSE):
                with patch.object(ExpertAgent, "_call_llm", return_value=MOCK_RESPONSE):
                    yield


class TestOptimistAgent:
    def test_init(self):
        agent = OptimistAgent(MODEL_CONFIG)
        assert "Optimist" in agent.name
        assert agent.role is not None

    def test_generate_returns_agent_response(self, mock_llm):
        agent = OptimistAgent(MODEL_CONFIG)
        response = agent.generate("What is AI?")
        assert isinstance(response, AgentResponse)
        assert response.answer == "This is a test answer."
        assert response.confidence == 0.85

    def test_generate_with_context(self, mock_llm):
        agent = OptimistAgent(MODEL_CONFIG)
        response = agent.generate("What is AI?", context="Some context")
        assert isinstance(response, AgentResponse)


class TestSkepticAgent:
    def test_init(self):
        agent = SkepticAgent(MODEL_CONFIG)
        assert "Skeptic" in agent.name

    def test_generate(self, mock_llm):
        agent = SkepticAgent(MODEL_CONFIG)
        response = agent.generate("Is AI safe?")
        assert isinstance(response, AgentResponse)


class TestAlternativeAgent:
    def test_init(self):
        agent = AlternativeAgent(MODEL_CONFIG)
        assert "Alternative" in agent.name

    def test_generate(self, mock_llm):
        agent = AlternativeAgent(MODEL_CONFIG)
        response = agent.generate("How does memory work?")
        assert isinstance(response, AgentResponse)


class TestExpertAgent:
    def test_init_with_domain(self):
        agent = ExpertAgent(domain="finance", model_config=MODEL_CONFIG)
        assert "finance" in agent.name.lower()
        assert "finance" in agent._system_prompt.lower()

    def test_generate(self, mock_llm):
        agent = ExpertAgent(domain="medicine", model_config=MODEL_CONFIG)
        response = agent.generate("What is insulin?")
        assert isinstance(response, AgentResponse)


class TestAgentResponse:
    def test_to_dict(self):
        r = AgentResponse(
            agent_name="Test Agent",
            role="Test Role",
            answer="Answer text",
            reasoning="Reasoning text",
            confidence=0.9,
        )
        d = r.to_dict()
        assert d["agent_name"] == "Test Agent"
        assert d["confidence"] == 0.9
        assert "timestamp" in d
