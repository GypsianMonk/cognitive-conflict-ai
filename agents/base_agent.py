"""
Base Agent — Abstract class for all Cognitive Conflict AI agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class AgentResponse:
    agent_name: str
    role: str
    answer: str
    reasoning: str
    confidence: float          # 0.0 – 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Cognitive Conflict system.
    Each agent has a unique role and produces a structured response.
    """

    def __init__(self, name: str, role: str, model_config: dict):
        self.name = name
        self.role = role
        self.model_config = model_config
        self._system_prompt = self._build_system_prompt()

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Build the role-specific system prompt."""
        pass

    @abstractmethod
    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        """Generate a response for the given query."""
        pass

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM backend.
        Override this to support different providers (Ollama, OpenAI, Anthropic).
        """
        import requests

        provider = self.model_config.get("provider", "ollama")

        if provider == "ollama":
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_config.get("name", "mistral"),
                    "prompt": f"{self._system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": self.model_config.get("temperature", 0.7),
                    },
                },
                timeout=60,
            )
            return response.json().get("response", "")

        raise NotImplementedError(f"Provider '{provider}' not yet implemented.")

    def __repr__(self):
        return f"<Agent name={self.name!r} role={self.role!r}>"
