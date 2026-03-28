"""
Alternative Agent — Provides lateral thinking and non-obvious perspectives.
Expert Agent   — Domain-injected specialist dynamically added per query.
"""

from typing import Optional
from agents.base_agent import BaseAgent, AgentResponse


class AlternativeAgent(BaseAgent):
    def __init__(self, model_config: dict):
        super().__init__(
            name="Agent-C (Alternative)",
            role="Lateral Thinker and Alternative Perspective",
            model_config=model_config,
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are a creative, lateral-thinking AI assistant. "
            "Your role is to provide NON-OBVIOUS perspectives — approaches that other analysts might miss. "
            "Think about analogies, alternative framings, second-order effects, and unconventional solutions. "
            "Avoid repeating what others have likely already said. Surprise with insight. "
            "Structure your response as:\n"
            "ANSWER: <your alternative or creative answer>\n"
            "REASONING: <why this perspective is valuable and different>\n"
            "CONFIDENCE: <your confidence from 0.0 to 1.0>"
        )

    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        prompt = f"Query: {query}"
        if context:
            prompt = f"Previous perspectives (think differently from these):\n{context}\n\nQuery: {query}"

        raw = self._call_llm(prompt)
        answer, reasoning, confidence = self._parse_response(raw)

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            answer=answer,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _parse_response(self, raw: str) -> tuple[str, str, float]:
        lines = raw.strip().splitlines()
        answer, reasoning, confidence = "", "", 0.7

        for line in lines:
            if line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.7

        if not answer:
            answer = raw.strip()

        return answer, reasoning, confidence


class ExpertAgent(BaseAgent):
    """
    Dynamically injected domain expert.
    The domain is specified at runtime based on query classification.
    """

    def __init__(self, domain: str, model_config: dict):
        self.domain = domain
        super().__init__(
            name=f"Expert Agent ({domain})",
            role=f"{domain} Domain Expert",
            model_config=model_config,
        )

    def _build_system_prompt(self) -> str:
        return (
            f"You are a world-class expert in {self.domain}. "
            f"You have deep domain knowledge and answer with precision and authority. "
            f"Reference domain-specific best practices, standards, and nuances. "
            f"Correct any common misconceptions related to {self.domain}. "
            "Structure your response as:\n"
            "ANSWER: <your expert answer>\n"
            "REASONING: <domain-specific reasoning and references>\n"
            "CONFIDENCE: <your confidence from 0.0 to 1.0>"
        )

    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        prompt = f"Query: {query}"
        if context:
            prompt = f"Context:\n{context}\n\nQuery: {query}"

        raw = self._call_llm(prompt)
        answer, reasoning, confidence = self._parse_response(raw)

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            answer=answer,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _parse_response(self, raw: str) -> tuple[str, str, float]:
        lines = raw.strip().splitlines()
        answer, reasoning, confidence = "", "", 0.9

        for line in lines:
            if line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.9

        if not answer:
            answer = raw.strip()

        return answer, reasoning, confidence
