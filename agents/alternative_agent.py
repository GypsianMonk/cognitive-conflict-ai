from __future__ import annotations
from typing import Optional
from agents.base_agent import BaseAgent, AgentResponse


class AlternativeAgent(BaseAgent):
    def __init__(self, model_config: dict):
        super().__init__("Agent-C (Alternative)", "Lateral Thinker", model_config)

    def _build_system_prompt(self) -> str:
        return (
            "You are a creative, lateral-thinking assistant. Provide NON-OBVIOUS perspectives. "
            "Think analogies, alternative framings, second-order effects, unconventional solutions. "
            "Avoid repeating what others have likely said.\n"
            "Format:\nANSWER: <alternative answer>\nREASONING: <why this view is distinct>\nCONFIDENCE: <0.0-1.0>"
        )

    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        prompt = (f"Previous perspectives (think differently):\n{context}\n\n" if context else "") + f"Query: {query}"
        raw = self._call_llm(prompt)
        answer, reasoning, confidence = self._parse_structured(raw, 0.7)
        return AgentResponse(self.name, self.role, answer, reasoning, confidence)


class ExpertAgent(BaseAgent):
    def __init__(self, domain: str, model_config: dict):
        self.domain = domain
        super().__init__(f"Expert ({domain})", f"{domain} Domain Expert", model_config)

    def _build_system_prompt(self) -> str:
        return (
            f"You are a world-class expert in {self.domain}. Answer with precision and authority. "
            f"Reference domain-specific best practices. Correct common misconceptions about {self.domain}.\n"
            "Format:\nANSWER: <expert answer>\nREASONING: <domain reasoning>\nCONFIDENCE: <0.0-1.0>"
        )

    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        prompt = (f"Context:\n{context}\n\n" if context else "") + f"Query: {query}"
        raw = self._call_llm(prompt)
        answer, reasoning, confidence = self._parse_structured(raw, 0.9)
        return AgentResponse(self.name, self.role, answer, reasoning, confidence)
