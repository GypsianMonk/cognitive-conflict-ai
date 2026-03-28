from __future__ import annotations
from typing import Optional
from agents.base_agent import BaseAgent, AgentResponse


class SkepticAgent(BaseAgent):
    def __init__(self, model_config: dict):
        super().__init__("Agent-B (Skeptic)", "Critical Adversarial Challenger", model_config)

    def _build_system_prompt(self) -> str:
        return (
            "You are a rigorous, skeptical assistant. Challenge assumptions. "
            "Find hidden biases, edge cases, logical fallacies, missing context, failure modes. "
            "Provide grounded critique — don't just say things are wrong, explain why.\n"
            "Format:\nANSWER: <critical perspective>\nREASONING: <specific weaknesses found>\nCONFIDENCE: <0.0-1.0>"
        )

    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        prompt = (f"Prior answer to critique:\n{context}\n\n" if context else "") + f"Query: {query}"
        raw = self._call_llm(prompt)
        answer, reasoning, confidence = self._parse_structured(raw, 0.75)
        return AgentResponse(self.name, self.role, answer, reasoning, confidence)
