from __future__ import annotations
from typing import Optional
from agents.base_agent import BaseAgent, AgentResponse


class OptimistAgent(BaseAgent):
    def __init__(self, model_config: dict):
        super().__init__("Agent-A (Optimist)", "Direct and Optimistic Answerer", model_config)

    def _build_system_prompt(self) -> str:
        return (
            "You are a direct, confident, solution-oriented assistant. "
            "Provide the clearest, most helpful answer. Focus on what CAN be done. "
            "Be decisive. Conclude with a clear recommendation.\n"
            "Format:\nANSWER: <answer>\nREASONING: <reasoning>\nCONFIDENCE: <0.0-1.0>"
        )

    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        prompt = (f"Context:\n{context}\n\n" if context else "") + f"Query: {query}"
        raw = self._call_llm(prompt)
        answer, reasoning, confidence = self._parse_structured(raw, 0.8)
        return AgentResponse(self.name, self.role, answer, reasoning, confidence)
