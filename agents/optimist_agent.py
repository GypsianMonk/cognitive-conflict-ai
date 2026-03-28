"""
Optimist Agent — Generates direct, confident, solution-oriented answers.
"""

from typing import Optional
from agents.base_agent import BaseAgent, AgentResponse


class OptimistAgent(BaseAgent):
    def __init__(self, model_config: dict):
        super().__init__(
            name="Agent-A (Optimist)",
            role="Direct and Optimistic Answerer",
            model_config=model_config,
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are a direct, confident, and solution-oriented AI assistant. "
            "Your role is to provide the most straightforward, helpful, and optimistic answer possible. "
            "Focus on what CAN be done, what IS likely true, and what the BEST-CASE interpretation is. "
            "Be decisive. Avoid excessive hedging. Always conclude with a clear recommendation or answer. "
            "Structure your response as:\n"
            "ANSWER: <your direct answer>\n"
            "REASONING: <why you believe this>\n"
            "CONFIDENCE: <your confidence from 0.0 to 1.0>"
        )

    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        prompt = f"Query: {query}"
        if context:
            prompt = f"Context:\n{context}\n\n{prompt}"

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
        answer, reasoning, confidence = "", "", 0.8

        for line in lines:
            if line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.8

        if not answer:
            answer = raw.strip()

        return answer, reasoning, confidence
