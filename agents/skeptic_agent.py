"""
Skeptic Agent — Critically challenges assumptions and identifies weaknesses.
"""

from typing import Optional
from agents.base_agent import BaseAgent, AgentResponse


class SkepticAgent(BaseAgent):
    def __init__(self, model_config: dict):
        super().__init__(
            name="Agent-B (Skeptic)",
            role="Critical and Adversarial Challenger",
            model_config=model_config,
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are a rigorous, skeptical AI assistant. "
            "Your role is to critically examine any question or claim and challenge assumptions. "
            "Look for: hidden biases, edge cases, logical fallacies, missing context, and potential failure modes. "
            "Do NOT simply say things are wrong — provide grounded, reasoned critique. "
            "It's okay to agree with parts of an argument while challenging others. "
            "Structure your response as:\n"
            "ANSWER: <your critical perspective>\n"
            "REASONING: <the specific weaknesses or risks you identified>\n"
            "CONFIDENCE: <your confidence from 0.0 to 1.0>"
        )

    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        prompt = f"Query: {query}"
        if context:
            prompt = f"Prior answer to critique:\n{context}\n\nOriginal query: {query}"

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
        answer, reasoning, confidence = "", "", 0.75

        for line in lines:
            if line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.75

        if not answer:
            answer = raw.strip()

        return answer, reasoning, confidence
