"""
Base Agent — Abstract class for all Cognitive Conflict AI agents.
Includes robust LLM call with retry, timeout, and multi-provider support.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    agent_name: str
    role: str
    answer: str
    reasoning: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    tool_calls: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "tool_calls": self.tool_calls,
        }


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, model_config: dict):
        self.name = name
        self.role = role
        self.model_config = model_config
        self._system_prompt = self._build_system_prompt()

    @abstractmethod
    def _build_system_prompt(self) -> str:
        pass

    @abstractmethod
    def generate(self, query: str, context: Optional[str] = None) -> AgentResponse:
        pass

    def _call_llm(self, prompt: str) -> str:
        """
        Call configured LLM with retry logic and graceful fallback.
        Supports: ollama, openai, anthropic, mock (for testing).
        """
        provider = self.model_config.get("provider", "ollama")
        max_retries = self.model_config.get("max_retries", 3)
        retry_delay = self.model_config.get("retry_delay", 1.0)
        timeout = self.model_config.get("timeout", 60)

        for attempt in range(max_retries):
            try:
                if provider == "ollama":
                    return self._call_ollama(prompt, timeout)
                elif provider == "openai":
                    return self._call_openai(prompt, timeout)
                elif provider == "anthropic":
                    return self._call_anthropic(prompt, timeout)
                elif provider == "mock":
                    return self._call_mock(prompt)
                else:
                    raise ValueError(f"Unknown provider: {provider!r}")
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, max_retries, e, retry_delay,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error("LLM call failed after %d attempts: %s", max_retries, e)
                    return self._fallback_response(prompt, str(e))

        return self._fallback_response(prompt, "max retries exceeded")

    def _call_ollama(self, prompt: str, timeout: int) -> str:
        import urllib.request, json
        url = self.model_config.get("ollama_base_url", "http://localhost:11434") + "/api/generate"
        payload = json.dumps({
            "model": self.model_config.get("name", "mistral"),
            "prompt": f"{self._system_prompt}\n\n{prompt}",
            "stream": False,
            "options": {"temperature": self.model_config.get("temperature", 0.7)},
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("response", "")

    def _call_openai(self, prompt: str, timeout: int) -> str:
        import urllib.request, json
        api_key = self.model_config.get("openai_api_key", "")
        if not api_key:
            raise ValueError("openai_api_key not set in config")
        payload = json.dumps({
            "model": self.model_config.get("name", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.model_config.get("temperature", 0.7),
            "max_tokens": self.model_config.get("max_tokens", 1024),
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, prompt: str, timeout: int) -> str:
        import urllib.request, json
        api_key = self.model_config.get("anthropic_api_key", "")
        if not api_key:
            raise ValueError("anthropic_api_key not set in config")
        payload = json.dumps({
            "model": self.model_config.get("name", "claude-haiku-4-5-20251001"),
            "max_tokens": self.model_config.get("max_tokens", 1024),
            "system": self._system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]

    def _call_mock(self, prompt: str) -> str:
        """Deterministic mock for testing — no LLM needed."""
        return (
            f"ANSWER: Mock answer for: {prompt[:60]}\n"
            f"REASONING: Mock reasoning from {self.name}\n"
            f"CONFIDENCE: 0.75"
        )

    def _fallback_response(self, prompt: str, error: str) -> str:
        """Return a structured fallback when all LLM calls fail."""
        return (
            f"ANSWER: Unable to generate answer (LLM unavailable: {error[:80]})\n"
            f"REASONING: LLM call failed — check your provider config and model availability\n"
            f"CONFIDENCE: 0.0"
        )

    def _parse_structured(self, raw: str, default_confidence: float = 0.7) -> tuple[str, str, float]:
        """Parse ANSWER/REASONING/CONFIDENCE format from LLM output."""
        answer, reasoning, confidence = "", "", default_confidence
        for line in raw.strip().splitlines():
            if line.startswith("ANSWER:"):
                answer = line[7:].strip()
            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = max(0.0, min(1.0, float(line[11:].strip())))
                except ValueError:
                    pass
        if not answer:
            answer = raw.strip()[:500]
        return answer, reasoning, confidence

    def __repr__(self) -> str:
        return f"<Agent name={self.name!r} role={self.role!r}>"
