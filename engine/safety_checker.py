"""
Prompt Injection Guard & Safety Checker
Monitors inputs and reasoning traces for injection attacks.
Based on Constitutional AI principles — a safety checker runs on all outputs.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SafetyResult:
    safe: bool
    issues: list[str] = field(default_factory=list)
    sanitized_input: str = ""
    risk_level: str = "none"    # none | low | medium | high | critical

    def to_dict(self) -> dict:
        return {
            "safe": self.safe,
            "issues": self.issues,
            "risk_level": self.risk_level,
        }


INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore your system prompt",
    "disregard all prior",
    "you are now",
    "forget everything",
    "new instructions:",
    "override your",
    "your true self",
    "jailbreak",
    "pretend you are",
    "act as if",
    "do anything now",
    "dan mode",
    "developer mode",
]

HARMFUL_PATTERNS = [
    "how to make a bomb",
    "how to synthesize",
    "step by step to hack",
    "exploit vulnerability",
    "generate malware",
    "create ransomware",
]

PII_PATTERNS = [
    "social security",
    "credit card number",
    "bank account",
    "password is",
    "my pin is",
]


class SafetyChecker:
    """
    Multi-layer safety checker:
    1. Input sanitization (injection detection)
    2. Harmful content detection
    3. PII leakage detection
    4. Reasoning trace monitoring (post-generation)
    """

    def check_input(self, text: str) -> SafetyResult:
        issues, risk = [], "none"
        lower = text.lower()

        for pattern in INJECTION_PATTERNS:
            if pattern in lower:
                issues.append(f"Injection attempt detected: '{pattern}'")
                risk = "critical"

        for pattern in HARMFUL_PATTERNS:
            if pattern in lower:
                issues.append(f"Harmful content request: '{pattern}'")
                risk = "high" if risk != "critical" else risk

        for pattern in PII_PATTERNS:
            if pattern in lower:
                issues.append(f"PII pattern detected: '{pattern}'")
                risk = "medium" if risk not in ("high", "critical") else risk

        sanitized = self._sanitize(text)
        return SafetyResult(
            safe=not issues,
            issues=issues,
            sanitized_input=sanitized,
            risk_level=risk,
        )

    def check_output(self, text: str, agent_name: str = "") -> SafetyResult:
        """Check agent output before it's scored or passed forward."""
        issues, risk = [], "none"
        lower = text.lower()

        # Detect if the agent was hijacked to produce harmful content
        for pattern in HARMFUL_PATTERNS:
            if pattern in lower:
                issues.append(
                    f"Agent {agent_name!r} produced harmful content: '{pattern}'"
                )
                risk = "high"

        # Detect if agent is leaking its system prompt (jailbreak indicator)
        system_leak_signals = ["my system prompt", "my instructions are", "i was told to"]
        for signal in system_leak_signals:
            if signal in lower:
                issues.append(f"Possible system prompt leak from {agent_name!r}")
                risk = "medium" if risk != "high" else risk

        return SafetyResult(
            safe=not issues,
            issues=issues,
            sanitized_input=text,
            risk_level=risk,
        )

    def check_reasoning_trace(self, trace: str) -> SafetyResult:
        """Monitor reasoning for signs of manipulation or injection mid-trace."""
        issues = []
        for pattern in INJECTION_PATTERNS:
            if pattern in trace.lower():
                issues.append(f"Injection pattern in reasoning trace: '{pattern}'")

        return SafetyResult(
            safe=not issues,
            issues=issues,
            sanitized_input=trace,
            risk_level="high" if issues else "none",
        )

    def _sanitize(self, text: str) -> str:
        """
        Basic sanitization: strip null bytes and control characters.
        In production: add more aggressive sanitization per your threat model.
        """
        import re
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return sanitized.strip()
