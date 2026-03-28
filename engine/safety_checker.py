"""
Prompt Injection Guard & Safety Checker
Monitors inputs and outputs for injection attacks, harmful content,
and PII leakage. Based on Constitutional AI principles.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SafetyResult:
    safe: bool
    issues: list = field(default_factory=list)
    sanitized_input: str = ""
    risk_level: str = "none"    # none | low | medium | high | critical

    def to_dict(self) -> dict:
        return {"safe": self.safe, "issues": self.issues, "risk_level": self.risk_level}


# All patterns lowercased — compared against lower(input)
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "ignore your system prompt",
    "disregard all prior",
    "disregard your instructions",
    "you are now",
    "forget everything",
    "new instructions:",
    "override your",
    "your true self",
    "jailbreak",
    "pretend you are",
    "act as if you have no",
    "do anything now",
    "dan mode",
    "developer mode",
    "god mode",
]

HARMFUL_PATTERNS = [
    "how to make a bomb",
    "how to synthesize",
    "step by step to hack",
    "exploit vulnerability",
    "generate malware",
    "create ransomware",
    "make explosives",
]

PII_PATTERNS = [
    "social security",
    "credit card number",
    "bank account number",
    "my password is",
    "my pin is",
]

SYSTEM_LEAK_SIGNALS = [
    "my system prompt",
    "my instructions are",
    "i was told to",
    "my training says",
]


class SafetyChecker:
    def check_input(self, text: str) -> SafetyResult:
        if not text or not text.strip():
            # Empty queries are allowed through — handled by validation elsewhere
            return SafetyResult(safe=True, sanitized_input="",
                                risk_level="none")

        issues: list[str] = []
        risk = "none"
        lower = text.lower()

        for pattern in INJECTION_PATTERNS:
            if pattern in lower:
                issues.append(f"Injection attempt: '{pattern}'")
                risk = "critical"
                break   # One critical is enough

        if risk != "critical":
            for pattern in HARMFUL_PATTERNS:
                if pattern in lower:
                    issues.append(f"Harmful content: '{pattern}'")
                    risk = "high"
                    break

        if risk not in ("critical", "high"):
            for pattern in PII_PATTERNS:
                if pattern in lower:
                    issues.append(f"PII detected: '{pattern}'")
                    risk = "medium"
                    break

        sanitized = self._sanitize(text)
        return SafetyResult(
            safe=not issues,
            issues=issues,
            sanitized_input=sanitized,
            risk_level=risk,
        )

    def check_output(self, text: str, agent_name: str = "") -> SafetyResult:
        issues: list[str] = []
        risk = "none"
        lower = text.lower()

        for pattern in HARMFUL_PATTERNS:
            if pattern in lower:
                issues.append(f"Agent {agent_name!r} produced harmful content")
                risk = "high"
                break

        for signal in SYSTEM_LEAK_SIGNALS:
            if signal in lower:
                issues.append(f"Possible prompt leak from {agent_name!r}")
                risk = "medium" if risk != "high" else risk
                break

        return SafetyResult(safe=not issues, issues=issues,
                            sanitized_input=text, risk_level=risk)

    def check_reasoning_trace(self, trace: str) -> SafetyResult:
        issues = []
        for pattern in INJECTION_PATTERNS:
            if pattern in trace.lower():
                issues.append(f"Injection in reasoning trace: '{pattern}'")
        return SafetyResult(safe=not issues, issues=issues,
                            sanitized_input=trace,
                            risk_level="high" if issues else "none")

    def _sanitize(self, text: str) -> str:
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
