"""
Metrics — Evaluation metrics and tracking for the Cognitive Conflict system.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class SystemMetrics:
    """Tracks system-level performance over time."""
    total_queries: int = 0
    avg_confidence: float = 0.0
    avg_debate_rounds: float = 0.0
    convergence_rate: float = 0.0
    avg_latency_seconds: float = 0.0
    total_contradictions_detected: int = 0
    domain_distribution: dict = field(default_factory=dict)

    def update(
        self,
        confidence: float,
        debate_rounds: int,
        converged: bool,
        latency: float,
        contradictions: int,
        domain: Optional[str],
    ):
        n = self.total_queries
        self.total_queries += 1
        self.avg_confidence = ((self.avg_confidence * n) + confidence) / (n + 1)
        self.avg_debate_rounds = ((self.avg_debate_rounds * n) + debate_rounds) / (n + 1)
        self.convergence_rate = ((self.convergence_rate * n) + (1 if converged else 0)) / (n + 1)
        self.avg_latency_seconds = ((self.avg_latency_seconds * n) + latency) / (n + 1)
        self.total_contradictions_detected += contradictions

        if domain:
            self.domain_distribution[domain] = self.domain_distribution.get(domain, 0) + 1

    def summary(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "avg_confidence_pct": round(self.avg_confidence, 1),
            "avg_debate_rounds": round(self.avg_debate_rounds, 2),
            "convergence_rate_pct": round(self.convergence_rate * 100, 1),
            "avg_latency_seconds": round(self.avg_latency_seconds, 2),
            "total_contradictions": self.total_contradictions_detected,
            "domain_distribution": self.domain_distribution,
        }


# Global singleton metrics tracker
_metrics = SystemMetrics()


def get_metrics() -> SystemMetrics:
    return _metrics


def record_output(output) -> None:
    """Record metrics from a FinalOutput object."""
    _metrics.update(
        confidence=output.confidence_score,
        debate_rounds=output.total_debate_rounds,
        converged=output.convergence_achieved,
        latency=output.duration_seconds,
        contradictions=len(output.contradictions),
        domain=output.domain,
    )
