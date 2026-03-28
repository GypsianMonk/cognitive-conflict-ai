"""
Audit Logger — Full Decision Provenance
Every decision, agent response, contradiction, and scoring step is logged
with full provenance. Enables post-hoc explainability and compliance.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


LOG_DIR = Path("logs")


@dataclass
class AuditEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""        # query | agent_response | debate_round | score | judge | output
    query_id: str = ""
    agent: Optional[str] = None
    data: dict = field(default_factory=dict)
    duration_ms: Optional[float] = None


class AuditLogger:
    """
    Structured audit log for every step of the Cognitive Conflict AI pipeline.
    Writes JSON-Lines to a rotating log file and maintains an in-memory buffer
    for the current session.
    """

    def __init__(self, log_dir: Path = LOG_DIR, session_id: Optional[str] = None):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.log_file = LOG_DIR / f"audit_{self.session_id}.jsonl"
        self._buffer: list[AuditEvent] = []
        self._query_id: str = ""

    def new_query(self, query: str) -> str:
        self._query_id = str(uuid.uuid4())[:12]
        self._log(AuditEvent(
            event_type="query",
            query_id=self._query_id,
            data={"query": query, "session_id": self.session_id},
        ))
        return self._query_id

    def log_agent_response(self, agent_name: str, answer: str,
                            reasoning: str, confidence: float,
                            duration_ms: float = 0.0):
        self._log(AuditEvent(
            event_type="agent_response",
            query_id=self._query_id,
            agent=agent_name,
            duration_ms=duration_ms,
            data={
                "answer_snippet": answer[:200],
                "reasoning_snippet": reasoning[:200],
                "confidence": confidence,
            },
        ))

    def log_debate_round(self, round_num: int, num_agents: int,
                          contradictions: list[str], convergence: float):
        self._log(AuditEvent(
            event_type="debate_round",
            query_id=self._query_id,
            data={
                "round": round_num,
                "agents": num_agents,
                "contradictions": contradictions,
                "convergence_score": convergence,
            },
        ))

    def log_scores(self, scored_responses: list[dict]):
        self._log(AuditEvent(
            event_type="score",
            query_id=self._query_id,
            data={"scored": scored_responses},
        ))

    def log_hallucination_check(self, risk: float, flag_count: int, flags: list[dict]):
        self._log(AuditEvent(
            event_type="hallucination_check",
            query_id=self._query_id,
            data={"risk": risk, "flag_count": flag_count, "flags": flags},
        ))

    def log_final_output(self, final_answer: str, confidence: float,
                          domain: Optional[str], complexity: str):
        self._log(AuditEvent(
            event_type="output",
            query_id=self._query_id,
            data={
                "answer_snippet": final_answer[:300],
                "confidence_score": confidence,
                "domain": domain,
                "complexity": complexity,
            },
        ))

    def get_session_log(self) -> list[dict]:
        return [self._event_to_dict(e) for e in self._buffer]

    def _log(self, event: AuditEvent):
        self._buffer.append(event)
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(self._event_to_dict(event)) + "\n")
        except IOError:
            pass  # Never crash due to logging failure

    def _event_to_dict(self, event: AuditEvent) -> dict:
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "query_id": event.query_id,
            "agent": event.agent,
            "duration_ms": event.duration_ms,
            "data": event.data,
        }


# Global logger instance (one per process)
_global_logger: Optional[AuditLogger] = None


def get_logger() -> AuditLogger:
    global _global_logger
    if _global_logger is None:
        _global_logger = AuditLogger()
    return _global_logger
