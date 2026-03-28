"""
Pipeline Architecture
Replaces the 160-line God method in DebateManager.
Each step is a discrete, testable unit. The runner chains them cleanly.

Step protocol: each step receives PipelineState and returns it (mutated or replaced).
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from agents.base_agent import AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Shared state passed through every pipeline step."""
    query: str
    config: object                              # AppConfig

    # Built up through the pipeline
    analysis: Optional[object] = None          # QueryAnalysis
    context: str = ""                          # Accumulated retrieval/KG context
    agents: list = field(default_factory=list)
    responses: list[AgentResponse] = field(default_factory=list)
    round_results: list = field(default_factory=list)
    working_memory: Optional[object] = None
    scored: list = field(default_factory=list)
    final_answer: str = ""
    reasoning_trace: str = ""
    confidence: float = 0.0
    convergence_achieved: bool = False
    rounds_run: int = 0
    reflection_count: int = 0
    task_decomposed: bool = False
    kg_context_used: bool = False
    tot_reasoning: str = ""
    got_summary: dict = field(default_factory=dict)

    # Reports
    safety_result: Optional[object] = None
    hal_report: Optional[object] = None
    fact_report: Optional[object] = None
    uncertainty: Optional[object] = None

    # Bookkeeping
    blocked: bool = False
    block_reason: str = ""
    start_time: float = field(default_factory=time.time)
    audit_query_id: str = ""
    errors: list[str] = field(default_factory=list)

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def add_context(self, text: str) -> None:
        if text and text.strip():
            self.context = (self.context + "\n\n" + text).strip()


@runtime_checkable
class PipelineStep(Protocol):
    name: str
    def run(self, state: PipelineState) -> PipelineState: ...


class Pipeline:
    """Runs a sequence of steps, stopping if any step sets state.blocked=True."""

    def __init__(self, steps: list):
        self.steps = steps

    def run(self, state: PipelineState) -> PipelineState:
        for step in self.steps:
            if state.blocked:
                break
            try:
                t0 = time.time()
                state = step.run(state)
                logger.debug("Step %-30s completed in %.2fs", step.name, time.time() - t0)
            except Exception as e:
                logger.error("Step %s failed: %s", step.name, e, exc_info=True)
                state.errors.append(f"{step.name}: {e}")
                # Continue — don't let one step kill the whole pipeline
        return state
