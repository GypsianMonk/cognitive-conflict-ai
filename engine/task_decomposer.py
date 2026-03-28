"""
Autonomous Task Decomposer
For complex queries, an orchestrator agent DECOMPOSES the task into subtasks,
assigns them to specialist agents, and synthesizes results.
Enables handling multi-step problems: research, writing, analysis pipelines.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class Subtask:
    id: str
    description: str
    assigned_agent_role: str    # optimist | skeptic | alternative | expert | fact_checker
    depends_on: list[str] = field(default_factory=list)   # ids of prerequisite subtasks
    result: Optional[str] = None
    completed: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "assigned_to": self.assigned_agent_role,
            "depends_on": self.depends_on,
            "completed": self.completed,
            "result_snippet": (self.result or "")[:150],
        }


@dataclass
class DecompositionPlan:
    original_query: str
    subtasks: list[Subtask]
    complexity: str
    estimated_agents_needed: int

    def execution_order(self) -> list[list[Subtask]]:
        """Return subtasks in dependency-aware execution waves."""
        completed_ids: set[str] = set()
        waves: list[list[Subtask]] = []
        remaining = list(self.subtasks)

        while remaining:
            ready = [t for t in remaining if all(d in completed_ids for d in t.depends_on)]
            if not ready:
                break  # Circular dependency — add remaining as a final wave
            waves.append(ready)
            completed_ids.update(t.id for t in ready)
            remaining = [t for t in remaining if t.id not in completed_ids]

        if remaining:
            waves.append(remaining)

        return waves


class TaskDecomposer:
    """
    Analyzes a query and breaks it into manageable, assignable subtasks.
    Uses pattern matching for common task types; LLM in production.
    """

    MULTI_STEP_SIGNALS = [
        "and then", "after that", "first", "second", "finally",
        "research", "analyze", "compare", "summarize", "write a report",
        "build a plan", "step by step", "multiple", "both",
    ]

    def should_decompose(self, query: str) -> bool:
        query_lower = query.lower()
        word_count = len(query.split())
        signal_count = sum(1 for s in self.MULTI_STEP_SIGNALS if s in query_lower)
        return word_count > 25 or signal_count >= 2

    def decompose(self, query: str) -> DecompositionPlan:
        subtasks = self._generate_subtasks(query)
        return DecompositionPlan(
            original_query=query,
            subtasks=subtasks,
            complexity="complex" if len(subtasks) > 3 else "medium",
            estimated_agents_needed=min(4, len(subtasks)),
        )

    def synthesize_results(self, plan: DecompositionPlan) -> str:
        """Combine all completed subtask results into a coherent answer."""
        parts = []
        for task in plan.subtasks:
            if task.result:
                parts.append(f"[{task.description}]\n{task.result}")
        return "\n\n".join(parts) if parts else "No subtask results available."

    def _generate_subtasks(self, query: str) -> list[Subtask]:
        """
        Generate a decomposition plan.
        In production, call LLM: 'Break this query into 3-5 specific subtasks.'
        """
        subtasks = []

        # Always: gather context
        subtasks.append(Subtask(
            id="t1",
            description="Gather and summarize relevant background information",
            assigned_agent_role="alternative",
            depends_on=[],
        ))

        # Always: direct analysis
        subtasks.append(Subtask(
            id="t2",
            description="Provide a direct analysis and answer",
            assigned_agent_role="optimist",
            depends_on=["t1"],
        ))

        # Always: critical review
        subtasks.append(Subtask(
            id="t3",
            description="Critically review the analysis for gaps and risks",
            assigned_agent_role="skeptic",
            depends_on=["t2"],
        ))

        # For research-type queries: add fact-check step
        query_lower = query.lower()
        if any(w in query_lower for w in ["research", "statistics", "data", "evidence", "prove"]):
            subtasks.append(Subtask(
                id="t4",
                description="Verify key claims with external sources",
                assigned_agent_role="fact_checker",
                depends_on=["t2"],
            ))

        # For complex or long queries: add synthesis step
        if len(query.split()) > 20:
            subtasks.append(Subtask(
                id="t5",
                description="Synthesize all perspectives into a unified recommendation",
                assigned_agent_role="optimist",
                depends_on=[t.id for t in subtasks],
            ))

        return subtasks
