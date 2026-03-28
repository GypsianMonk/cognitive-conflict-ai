"""
Working Memory Buffer
Each debate round has a SHARED WORKING MEMORY that accumulates:
- Agreed premises between agents
- Detected contradictions
- Intermediate conclusions
- Open questions not yet resolved
Like a shared whiteboard — agents build cumulatively, not from scratch.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class MemoryEntry:
    content: str
    entry_type: str         # premise | contradiction | conclusion | open_question
    source_agent: str
    round_added: int
    confidence: float
    timestamp: float = field(default_factory=time.time)


class WorkingMemory:
    """
    Shared mutable context maintained across all debate rounds.
    Agents READ from it at the start of each round and WRITE to it after.
    """

    def __init__(self):
        self._entries: list[MemoryEntry] = []
        self._current_round: int = 0

    def set_round(self, round_num: int):
        self._current_round = round_num

    def add_premise(self, content: str, agent: str, confidence: float = 0.8):
        """Add an agreed-upon or asserted premise."""
        self._add(content, "premise", agent, confidence)

    def add_contradiction(self, content: str, agent: str):
        """Record a contradiction found between agents."""
        self._add(content, "contradiction", agent, 0.0)

    def add_conclusion(self, content: str, agent: str, confidence: float = 0.7):
        """Record an intermediate conclusion."""
        self._add(content, "conclusion", agent, confidence)

    def add_open_question(self, question: str, agent: str):
        """Record an unresolved question for future rounds."""
        self._add(question, "open_question", agent, 0.5)

    def get_premises(self) -> list[MemoryEntry]:
        return [e for e in self._entries if e.entry_type == "premise"]

    def get_contradictions(self) -> list[MemoryEntry]:
        return [e for e in self._entries if e.entry_type == "contradiction"]

    def get_conclusions(self) -> list[MemoryEntry]:
        return [e for e in self._entries if e.entry_type == "conclusion"]

    def get_open_questions(self) -> list[MemoryEntry]:
        return [e for e in self._entries if e.entry_type == "open_question"]

    def build_context_string(self) -> str:
        """Format working memory as a context string for agents."""
        parts = ["=== WORKING MEMORY ==="]

        premises = self.get_premises()
        if premises:
            parts.append("Established premises:")
            for p in premises[-3:]:  # Show last 3
                parts.append(f"  • {p.content} (confidence: {p.confidence:.1f})")

        conclusions = self.get_conclusions()
        if conclusions:
            parts.append("Intermediate conclusions:")
            for c in conclusions[-2:]:
                parts.append(f"  → {c.content}")

        contradictions = self.get_contradictions()
        if contradictions:
            parts.append("Unresolved contradictions:")
            for c in contradictions:
                parts.append(f"  ⚠ {c.content}")

        questions = self.get_open_questions()
        if questions:
            parts.append("Open questions:")
            for q in questions[-2:]:
                parts.append(f"  ? {q.content}")

        parts.append("======================")
        return "\n".join(parts)

    def summary(self) -> dict:
        return {
            "total_entries": len(self._entries),
            "premises": len(self.get_premises()),
            "contradictions": len(self.get_contradictions()),
            "conclusions": len(self.get_conclusions()),
            "open_questions": len(self.get_open_questions()),
        }

    def _add(self, content: str, entry_type: str, agent: str, confidence: float):
        self._entries.append(MemoryEntry(
            content=content,
            entry_type=entry_type,
            source_agent=agent,
            round_added=self._current_round,
            confidence=confidence,
        ))
