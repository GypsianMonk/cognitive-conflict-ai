"""
Agent Persona Evolution
Agents EVOLVE their personas based on which arguments won past debates.
A skeptic agent that keeps being overruled learns to moderate its stance.
Uses lightweight RL-style reward signals to update agent behavior.
"""

from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


DB_PATH = Path("memory/persona_evolution.db")


@dataclass
class AgentPerformanceRecord:
    agent_name: str
    query_domain: str
    won: bool           # Did this agent's response score highest?
    confidence_accuracy: float  # How close was stated confidence to actual outcome?
    contradiction_rate: float   # How often did this agent cause contradictions?


class PersonaEvolver:
    """
    Tracks agent win rates and accuracy per domain.
    Generates updated system prompt modifiers based on performance history.
    """

    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                query_domain TEXT,
                won INTEGER,
                confidence_accuracy REAL,
                contradiction_rate REAL,
                timestamp REAL DEFAULT (unixepoch())
            )
        """)
        self.conn.commit()

    def record_outcome(self, record: AgentPerformanceRecord):
        self.conn.execute(
            "INSERT INTO agent_history (agent_name, query_domain, won, confidence_accuracy, contradiction_rate) "
            "VALUES (?, ?, ?, ?, ?)",
            (record.agent_name, record.query_domain or "general",
             int(record.won), record.confidence_accuracy, record.contradiction_rate),
        )
        self.conn.commit()

    def get_win_rate(self, agent_name: str, domain: str = "") -> float:
        """Return win rate for an agent, optionally filtered by domain."""
        if domain:
            cursor = self.conn.execute(
                "SELECT AVG(won) FROM agent_history WHERE agent_name=? AND query_domain=?",
                (agent_name, domain),
            )
        else:
            cursor = self.conn.execute(
                "SELECT AVG(won) FROM agent_history WHERE agent_name=?", (agent_name,)
            )
        row = cursor.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.5

    def get_persona_modifier(self, agent_name: str, domain: str = "") -> str:
        """
        Generate a system prompt modifier based on this agent's performance history.
        Agents with low win rates are nudged toward the strategies that win.
        """
        win_rate = self.get_win_rate(agent_name, domain)
        avg_contradiction = self._avg_contradiction_rate(agent_name)

        modifiers = []

        if win_rate < 0.3:
            modifiers.append(
                "Your recent arguments have not been the most persuasive. "
                "Focus on being more specific, evidence-based, and acknowledging valid counterpoints."
            )
        elif win_rate > 0.7:
            modifiers.append(
                "Your arguments have been strong. Maintain your current approach "
                "but challenge yourself to explore edge cases you might be missing."
            )

        if avg_contradiction > 0.4:
            modifiers.append(
                "You have been causing many contradictions. Try to build on others' valid points "
                "rather than opposing them entirely."
            )

        if domain:
            domain_wr = self.get_win_rate(agent_name, domain)
            if domain_wr < 0.25:
                modifiers.append(
                    f"You have historically struggled with {domain} topics. "
                    f"Be more cautious and explicit about your uncertainty in this domain."
                )

        return " ".join(modifiers) if modifiers else ""

    def leaderboard(self) -> list[dict]:
        cursor = self.conn.execute("""
            SELECT agent_name, COUNT(*) as total, AVG(won) as win_rate,
                   AVG(confidence_accuracy) as avg_accuracy,
                   AVG(contradiction_rate) as avg_contradictions
            FROM agent_history
            GROUP BY agent_name
            ORDER BY win_rate DESC
        """)
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        for r in rows:
            r["win_rate"] = round(r["win_rate"] or 0, 3)
            r["avg_accuracy"] = round(r["avg_accuracy"] or 0, 3)
        return rows

    def _avg_contradiction_rate(self, agent_name: str) -> float:
        cursor = self.conn.execute(
            "SELECT AVG(contradiction_rate) FROM agent_history WHERE agent_name=?", (agent_name,)
        )
        row = cursor.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0
