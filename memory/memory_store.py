"""
Memory Store — Persists past queries, conflicts, and outcomes for learning.
Uses SQLite for lightweight local storage. Swap for Redis/Postgres in production.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional


DB_PATH = Path("memory/conflict_memory.db")


class MemoryStore:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                domain TEXT,
                complexity TEXT,
                final_answer TEXT,
                confidence REAL,
                convergence_achieved INTEGER,
                contradictions TEXT,
                debate_rounds INTEGER,
                timestamp REAL
            )
        """)
        self.conn.commit()

    def save(self, output) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO query_history
                (query, domain, complexity, final_answer, confidence,
                 convergence_achieved, contradictions, debate_rounds, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                output.query,
                output.domain,
                output.complexity,
                output.final_answer,
                output.confidence_score,
                int(output.convergence_achieved),
                json.dumps(output.contradictions),
                output.total_debate_rounds,
                time.time(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def search_similar(self, query: str, limit: int = 5) -> list[dict]:
        """Simple keyword search. Replace with vector similarity in production."""
        words = query.lower().split()
        conditions = " OR ".join(["LOWER(query) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]

        cursor = self.conn.execute(
            f"""
            SELECT query, domain, final_answer, confidence, timestamp
            FROM query_history
            WHERE {conditions}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params + [limit],
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_all(self, limit: int = 100) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT * FROM query_history ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()
