"""
Preference Learning — Feedback-Driven Scoring
When users rate answers, that feedback trains a preference model
that reweights scoring criteria over time.
The system learns what quality means for your use case.
"""

from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DB_PATH = Path("memory/preference_model.db")


@dataclass
class UserFeedback:
    query_id: str
    rating: int             # 1-5 stars
    helpful: bool
    comment: Optional[str]
    domain: Optional[str]


@dataclass
class LearnedWeights:
    relevance_weight: float = 0.4
    coherence_weight: float = 0.4
    contradiction_penalty: float = 0.2
    freshness_weight: float = 0.0
    confidence_weight: float = 0.0
    iteration: int = 0

    def to_dict(self) -> dict:
        return {
            "relevance_weight": round(self.relevance_weight, 4),
            "coherence_weight": round(self.coherence_weight, 4),
            "contradiction_penalty": round(self.contradiction_penalty, 4),
            "freshness_weight": round(self.freshness_weight, 4),
            "confidence_weight": round(self.confidence_weight, 4),
            "iteration": self.iteration,
        }


class PreferenceModel:
    """
    Collects user feedback and adjusts scoring weights to match user preferences.
    Uses a simple gradient-free hill-climbing approach.
    Replace with a proper reward model (RLHF-style) in production.
    """

    LEARNING_RATE = 0.02
    MIN_FEEDBACK_FOR_UPDATE = 10

    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()
        self._weights = self._load_weights()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                rating INTEGER,
                helpful INTEGER,
                comment TEXT,
                domain TEXT,
                timestamp REAL DEFAULT (unixepoch())
            );
            CREATE TABLE IF NOT EXISTS learned_weights (
                id INTEGER PRIMARY KEY,
                weights TEXT,
                iteration INTEGER DEFAULT 0,
                updated_at REAL DEFAULT (unixepoch())
            );
        """)
        self.conn.commit()

    def record_feedback(self, feedback: UserFeedback):
        self.conn.execute(
            "INSERT INTO feedback (query_id, rating, helpful, comment, domain) VALUES (?,?,?,?,?)",
            (feedback.query_id, feedback.rating, int(feedback.helpful),
             feedback.comment, feedback.domain),
        )
        self.conn.commit()

        # Check if we have enough feedback to update weights
        count = self.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if count % self.MIN_FEEDBACK_FOR_UPDATE == 0:
            self._update_weights()

    def get_weights(self) -> LearnedWeights:
        return self._weights

    def get_feedback_stats(self) -> dict:
        cursor = self.conn.execute(
            "SELECT COUNT(*), AVG(rating), SUM(helpful) FROM feedback"
        )
        row = cursor.fetchone()
        total = row[0] or 0
        return {
            "total_feedback": total,
            "avg_rating": round(row[1] or 0, 2),
            "helpful_rate": round((row[2] or 0) / max(1, total), 3),
            "current_weights": self._weights.to_dict(),
        }

    def _update_weights(self):
        """
        Simple heuristic weight adjustment based on feedback patterns.
        High ratings → current weights are good → small perturbation.
        Low ratings → shift weights toward underrepresented dimensions.
        """
        cursor = self.conn.execute(
            "SELECT AVG(rating), AVG(helpful) FROM feedback ORDER BY timestamp DESC LIMIT 50"
        )
        row = cursor.fetchone()
        avg_rating = row[0] or 3.0
        helpful_rate = row[1] or 0.5

        lr = self.LEARNING_RATE
        w = self._weights

        if avg_rating < 3.0:
            # Users unhappy → boost coherence and reduce contradiction tolerance
            w.coherence_weight = min(0.6, w.coherence_weight + lr)
            w.contradiction_penalty = min(0.4, w.contradiction_penalty + lr)
            w.relevance_weight = max(0.2, w.relevance_weight - lr)
        elif avg_rating > 4.0:
            # Users happy → small boost to relevance (usually key driver)
            w.relevance_weight = min(0.6, w.relevance_weight + lr * 0.5)

        if helpful_rate < 0.4:
            # Not helpful → boost freshness (possibly stale answers)
            w.freshness_weight = min(0.2, w.freshness_weight + lr)

        # Normalize weights to sum to 1 (excluding penalties)
        total = w.relevance_weight + w.coherence_weight + w.freshness_weight + w.confidence_weight
        if total > 0:
            scale = 1.0 / total
            w.relevance_weight = round(w.relevance_weight * scale, 4)
            w.coherence_weight = round(w.coherence_weight * scale, 4)
            w.freshness_weight = round(w.freshness_weight * scale, 4)
            w.confidence_weight = round(w.confidence_weight * scale, 4)

        w.iteration += 1
        self._save_weights(w)
        self._weights = w

    def _save_weights(self, w: LearnedWeights):
        self.conn.execute(
            "INSERT OR REPLACE INTO learned_weights (id, weights, iteration) VALUES (1, ?, ?)",
            (json.dumps(w.to_dict()), w.iteration),
        )
        self.conn.commit()

    def _load_weights(self) -> LearnedWeights:
        cursor = self.conn.execute("SELECT weights FROM learned_weights WHERE id=1")
        row = cursor.fetchone()
        if row:
            data = json.loads(row[0])
            return LearnedWeights(**data)
        return LearnedWeights()
