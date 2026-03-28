"""
Cross-Session Knowledge Graph
Builds a persistent graph of validated facts from past queries.
When a new query arrives, relevant nodes are retrieved and injected
as grounding context — giving agents factual anchors.
"""

from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DB_PATH = Path("memory/knowledge_graph.db")


@dataclass
class KGNode:
    id: str
    content: str
    node_type: str          # fact | concept | entity | relation
    confidence: float
    source_query: str
    tags: list[str] = field(default_factory=list)


@dataclass
class KGEdge:
    from_id: str
    to_id: str
    relation: str           # supports | contradicts | related_to | part_of | causes


class KnowledgeGraph:
    """
    Persistent knowledge graph built from validated agent outputs.
    Nodes = validated facts/concepts. Edges = relationships between them.
    """

    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS kg_nodes (
                id TEXT PRIMARY KEY,
                content TEXT,
                node_type TEXT,
                confidence REAL,
                source_query TEXT,
                tags TEXT,
                created_at REAL DEFAULT (unixepoch())
            );
            CREATE TABLE IF NOT EXISTS kg_edges (
                from_id TEXT,
                to_id TEXT,
                relation TEXT,
                created_at REAL DEFAULT (unixepoch()),
                PRIMARY KEY (from_id, to_id, relation)
            );
        """)
        self.conn.commit()

    def add_node(self, node: KGNode):
        self.conn.execute(
            "INSERT OR REPLACE INTO kg_nodes (id, content, node_type, confidence, source_query, tags) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (node.id, node.content, node.node_type, node.confidence,
             node.source_query, json.dumps(node.tags)),
        )
        self.conn.commit()

    def add_edge(self, edge: KGEdge):
        self.conn.execute(
            "INSERT OR IGNORE INTO kg_edges (from_id, to_id, relation) VALUES (?, ?, ?)",
            (edge.from_id, edge.to_id, edge.relation),
        )
        self.conn.commit()

    def search(self, query: str, limit: int = 5) -> list[KGNode]:
        """Keyword search across node content."""
        words = query.lower().split()
        if not words:
            return []
        conditions = " OR ".join("LOWER(content) LIKE ?" for _ in words)
        params = [f"%{w}%" for w in words] + [limit]
        cursor = self.conn.execute(
            f"SELECT id, content, node_type, confidence, source_query, tags "
            f"FROM kg_nodes WHERE {conditions} ORDER BY confidence DESC LIMIT ?",
            params,
        )
        return [
            KGNode(
                id=row[0], content=row[1], node_type=row[2],
                confidence=row[3], source_query=row[4],
                tags=json.loads(row[5] or "[]"),
            )
            for row in cursor.fetchall()
        ]

    def build_context(self, query: str) -> str:
        """Build a grounding context string from relevant KG nodes."""
        nodes = self.search(query)
        if not nodes:
            return ""
        parts = ["=== KNOWLEDGE GRAPH CONTEXT ==="]
        for node in nodes:
            parts.append(f"[{node.node_type.upper()}] {node.content} (confidence: {node.confidence:.2f})")
        parts.append("===============================")
        return "\n".join(parts)

    def ingest_from_output(self, output, query: str):
        """Extract and store validated facts from a FinalOutput."""
        import hashlib

        # Store the final answer as a high-confidence fact node
        node_id = hashlib.md5((query + output.final_answer).encode()).hexdigest()[:12]
        node = KGNode(
            id=node_id,
            content=output.final_answer[:300],
            node_type="fact",
            confidence=output.confidence_score / 100.0,
            source_query=query,
            tags=[output.domain or "general", output.complexity],
        )
        self.add_node(node)

    def stats(self) -> dict:
        node_count = self.conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
        edge_count = self.conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        return {"nodes": node_count, "edges": edge_count}
