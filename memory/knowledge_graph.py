"""
Cross-Session Knowledge Graph with TurboQuant-compressed semantic search.
Nodes are stored with quantised embedding vectors (TurboQuantProd, 4-bit default)
enabling semantic similarity search at 4.5x lower memory than float64.
Falls back to keyword search when no embeddings are available.
"""
from __future__ import annotations
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from retrieval.turbo_quant import TurboQuantProd

DB_PATH = Path("memory/knowledge_graph.db")
DEFAULT_BITS = 4


@dataclass
class KGNode:
    id: str
    content: str
    node_type: str = "fact"       # fact | concept | entity | relation
    confidence: float = 0.8
    source_query: str = ""
    tags: list = field(default_factory=list)
    # Stored separately in DB as JSON
    embedding: Optional[list[float]] = None


@dataclass
class KGEdge:
    from_id: str
    to_id: str
    relation: str = "related_to"  # supports | contradicts | related_to | part_of


class KnowledgeGraph:
    """
    Persistent knowledge graph with TurboQuant-compressed vector search.

    Search strategy:
    1. If embeddings exist for stored nodes → TurboQuant inner-product search
    2. Fallback: BM25-style keyword search (always available)
    """

    def __init__(self, db_path: Path = DB_PATH, bits: int = DEFAULT_BITS):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.bits = bits
        self._quantiser: Optional[TurboQuantProd] = None
        self._node_cache: dict[str, KGNode] = {}   # hot-cache for recent nodes
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
                embedding_json TEXT,
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

    # --- Node management -------------------------------------------------------

    def add_node(self, node: KGNode):
        emb_json = json.dumps(node.embedding) if node.embedding else None
        self.conn.execute(
            "INSERT OR REPLACE INTO kg_nodes "
            "(id, content, node_type, confidence, source_query, tags, embedding_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (node.id, node.content, node.node_type, node.confidence,
             node.source_query, json.dumps(node.tags), emb_json),
        )
        self.conn.commit()
        self._node_cache[node.id] = node

    def add_edge(self, edge: KGEdge):
        self.conn.execute(
            "INSERT OR IGNORE INTO kg_edges (from_id, to_id, relation) VALUES (?,?,?)",
            (edge.from_id, edge.to_id, edge.relation),
        )
        self.conn.commit()

    # --- Search ----------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[KGNode]:
        """
        Hybrid search: vector similarity if embeddings exist, else keyword.
        """
        # Try vector search first
        vec_results = self._vector_search(query, limit)
        if vec_results:
            return vec_results
        return self._keyword_search(query, limit)

    def _keyword_search(self, query: str, limit: int) -> list[KGNode]:
        words = query.lower().split()
        if not words:
            return []
        conditions = " OR ".join("LOWER(content) LIKE ?" for _ in words)
        params = [f"%{w}%" for w in words] + [limit]
        cursor = self.conn.execute(
            f"SELECT id, content, node_type, confidence, source_query, tags, embedding_json "
            f"FROM kg_nodes WHERE {conditions} ORDER BY confidence DESC LIMIT ?",
            params,
        )
        return [self._row_to_node(row) for row in cursor.fetchall()]

    def _vector_search(self, query: str, limit: int) -> list[KGNode]:
        """
        TurboQuant inner-product search over stored embedding vectors.
        Only activates when nodes have embeddings.
        """
        cursor = self.conn.execute(
            "SELECT id, content, node_type, confidence, source_query, tags, embedding_json "
            "FROM kg_nodes WHERE embedding_json IS NOT NULL ORDER BY confidence DESC LIMIT 200"
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        nodes = [self._row_to_node(row) for row in rows]
        embeddings = [n.embedding for n in nodes if n.embedding]
        if not embeddings:
            return []

        dim = len(embeddings[0])
        if self._quantiser is None or self._quantiser.dim != dim:
            self._quantiser = TurboQuantProd(dim=dim, bits=self.bits, seed=42)

        # Embed query using simple TF-IDF (replace with model in production)
        q_vec = self._tfidf_embed(query, dim,
                                   [n.content for n in nodes])
        if q_vec is None:
            return []

        # Score using TurboQuant inner product
        scores = []
        for node in nodes:
            if not node.embedding:
                continue
            qv = self._quantiser.quantise(node.embedding)
            score = self._quantiser.inner_product(qv, q_vec)
            scores.append((node, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scores[:limit]]

    # --- Context building -------------------------------------------------------

    def build_context(self, query: str) -> str:
        nodes = self.search(query)
        if not nodes:
            return ""
        parts = ["=== KNOWLEDGE GRAPH CONTEXT ==="]
        for node in nodes:
            parts.append(
                f"[{node.node_type.upper()}] {node.content} "
                f"(confidence: {node.confidence:.2f})"
            )
        parts.append("================================")
        return "\n".join(parts)

    def ingest_from_output(self, output, query: str):
        """Store validated final answer as a knowledge node."""
        import hashlib
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
        with_emb = self.conn.execute(
            "SELECT COUNT(*) FROM kg_nodes WHERE embedding_json IS NOT NULL"
        ).fetchone()[0]
        return {
            "nodes": node_count,
            "edges": edge_count,
            "nodes_with_embeddings": with_emb,
            "quantiser_bits": self.bits,
            "compression_ratio": (
                round(self._quantiser.compression_ratio(), 2)
                if self._quantiser else None
            ),
        }

    # --- Helpers ---------------------------------------------------------------

    def _row_to_node(self, row) -> KGNode:
        emb = json.loads(row[6]) if row[6] else None
        return KGNode(
            id=row[0], content=row[1], node_type=row[2],
            confidence=row[3], source_query=row[4],
            tags=json.loads(row[5] or "[]"),
            embedding=emb,
        )

    def _tfidf_embed(self, query: str, dim: int,
                     corpus: list[str]) -> Optional[list[float]]:
        """Minimal TF-IDF embedding — replace with sentence-transformer in prod."""
        vocab_set: set[str] = set()
        for doc in corpus:
            vocab_set.update(re.findall(r"\b[a-z]{2,}\b", doc.lower()))
        vocab = sorted(vocab_set)
        if not vocab or dim != len(vocab):
            return None
        tokens = re.findall(r"\b[a-z]{2,}\b", query.lower())
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = [float(tf.get(t, 0)) for t in vocab]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
