"""
Hybrid RAG — Dense + Sparse Retrieval
Combines vector (semantic) search with BM25 keyword search.
Dense: finds conceptually similar documents.
Sparse: finds exact keyword matches.
Together they produce the most relevant context for agent grounding.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    id: str
    content: str
    source: str
    published_date: Optional[str] = None
    trust_score: float = 0.5        # 0 = untrusted, 1 = fully trusted
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedDoc:
    document: Document
    dense_score: float
    sparse_score: float
    combined_score: float
    relevance_rank: int


class BM25Retriever:
    """
    Classic BM25 sparse retrieval over an in-memory document corpus.
    In production: replace with Elasticsearch or OpenSearch.
    """

    K1, B = 1.5, 0.75

    def __init__(self, documents: list[Document]):
        self.docs = documents
        self.corpus = [self._tokenize(d.content) for d in documents]
        self.avg_dl = sum(len(t) for t in self.corpus) / max(1, len(self.corpus))
        self.idf = self._compute_idf()

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        query_terms = self._tokenize(query)
        scores = []

        for idx, tokens in enumerate(self.corpus):
            score = 0.0
            tf_counts = {}
            for t in tokens:
                tf_counts[t] = tf_counts.get(t, 0) + 1

            for term in query_terms:
                if term not in tf_counts:
                    continue
                tf = tf_counts[term]
                idf = self.idf.get(term, 0.0)
                dl = len(tokens)
                tf_norm = tf * (self.K1 + 1) / (tf + self.K1 * (1 - self.B + self.B * dl / self.avg_dl))
                score += idf * tf_norm

            scores.append((self.docs[idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def _compute_idf(self) -> dict[str, float]:
        N = len(self.corpus)
        df: dict[str, int] = {}
        for tokens in self.corpus:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        return {term: math.log((N - n + 0.5) / (n + 0.5) + 1) for term, n in df.items()}


class DenseRetriever:
    """
    Dense retrieval using TF-IDF cosine similarity as a lightweight stand-in.
    In production: replace with sentence-transformers + vector DB (Chroma, Pinecone, Weaviate).
    """

    def __init__(self, documents: list[Document]):
        self.docs = documents
        self.doc_vectors = [self._vectorize(d.content) for d in documents]

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        query_vec = self._vectorize(query)
        scores = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine(query_vec, doc_vec)
            scores.append((self.docs[idx], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _vectorize(self, text: str) -> dict[str, float]:
        tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
        counts: dict[str, float] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        # L2 normalize
        magnitude = math.sqrt(sum(v ** 2 for v in counts.values())) or 1.0
        return {k: v / magnitude for k, v in counts.items()}

    def _cosine(self, a: dict, b: dict) -> float:
        return sum(a.get(k, 0) * v for k, v in b.items())


class HybridRetriever:
    """
    Combines BM25 (sparse) + Dense retrieval with a weighted fusion.
    Falls back gracefully if the corpus is empty.
    """

    def __init__(self, documents: list[Document],
                 dense_weight: float = 0.6, sparse_weight: float = 0.4):
        self.documents = documents
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        if documents:
            self.bm25 = BM25Retriever(documents)
            self.dense = DenseRetriever(documents)
        else:
            self.bm25 = None
            self.dense = None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        if not self.documents:
            return []

        dense_results = {d.id: s for d, s in self.dense.retrieve(query, top_k * 2)}
        sparse_results = {d.id: s for d, s in self.bm25.retrieve(query, top_k * 2)}

        all_ids = set(dense_results) | set(sparse_results)
        combined = []

        for doc_id in all_ids:
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if not doc:
                continue

            d_score = dense_results.get(doc_id, 0.0)
            s_score = sparse_results.get(doc_id, 0.0)

            # Normalize sparse score (BM25 is unbounded)
            max_sparse = max(sparse_results.values()) if sparse_results else 1.0
            s_norm = s_score / max_sparse if max_sparse > 0 else 0.0

            combined_score = (d_score * self.dense_weight + s_norm * self.sparse_weight)
            # Apply document trust score as a multiplier
            combined_score *= doc.trust_score

            combined.append(RetrievedDoc(
                document=doc,
                dense_score=round(d_score, 4),
                sparse_score=round(s_score, 4),
                combined_score=round(combined_score, 4),
                relevance_rank=0,
            ))

        combined.sort(key=lambda r: r.combined_score, reverse=True)
        for idx, r in enumerate(combined[:top_k]):
            r.relevance_rank = idx + 1

        return combined[:top_k]

    def build_context(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k)
        if not results:
            return ""
        parts = ["=== RETRIEVED CONTEXT ==="]
        for r in results:
            parts.append(
                f"[Source: {r.document.source} | Trust: {r.document.trust_score:.1f} | "
                f"Score: {r.combined_score:.3f}]\n{r.document.content[:300]}"
            )
        parts.append("=========================")
        return "\n\n".join(parts)
