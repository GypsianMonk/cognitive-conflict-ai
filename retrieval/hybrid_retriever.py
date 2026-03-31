"""
Hybrid Retriever — Dense + BM25 Sparse + TurboQuant Compression
Combines semantic dense retrieval with BM25 keyword search.
Dense retrieval uses TurboQuant (arXiv:2504.19874) for memory-efficient
vector storage: 8-16x compression with near-zero quality loss.
"""

from __future__ import annotations
import math
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from retrieval.turbo_quant import TurboQuantMSE, TurboQuantProd


@dataclass
class Document:
    id: str
    content: str
    source: str
    published_date: Optional[str] = None
    trust_score: float = 0.5
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedDoc:
    document: Document
    dense_score: float
    sparse_score: float
    combined_score: float
    relevance_rank: int


# ── BM25 Retriever (unchanged) ────────────────────────────────────────────────

class BM25Retriever:
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
            tf_counts: dict[str, int] = {}
            for t in tokens:
                tf_counts[t] = tf_counts.get(t, 0) + 1
            for term in query_terms:
                if term not in tf_counts:
                    continue
                tf = tf_counts[term]
                idf = self.idf.get(term, 0.0)
                dl = len(tokens)
                tf_norm = tf * (self.K1 + 1) / (
                    tf + self.K1 * (1 - self.B + self.B * dl / self.avg_dl)
                )
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
        return {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}


# ── TurboQuant Dense Retriever ────────────────────────────────────────────────

class TurboQuantDenseRetriever:
    """
    Dense retrieval using TF-IDF vectorisation + TurboQuant compression.

    Memory usage vs plain DenseRetriever:
      - FP32 dict storage:    O(n × vocab × 4 bytes)
      - TurboQuant 4-bit:     O(n × vocab × 0.5 bytes)  = 8x smaller
      - TurboQuant 2-bit:     O(n × vocab × 0.25 bytes) = 16x smaller

    Quality guarantee (from paper Theorem 2):
      - 4-bit: inner product distortion ≤ 0.047/d per query pair
      - 3-bit: inner product distortion ≤ 0.18/d per query pair
    """

    def __init__(self, documents: list[Document], bit_width: int = 4,
                 vocab_size: int = 4096):
        self.docs = documents
        self.bit_width = bit_width
        self.vocab_size = vocab_size

        # Build vocab from all documents
        all_tokens: set[str] = set()
        self.doc_tokens = []
        for doc in documents:
            tokens = re.findall(r"\b[a-z]{2,}\b", doc.content.lower())
            self.doc_tokens.append(tokens)
            all_tokens.update(tokens)

        self.vocab = {w: i for i, w in enumerate(sorted(all_tokens)[:vocab_size])}
        self.actual_dim = min(len(self.vocab), vocab_size)

        if self.actual_dim == 0 or not documents:
            self.tq = None
            self.quantised = []
            return

        # Vectorise all documents
        raw_vecs = np.zeros((len(documents), self.actual_dim), dtype=np.float32)
        for i, tokens in enumerate(self.doc_tokens):
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                if t in self.vocab:
                    raw_vecs[i, self.vocab[t]] = c

        # L2 normalise
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.doc_vecs_fp32 = raw_vecs / norms   # keep for comparison

        # TurboQuant compress (unbiased inner-product variant)
        self.tq = TurboQuantProd(dim=self.actual_dim, bit_width=bit_width)
        self.quantised = [self.tq.quantise(v) for v in self.doc_vecs_fp32]

        mem = self.tq.memory_bytes(len(documents))
        self._memory_info = mem

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        if not self.tq or not self.quantised:
            return []

        # Vectorise query
        q_tokens = re.findall(r"\b[a-z]{2,}\b", query.lower())
        q_vec = np.zeros(self.actual_dim, dtype=np.float32)
        for t in q_tokens:
            if t in self.vocab:
                q_vec[self.vocab[t]] += 1.0

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec /= q_norm

        # Compute unbiased inner products via TurboQuant
        scores = [
            (self.docs[i], self.tq.inner_product(q_vec, q))
            for i, q in enumerate(self.quantised)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def memory_info(self) -> dict:
        return getattr(self, "_memory_info", {})


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Dense (TurboQuant-compressed) + BM25 sparse retrieval.
    dense_weight=0.6, sparse_weight=0.4 by default.
    """

    def __init__(self, documents: list[Document],
                 dense_weight: float = 0.6,
                 sparse_weight: float = 0.4,
                 bit_width: int = 4):
        self.documents = documents
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        if documents:
            self.dense = TurboQuantDenseRetriever(documents, bit_width=bit_width)
            self.bm25 = BM25Retriever(documents)
        else:
            self.dense = None
            self.bm25 = None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        if not self.documents:
            return []

        dense_results = {d.id: s for d, s in self.dense.retrieve(query, top_k * 2)}
        sparse_results = {d.id: s for d, s in self.bm25.retrieve(query, top_k * 2)}

        max_sparse = max(sparse_results.values(), default=1.0)
        all_ids = set(dense_results) | set(sparse_results)
        combined = []

        for doc_id in all_ids:
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if not doc:
                continue
            d_score = dense_results.get(doc_id, 0.0)
            s_score = sparse_results.get(doc_id, 0.0)
            s_norm = s_score / max_sparse if max_sparse > 0 else 0.0
            score = (d_score * self.dense_weight + s_norm * self.sparse_weight) * doc.trust_score
            combined.append(RetrievedDoc(
                document=doc,
                dense_score=round(d_score, 4),
                sparse_score=round(s_score, 4),
                combined_score=round(score, 4),
                relevance_rank=0,
            ))

        combined.sort(key=lambda r: r.combined_score, reverse=True)
        for i, r in enumerate(combined[:top_k]):
            r.relevance_rank = i + 1
        return combined[:top_k]

    def build_context(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k)
        if not results:
            return ""
        parts = ["=== RETRIEVED CONTEXT ==="]
        for r in results:
            parts.append(
                f"[{r.document.source} | trust={r.document.trust_score:.1f} | "
                f"score={r.combined_score:.3f}]\n{r.document.content[:300]}"
            )
        parts.append("=========================")
        return "\n\n".join(parts)

    def memory_info(self) -> dict:
        if self.dense:
            return self.dense.memory_info()
        return {}
