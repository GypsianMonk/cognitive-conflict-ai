"""
Graph of Thoughts (GoT) Engine
Reasoning nodes form a directed graph — not just linear chains.
Ideas from different agents can MERGE, SPLIT, and RECOMBINE.
Enables non-linear reasoning breakthroughs impossible in sequential debate.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class ThoughtNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    node_type: str = "thought"   # thought | merge | split | conclusion
    score: float = 0.0
    source_agents: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ThoughtEdge:
    from_id: str
    to_id: str
    relation: str = "leads_to"   # leads_to | challenges | supports | merges_into


class GraphOfThoughts:
    """
    Maintains a directed graph of reasoning nodes.
    Supports merge (combining 2 thoughts), split (diverging), and aggregation.
    """

    def __init__(self):
        self.nodes: dict[str, ThoughtNode] = {}
        self.edges: list[ThoughtEdge] = []

    def add_node(self, content: str, node_type: str = "thought",
                 agent: str = "", score: float = 0.0) -> ThoughtNode:
        node = ThoughtNode(content=content, node_type=node_type,
                           source_agents=[agent] if agent else [], score=score)
        self.nodes[node.id] = node
        return node

    def add_edge(self, from_node: ThoughtNode, to_node: ThoughtNode,
                 relation: str = "leads_to") -> ThoughtEdge:
        edge = ThoughtEdge(from_id=from_node.id, to_id=to_node.id, relation=relation)
        self.edges.append(edge)
        return edge

    def merge_nodes(self, node_a: ThoughtNode, node_b: ThoughtNode,
                    merged_content: str) -> ThoughtNode:
        """Combine two thoughts into one merged node."""
        merged = self.add_node(
            content=merged_content,
            node_type="merge",
            score=(node_a.score + node_b.score) / 2,
        )
        merged.source_agents = list(set(node_a.source_agents + node_b.source_agents))
        self.add_edge(node_a, merged, "merges_into")
        self.add_edge(node_b, merged, "merges_into")
        return merged

    def split_node(self, source: ThoughtNode,
                   branches: list[str]) -> list[ThoughtNode]:
        """Diverge one thought into multiple branches."""
        children = []
        for branch in branches:
            child = self.add_node(content=branch, node_type="split",
                                  score=source.score * 0.9)
            self.add_edge(source, child, "leads_to")
            children.append(child)
        return children

    def get_strongest_path(self) -> list[ThoughtNode]:
        """Return the highest-scoring chain through the graph using greedy traversal."""
        if not self.nodes:
            return []

        # Find root nodes (no incoming edges)
        to_ids = {e.to_id for e in self.edges}
        roots = [n for n in self.nodes.values() if n.id not in to_ids]
        if not roots:
            roots = [max(self.nodes.values(), key=lambda n: n.score)]

        best_path: list[ThoughtNode] = []
        best_score = -1.0

        def dfs(node: ThoughtNode, path: list[ThoughtNode]):
            nonlocal best_path, best_score
            path = path + [node]
            children_ids = [e.to_id for e in self.edges if e.from_id == node.id]
            if not children_ids:
                score = sum(n.score for n in path)
                if score > best_score:
                    best_score = score
                    best_path = path
            else:
                for cid in children_ids:
                    if cid in self.nodes:
                        dfs(self.nodes[cid], path)

        for root in roots:
            dfs(root, [])

        return best_path

    def detect_cycles(self) -> list[list[str]]:
        """Detect circular reasoning loops in the graph."""
        visited, cycles = set(), []

        def dfs(node_id: str, path: list[str]):
            if node_id in path:
                cycle_start = path.index(node_id)
                cycles.append(path[cycle_start:])
                return
            if node_id in visited:
                return
            visited.add(node_id)
            for edge in self.edges:
                if edge.from_id == node_id:
                    dfs(edge.to_id, path + [node_id])

        for node_id in self.nodes:
            dfs(node_id, [])

        return cycles

    def summary(self) -> dict:
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "merge_nodes": sum(1 for n in self.nodes.values() if n.node_type == "merge"),
            "cycles_detected": len(self.detect_cycles()),
            "strongest_path_length": len(self.get_strongest_path()),
        }
