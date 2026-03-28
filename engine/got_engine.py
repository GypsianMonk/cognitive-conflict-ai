"""
Graph of Thoughts (GoT) Engine
Reasoning nodes form a directed graph — ideas from different agents
can merge, split, and recombine. GotNode/GotEdge (no name collision).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class GotNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    node_type: str = "thought"   # thought | merge | split | conclusion
    score: float = 0.0
    source_agents: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class GotEdge:
    from_id: str
    to_id: str
    relation: str = "leads_to"


class GraphOfThoughts:
    def __init__(self):
        self.nodes: dict[str, GotNode] = {}
        self.edges: list[GotEdge] = []

    def add_node(self, content: str, node_type: str = "thought",
                 agent: str = "", score: float = 0.0) -> GotNode:
        node = GotNode(content=content, node_type=node_type,
                       source_agents=[agent] if agent else [], score=score)
        self.nodes[node.id] = node
        return node

    def add_edge(self, from_node: GotNode, to_node: GotNode,
                 relation: str = "leads_to") -> GotEdge:
        edge = GotEdge(from_id=from_node.id, to_id=to_node.id, relation=relation)
        self.edges.append(edge)
        return edge

    def merge_nodes(self, a: GotNode, b: GotNode, content: str) -> GotNode:
        merged = self.add_node(content=content, node_type="merge",
                               score=(a.score + b.score) / 2)
        merged.source_agents = list(set(a.source_agents + b.source_agents))
        self.add_edge(a, merged, "merges_into")
        self.add_edge(b, merged, "merges_into")
        return merged

    def split_node(self, source: GotNode, branches: list[str]) -> list[GotNode]:
        children = []
        for branch in branches:
            child = self.add_node(content=branch, node_type="split",
                                  score=source.score * 0.9)
            self.add_edge(source, child, "leads_to")
            children.append(child)
        return children

    def get_strongest_path(self) -> list[GotNode]:
        if not self.nodes:
            return []
        to_ids = {e.to_id for e in self.edges}
        roots = [n for n in self.nodes.values() if n.id not in to_ids]
        if not roots:
            roots = [max(self.nodes.values(), key=lambda n: n.score)]

        best_path: list[GotNode] = []
        best_score = -1.0

        stack = [(r, []) for r in roots]
        while stack:
            node, path = stack.pop()
            path = path + [node]
            children_ids = [e.to_id for e in self.edges if e.from_id == node.id]
            if not children_ids:
                score = sum(n.score for n in path)
                if score > best_score:
                    best_score = score
                    best_path = path[:]
            else:
                for cid in children_ids:
                    if cid in self.nodes:
                        stack.append((self.nodes[cid], path))
        return best_path

    def detect_cycles(self) -> list[list[str]]:
        visited: set[str] = set()
        cycles: list[list[str]] = []

        def dfs(node_id: str, path: list[str]) -> None:
            if node_id in path:
                cycles.append(path[path.index(node_id):])
                return
            if node_id in visited:
                return
            visited.add(node_id)
            for edge in self.edges:
                if edge.from_id == node_id:
                    dfs(edge.to_id, path + [node_id])

        for node_id in list(self.nodes):
            dfs(node_id, [])
        return cycles

    def build_from_agent_responses(self, responses: list) -> Optional[GotNode]:
        """Build graph from agent responses and return a merged synthesis node."""
        if not responses:
            return None
        nodes = []
        for r in responses:
            n = self.add_node(content=r.answer, node_type="thought",
                              agent=r.agent_name, score=r.confidence)
            nodes.append(n)
        if len(nodes) >= 2:
            merged_content = " | ".join(n.content[:80] for n in nodes[:3])
            return self.merge_nodes(nodes[0], nodes[1], merged_content)
        return nodes[0] if nodes else None

    def summary(self) -> dict:
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "merge_nodes": sum(1 for n in self.nodes.values() if n.node_type == "merge"),
            "cycles_detected": len(self.detect_cycles()),
            "strongest_path_length": len(self.get_strongest_path()),
        }
