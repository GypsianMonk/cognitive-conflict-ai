"""
Tree of Thoughts (ToT) Engine
Agents explore a branching tree of reasoning paths, pruning weak branches.
ThoughtNode renamed to TotNode to avoid collision with got_engine.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class TotNode:
    thought: str
    depth: int
    score: float = 0.0
    parent: Optional["TotNode"] = None
    children: list = field(default_factory=list)
    pruned: bool = False

    def path_to_root(self) -> list[str]:
        path, node = [], self
        while node:
            path.append(node.thought)
            node = node.parent
        return list(reversed(path))

    def full_reasoning(self) -> str:
        return " → ".join(self.path_to_root())


class TreeOfThoughts:
    def __init__(
        self,
        max_depth: int = 3,
        branching_factor: int = 3,
        beam_width: int = 2,
        score_fn: Optional[Callable] = None,
        generate_fn: Optional[Callable] = None,
        llm_call_fn: Optional[Callable] = None,
    ):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.beam_width = beam_width
        self._score_fn = score_fn or self._default_score
        self._generate_fn = generate_fn or self._default_generate
        self._llm_call_fn = llm_call_fn   # Injected from agent if available

    def run(self, query: str) -> tuple[str, list]:
        root = TotNode(thought=query, depth=0, score=1.0)
        beam: list[TotNode] = [root]

        for depth in range(1, self.max_depth + 1):
            candidates: list[TotNode] = []
            for node in beam:
                branches = self._generate_fn(query, node.thought)
                for branch in branches[:self.branching_factor]:
                    child = TotNode(thought=branch, depth=depth, parent=node)
                    child.score = self._score_fn(query, child.full_reasoning())
                    node.children.append(child)
                    candidates.append(child)

            candidates.sort(key=lambda n: n.score, reverse=True)
            for p in candidates[self.beam_width:]:
                p.pruned = True
            beam = candidates[:self.beam_width]

        all_leaves = self._collect_leaves(root)
        best = max(beam, key=lambda n: n.score) if beam else root
        return best.full_reasoning(), all_leaves

    def _collect_leaves(self, root: TotNode) -> list[TotNode]:
        leaves, stack = [], [root]
        while stack:
            node = stack.pop()
            if not node.children:
                leaves.append(node)
            else:
                stack.extend(node.children)
        return leaves

    def _default_score(self, query: str, reasoning: str) -> float:
        steps = reasoning.count("→") + 1
        length_score = min(1.0, len(reasoning) / 400)
        depth_score = min(1.0, steps / self.max_depth)
        return round((length_score * 0.5) + (depth_score * 0.5), 4)

    def _default_generate(self, query: str, current: str) -> list[str]:
        """
        If an LLM call function is injected, use it for real branching.
        Otherwise use structured heuristic branches.
        """
        if self._llm_call_fn:
            try:
                prompt = (
                    f"Query: {query}\nCurrent reasoning step: {current}\n"
                    f"Generate {self.branching_factor} distinct next reasoning steps. "
                    f"One per line, no numbering."
                )
                raw = self._llm_call_fn(prompt)
                branches = [l.strip() for l in raw.strip().splitlines() if l.strip()]
                if branches:
                    return branches[:self.branching_factor]
            except Exception:
                pass

        # Heuristic fallback
        return [
            f"{current} → direct solution",
            f"{current} → challenge assumptions",
            f"{current} → alternative perspective",
        ]
