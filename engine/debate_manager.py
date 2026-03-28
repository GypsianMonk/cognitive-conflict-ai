"""
Debate Manager — Top-level orchestrator for the full Cognitive Conflict pipeline.
Combines query understanding, agent selection, conflict engine, scoring, and judging.
"""

from typing import Optional
from dataclasses import dataclass

from engine.query_understanding import QueryUnderstanding, QueryAnalysis
from engine.conflict_engine import ConflictEngine, DebateResult
from agents.optimist_agent import OptimistAgent
from agents.skeptic_agent import SkepticAgent
from agents.alternative_agent import AlternativeAgent, ExpertAgent
from scoring.scorer import Scorer
from scoring.metrics import ScoredResponse


@dataclass
class FinalOutput:
    query: str
    final_answer: str
    reasoning_trace: str
    confidence_score: float          # 0–100 percentage
    convergence_achieved: bool
    total_debate_rounds: int
    agent_responses: list[dict]
    top_scored: list[dict]
    contradictions: list[str]
    domain: Optional[str]
    complexity: str
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "reasoning_trace": self.reasoning_trace,
            "confidence_score": f"{self.confidence_score:.1f}%",
            "convergence_achieved": self.convergence_achieved,
            "total_debate_rounds": self.total_debate_rounds,
            "domain": self.domain,
            "complexity": self.complexity,
            "contradictions": self.contradictions,
            "agent_responses": self.agent_responses,
            "top_scored": self.top_scored,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class DebateManager:
    """
    The primary entry point for the Cognitive Conflict AI system.
    Call .run(query) to get a validated, conflict-resolved answer.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config.get("model", {})
        self.query_analyzer = QueryUnderstanding()
        self.scorer = Scorer(config.get("scoring", {}))

    def run(self, query: str, retrieval_context: Optional[str] = None) -> FinalOutput:
        # Step 1: Understand the query
        analysis: QueryAnalysis = self.query_analyzer.analyze(query)

        # Step 2: Build agents
        agents = self._build_agents(analysis)

        # Step 3: Run the conflict engine (debate)
        engine = ConflictEngine(max_rounds=analysis.debate_rounds)
        debate: DebateResult = engine.run_debate(query, agents, retrieval_context)

        # Step 4: Score all final responses
        scored = self.scorer.score_all(debate.final_responses)

        # Step 5: Judge — produce final answer
        final_answer, reasoning_trace = self._judge(query, scored, debate)

        # Step 6: Estimate confidence
        confidence = self._estimate_confidence(scored, debate)

        return FinalOutput(
            query=query,
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            confidence_score=confidence,
            convergence_achieved=debate.convergence_achieved,
            total_debate_rounds=debate.total_rounds,
            agent_responses=[r.to_dict() for r in debate.final_responses],
            top_scored=[s.to_dict() for s in scored[:2]],
            contradictions=debate.contradiction_summary,
            domain=analysis.domain,
            complexity=analysis.complexity,
            duration_seconds=debate.duration_seconds,
        )

    def _build_agents(self, analysis: QueryAnalysis) -> list:
        agents = [
            OptimistAgent(self.model_config),
            SkepticAgent(self.model_config),
            AlternativeAgent(self.model_config),
        ]
        if analysis.requires_expert and analysis.domain:
            agents.append(ExpertAgent(domain=analysis.domain, model_config=self.model_config))
        return agents

    def _judge(self, query: str, scored: list[ScoredResponse], debate: DebateResult) -> tuple[str, str]:
        """
        Produce the final answer by synthesizing top-scored perspectives.
        In production, this should call the LLM with a synthesis prompt.
        """
        if not scored:
            return "Unable to produce an answer.", "No agent responses were scored."

        best = scored[0]
        reasoning_parts = []

        reasoning_parts.append(f"Query analyzed. Complexity: detected from content.")
        reasoning_parts.append(f"Ran {debate.total_rounds} debate round(s) with {len(debate.final_responses)} agents.")

        if debate.contradiction_summary:
            reasoning_parts.append(f"Contradictions detected: {'; '.join(debate.contradiction_summary)}")
        else:
            reasoning_parts.append("No significant contradictions detected between agents.")

        reasoning_parts.append(
            f"Top-scoring response from {best.agent_name} (score: {best.total_score:.3f}). "
            f"Reasoning: {best.reasoning}"
        )

        if len(scored) > 1:
            runner_up = scored[1]
            reasoning_parts.append(
                f"Runner-up: {runner_up.agent_name} (score: {runner_up.total_score:.3f})."
            )

        reasoning_trace = " | ".join(reasoning_parts)
        return best.answer, reasoning_trace

    def _estimate_confidence(self, scored: list[ScoredResponse], debate: DebateResult) -> float:
        """
        Estimate system-level confidence as a percentage (0–100).
        Based on: top score magnitude, convergence, and contradiction count.
        """
        if not scored:
            return 0.0

        base = scored[0].total_score * 100
        convergence_bonus = 10 if debate.convergence_achieved else 0
        contradiction_penalty = len(debate.contradiction_summary) * 5

        confidence = base + convergence_bonus - contradiction_penalty
        return round(max(0.0, min(100.0, confidence)), 1)
