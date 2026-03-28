"""
Debate Manager v3 — Pipeline-Based Orchestrator
Replaces the 160-line God method with a composable pipeline.
Each feature is a discrete, testable step.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from config import AppConfig
from engine.pipeline import Pipeline, PipelineState
from engine.steps import (
    SafetyStep, QueryUnderstandingStep, KnowledgeGraphStep,
    TaskDecompositionStep, TreeOfThoughtsStep, BuildAgentsStep,
    DebateStep, ReflectionStep, OutputSafetyStep, HallucinationStep,
    FactCheckStep, UncertaintyStep, ScoringStep, GraphOfThoughtsStep,
    JudgeStep, PersonaEvolutionStep, PersistStep,
)
from engine.query_understanding import QueryUnderstanding
from engine.blind_first_round import BlindFirstRound
from engine.audit_logger import get_logger
from engine.safety_checker import SafetyChecker
from engine.task_decomposer import TaskDecomposer
from engine.tot_engine import TreeOfThoughts

from agents.optimist_agent import OptimistAgent
from agents.skeptic_agent import SkepticAgent
from agents.alternative_agent import AlternativeAgent, ExpertAgent
from agents.fact_checker import FactChecker
from agents.persona_evolver import PersonaEvolver

from scoring.scorer import Scorer
from scoring.hallucination_detector import HallucinationDetector
from scoring.uncertainty import UncertaintyDecomposer
from scoring.preference_model import PreferenceModel

from memory.memory_store import MemoryStore
from memory.knowledge_graph import KnowledgeGraph


@dataclass
class FinalOutput:
    query: str
    final_answer: str
    reasoning_trace: str
    confidence_score: float
    convergence_achieved: bool
    total_debate_rounds: int
    agent_responses: list
    top_scored: list
    contradictions: list
    domain: Optional[str]
    complexity: str
    duration_seconds: float
    hallucination_risk: float = 0.0
    uncertainty: dict = field(default_factory=dict)
    fact_check: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    working_memory_summary: dict = field(default_factory=dict)
    kg_context_used: bool = False
    task_decomposed: bool = False
    reflection_count: int = 0
    audit_query_id: str = ""
    got_summary: dict = field(default_factory=dict)
    pipeline_errors: list = field(default_factory=list)

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
            "hallucination_risk": round(self.hallucination_risk, 3),
            "uncertainty": self.uncertainty,
            "fact_check": self.fact_check,
            "safety": self.safety,
            "working_memory_summary": self.working_memory_summary,
            "kg_context_used": self.kg_context_used,
            "task_decomposed": self.task_decomposed,
            "reflection_count": self.reflection_count,
            "audit_query_id": self.audit_query_id,
            "got_summary": self.got_summary,
            "pipeline_errors": self.pipeline_errors,
        }


class DebateManager:
    """
    Assembles and runs the full Cognitive Conflict AI pipeline.
    Accepts AppConfig or a legacy dict for backward compatibility.
    """

    def __init__(self, config):
        if isinstance(config, dict):
            self._cfg = AppConfig.from_dict(config)
        else:
            self._cfg = config

        mc = self._cfg.model.__dict__

        # Shared services
        self._safety = SafetyChecker()
        self._analyzer = QueryUnderstanding()
        self._blind_round = BlindFirstRound(max_workers=4)
        self._hal_detector = HallucinationDetector()
        self._uncertainty = UncertaintyDecomposer()
        self._fact_checker = FactChecker()
        self._task_decomposer = TaskDecomposer()
        self._persona_evolver = PersonaEvolver()
        self._preference_model = PreferenceModel()
        self._tot = TreeOfThoughts(
            max_depth=2, branching_factor=3, beam_width=2,
        )

        memory_on = self._cfg.memory.enabled
        self._memory = MemoryStore() if memory_on else None
        self._kg = KnowledgeGraph() if (memory_on and self._cfg.memory.kg_enabled) else None

        self._pipeline = self._build_pipeline(mc)

    def _build_pipeline(self, mc: dict) -> Pipeline:
        def agent_factory(analysis):
            agents = [
                OptimistAgent(mc),
                SkepticAgent(mc),
                AlternativeAgent(mc),
            ]
            if analysis and analysis.requires_expert and analysis.domain:
                agents.append(ExpertAgent(domain=analysis.domain, model_config=mc))
            return agents

        def scorer_factory():
            learned = self._preference_model.get_weights()
            return Scorer({
                "relevance_weight": learned.relevance_weight,
                "coherence_weight": learned.coherence_weight,
                "contradiction_penalty": learned.contradiction_penalty,
            })

        return Pipeline([
            SafetyStep(self._safety),
            QueryUnderstandingStep(self._analyzer),
            KnowledgeGraphStep(self._kg),
            TaskDecompositionStep(self._task_decomposer),
            TreeOfThoughtsStep(self._tot),
            BuildAgentsStep(agent_factory),
            DebateStep(self._blind_round),
            ReflectionStep(),
            OutputSafetyStep(self._safety),
            HallucinationStep(self._hal_detector),
            FactCheckStep(self._fact_checker),
            UncertaintyStep(self._uncertainty),
            ScoringStep(scorer_factory),
            GraphOfThoughtsStep(),
            JudgeStep(),
            PersonaEvolutionStep(self._persona_evolver),
            PersistStep(self._memory, self._kg),
        ])

    def run(self, query: str, retrieval_context: Optional[str] = None) -> FinalOutput:
        state = PipelineState(query=query, config=self._cfg)
        if retrieval_context:
            state.add_context(retrieval_context)

        state = self._pipeline.run(state)
        return self._state_to_output(state)

    def _state_to_output(self, state: PipelineState) -> FinalOutput:
        return FinalOutput(
            query=state.query,
            final_answer=state.final_answer or "No answer generated.",
            reasoning_trace=state.reasoning_trace,
            confidence_score=state.confidence,
            convergence_achieved=state.convergence_achieved,
            total_debate_rounds=state.rounds_run,
            agent_responses=[r.to_dict() for r in state.responses],
            top_scored=[s.to_dict() for s in state.scored[:2]],
            contradictions=[
                f.description for f in (state.hal_report.flags if state.hal_report else [])
                if f.severity in ("high", "critical")
            ],
            domain=state.analysis.domain if state.analysis else None,
            complexity=state.analysis.complexity if state.analysis else "unknown",
            duration_seconds=state.elapsed(),
            hallucination_risk=state.hal_report.overall_risk if state.hal_report else 0.0,
            uncertainty=state.uncertainty.to_dict() if state.uncertainty else {},
            fact_check=state.fact_report.to_dict() if state.fact_report else {},
            safety=state.safety_result.to_dict() if state.safety_result else {},
            working_memory_summary=state.working_memory.summary() if state.working_memory else {},
            kg_context_used=state.kg_context_used,
            task_decomposed=state.task_decomposed,
            reflection_count=state.reflection_count,
            audit_query_id=state.audit_query_id,
            got_summary=state.got_summary,
            pipeline_errors=state.errors,
        )
