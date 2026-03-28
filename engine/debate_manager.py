"""
Debate Manager v2 — Full Production Orchestrator
Integrates ALL features:
  - Anti-herding blind first round
  - Tree of Thoughts + Graph of Thoughts
  - Self-reflection loop
  - Working memory
  - Tool-using agents + Fact checking
  - Hallucination detection
  - Uncertainty decomposition
  - Audit logging + Safety checking
  - Task decomposition
  - Persona evolution
  - Knowledge graph
  - Preference-driven scoring
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

from engine.query_understanding import QueryUnderstanding, QueryAnalysis
from engine.blind_first_round import BlindFirstRound
from engine.working_memory import WorkingMemory
from engine.audit_logger import get_logger
from engine.safety_checker import SafetyChecker
from engine.task_decomposer import TaskDecomposer
from engine.tot_engine import TreeOfThoughts

from agents.optimist_agent import OptimistAgent
from agents.skeptic_agent import SkepticAgent
from agents.alternative_agent import AlternativeAgent, ExpertAgent
from agents.reflective_agent import ReflectiveAgent
from agents.fact_checker import FactChecker
from agents.persona_evolver import PersonaEvolver, AgentPerformanceRecord

from scoring.scorer import Scorer
from scoring.metrics import record_output
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
        }


class DebateManager:
    """
    Production-grade orchestrator. Call .run(query) to get a
    safety-checked, fact-verified, conflict-resolved answer.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config.get("model", {})
        self.memory_enabled = config.get("memory", {}).get("enabled", True)

        self.query_analyzer = QueryUnderstanding()
        self.scorer = Scorer(config.get("scoring", {}))
        self.safety = SafetyChecker()
        self.blind_round = BlindFirstRound()
        self.hal_detector = HallucinationDetector()
        self.uncertainty_decomposer = UncertaintyDecomposer()
        self.fact_checker = FactChecker()
        self.task_decomposer = TaskDecomposer()
        self.persona_evolver = PersonaEvolver()
        self.preference_model = PreferenceModel()
        self.tot = TreeOfThoughts(max_depth=2, branching_factor=2, beam_width=2)
        self.logger = get_logger()

        if self.memory_enabled:
            self.memory = MemoryStore()
            self.kg = KnowledgeGraph()
        else:
            self.memory = None
            self.kg = None

    def run(self, query: str, retrieval_context: Optional[str] = None) -> FinalOutput:
        start = time.time()
        audit_id = self.logger.new_query(query)

        # Step 0: Safety check
        safety_result = self.safety.check_input(query)
        if not safety_result.safe:
            return self._blocked_output(query, safety_result, audit_id, time.time() - start)
        query = safety_result.sanitized_input

        # Step 1: Query understanding
        analysis: QueryAnalysis = self.query_analyzer.analyze(query)

        # Step 2: Knowledge graph context
        kg_context, kg_used = "", False
        if self.kg:
            kg_context = self.kg.build_context(query)
            kg_used = bool(kg_context)

        context = "\n\n".join(filter(None, [retrieval_context, kg_context]))

        # Step 3: Task decomposition
        task_decomposed = False
        if self.task_decomposer.should_decompose(query):
            task_decomposed = True
            plan = self.task_decomposer.decompose(query)
            context = (
                f"Task decomposed into {len(plan.subtasks)} subtasks:\n"
                + "\n".join(f"  {t.id}: {t.description}" for t in plan.subtasks)
                + ("\n\n" + context if context else "")
            )

        # Step 4: Tree of Thoughts pre-reasoning
        tot_reasoning, _ = self.tot.run(query)
        if tot_reasoning:
            context = f"Pre-reasoning: {tot_reasoning}\n\n" + context

        # Step 5: Build agents
        agents = self._build_agents(analysis)
        working_mem = WorkingMemory()

        # Step 6: Multi-round debate with blind first round
        all_responses, rounds_run = [], 0
        context_for_round = context or None
        convergence_achieved = False
        variance = 0.0

        for round_num in range(1, analysis.debate_rounds + 1):
            working_mem.set_round(round_num)
            blind_result = self.blind_round.run_blind_round(
                query, agents, round_num, prior_context=context_for_round
            )
            all_responses = blind_result.responses
            rounds_run = round_num

            for r in all_responses:
                working_mem.add_premise(r.answer[:100], r.agent_name, r.confidence)
                self.logger.log_agent_response(r.agent_name, r.answer, r.reasoning, r.confidence)

            confidences = [r.confidence for r in all_responses]
            if len(confidences) >= 2:
                mean = sum(confidences) / len(confidences)
                variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
                convergence_achieved = variance < 0.04

            context_for_round = self.blind_round.build_context_from_round(blind_result)
            self.logger.log_debate_round(round_num, len(agents), [], round(1.0 - variance, 3))

            if convergence_achieved and round_num > 1:
                break

        # Step 7: Self-reflection
        reflection_count = 0
        if self.config.get("agents", {}).get("enable_reflection", True):
            for i, agent in enumerate(agents):
                reflective = ReflectiveAgent(agent)
                refined, ref_result = reflective.generate_with_reflection(query, context or None)
                if ref_result.improved:
                    all_responses[i] = refined
                    reflection_count += 1

        # Step 8: Safety check outputs
        safe_responses = [
            r for r in all_responses if self.safety.check_output(r.answer, r.agent_name).safe
        ] or all_responses

        # Step 9: Hallucination detection
        hal_report = self.hal_detector.analyze(safe_responses)
        self.logger.log_hallucination_check(
            hal_report.overall_risk, len(hal_report.flags),
            hal_report.to_dict().get("flags", [])
        )

        # Step 10: Fact checking
        fact_report = self.fact_checker.verify_responses(safe_responses)

        # Step 11: Uncertainty decomposition
        uncertainty = self.uncertainty_decomposer.decompose(safe_responses)

        # Step 12: Preference-weighted scoring
        learned = self.preference_model.get_weights()
        adaptive_scorer = Scorer({
            "relevance_weight": learned.relevance_weight,
            "coherence_weight": learned.coherence_weight,
            "contradiction_penalty": learned.contradiction_penalty,
        })
        scored = adaptive_scorer.score_all(safe_responses)

        # Step 13: Judge
        final_answer, reasoning_trace = self._judge(scored)

        # Step 14: Confidence
        confidence = self._estimate_confidence(scored, convergence_achieved, hal_report.overall_risk, fact_report.overall_credibility)

        # Step 15: Persona evolution
        if scored:
            winner_name = scored[0].agent_name
            for r in safe_responses:
                self.persona_evolver.record_outcome(AgentPerformanceRecord(
                    agent_name=r.agent_name,
                    query_domain=analysis.domain or "general",
                    won=(r.agent_name == winner_name),
                    confidence_accuracy=abs(r.confidence - confidence / 100.0),
                    contradiction_rate=hal_report.overall_risk,
                ))

        # Step 16: Persist
        output = FinalOutput(
            query=query,
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            confidence_score=confidence,
            convergence_achieved=convergence_achieved,
            total_debate_rounds=rounds_run,
            agent_responses=[r.to_dict() for r in safe_responses],
            top_scored=[s.to_dict() for s in scored[:2]],
            contradictions=[f.description for f in hal_report.flags if f.severity in ("high","critical")],
            domain=analysis.domain,
            complexity=analysis.complexity,
            duration_seconds=time.time() - start,
            hallucination_risk=hal_report.overall_risk,
            uncertainty=uncertainty.to_dict(),
            fact_check=fact_report.to_dict(),
            safety=safety_result.to_dict(),
            working_memory_summary=working_mem.summary(),
            kg_context_used=kg_used,
            task_decomposed=task_decomposed,
            reflection_count=reflection_count,
            audit_query_id=audit_id,
        )

        if self.memory:
            self.memory.save(output)
        if self.kg:
            self.kg.ingest_from_output(output, query)

        record_output(output)
        self.logger.log_final_output(final_answer, confidence, analysis.domain, analysis.complexity)
        return output

    def _build_agents(self, analysis: QueryAnalysis) -> list:
        agents = [
            OptimistAgent(self.model_config),
            SkepticAgent(self.model_config),
            AlternativeAgent(self.model_config),
        ]
        if analysis.requires_expert and analysis.domain:
            agents.append(ExpertAgent(domain=analysis.domain, model_config=self.model_config))
        return agents

    def _judge(self, scored) -> tuple[str, str]:
        if not scored:
            return "Unable to produce an answer.", "No scored responses."
        best = scored[0]
        parts = [f"Judge selected {best.agent_name} (score={best.total_score:.3f})."]
        if len(scored) > 1:
            parts.append(f"Runner-up: {scored[1].agent_name} (score={scored[1].total_score:.3f}).")
        return best.answer, " | ".join(parts)

    def _estimate_confidence(self, scored, convergence: bool, hal_risk: float, credibility: float) -> float:
        base = (scored[0].total_score * 100) if scored else 50.0
        base += 10 if convergence else 0
        base -= hal_risk * 20
        base += (credibility - 0.5) * 10
        return round(max(0.0, min(100.0, base)), 1)

    def _blocked_output(self, query, safety_result, audit_id, duration) -> FinalOutput:
        return FinalOutput(
            query=query,
            final_answer="This query was blocked by the safety filter.",
            reasoning_trace=f"Safety issues: {'; '.join(safety_result.issues)}",
            confidence_score=0.0,
            convergence_achieved=False,
            total_debate_rounds=0,
            agent_responses=[], top_scored=[], contradictions=[],
            domain=None, complexity="blocked",
            duration_seconds=duration,
            safety=safety_result.to_dict(),
            audit_query_id=audit_id,
        )
