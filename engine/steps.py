"""
Pipeline Steps — each step is a focused, testable unit.
Import and compose these in DebateManager.
"""
from __future__ import annotations
import logging
import time
from typing import Optional

from engine.pipeline import PipelineState

logger = logging.getLogger(__name__)


class SafetyStep:
    name = "safety"

    def __init__(self, checker):
        self.checker = checker

    def run(self, state: PipelineState) -> PipelineState:
        from engine.audit_logger import get_logger
        result = self.checker.check_input(state.query)
        state.safety_result = result
        state.audit_query_id = get_logger().new_query(state.query)
        if not result.safe:
            state.blocked = True
            state.block_reason = "; ".join(result.issues)
            state.final_answer = "Query blocked by safety filter."
            state.reasoning_trace = state.block_reason
        else:
            state.query = result.sanitized_input
        return state


class QueryUnderstandingStep:
    name = "query_understanding"

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def run(self, state: PipelineState) -> PipelineState:
        state.analysis = self.analyzer.analyze(state.query)
        return state


class KnowledgeGraphStep:
    name = "knowledge_graph"

    def __init__(self, kg):
        self.kg = kg

    def run(self, state: PipelineState) -> PipelineState:
        if not self.kg:
            return state
        ctx = self.kg.build_context(state.query)
        if ctx:
            state.add_context(ctx)
            state.kg_context_used = True
        return state


class TaskDecompositionStep:
    name = "task_decomposition"

    def __init__(self, decomposer):
        self.decomposer = decomposer

    def run(self, state: PipelineState) -> PipelineState:
        if not self.decomposer.should_decompose(state.query):
            return state
        plan = self.decomposer.decompose(state.query)
        state.task_decomposed = True
        summary = "\n".join(f"  {t.id}: {t.description}" for t in plan.subtasks)
        state.add_context(f"Task decomposed into {len(plan.subtasks)} subtasks:\n{summary}")
        return state


class TreeOfThoughtsStep:
    name = "tree_of_thoughts"

    def __init__(self, tot):
        self.tot = tot

    def run(self, state: PipelineState) -> PipelineState:
        reasoning, _ = self.tot.run(state.query)
        if reasoning:
            state.tot_reasoning = reasoning
            state.add_context(f"Pre-reasoning path: {reasoning}")
        return state


class BuildAgentsStep:
    name = "build_agents"

    def __init__(self, agent_factory):
        self.factory = agent_factory

    def run(self, state: PipelineState) -> PipelineState:
        state.agents = self.factory(state.analysis)
        return state


class DebateStep:
    name = "debate"

    def __init__(self, blind_round, max_rounds_override: Optional[int] = None):
        self.blind_round = blind_round
        self.max_rounds_override = max_rounds_override

    def run(self, state: PipelineState) -> PipelineState:
        from engine.working_memory import WorkingMemory
        from engine.audit_logger import get_logger

        wm = WorkingMemory()
        state.working_memory = wm
        audit = get_logger()

        max_rounds = self.max_rounds_override or (
            state.analysis.debate_rounds if state.analysis else 2
        )
        context_for_round: Optional[str] = state.context or None
        variance = 0.0

        for round_num in range(1, max_rounds + 1):
            wm.set_round(round_num)
            result = self.blind_round.run_blind_round(
                state.query, state.agents, round_num, prior_context=context_for_round
            )
            state.round_results.append(result)
            state.responses = result.responses
            state.rounds_run = round_num

            for r in result.responses:
                wm.add_premise(r.answer[:100], r.agent_name, r.confidence)
                audit.log_agent_response(r.agent_name, r.answer, r.reasoning, r.confidence)

            confidences = [r.confidence for r in result.responses]
            if len(confidences) >= 2:
                mean = sum(confidences) / len(confidences)
                variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
                state.convergence_achieved = variance < 0.04

            audit.log_debate_round(round_num, len(state.agents), [], round(1.0 - variance, 3))
            context_for_round = self.blind_round.build_context_from_round(result)

            if state.convergence_achieved and round_num > 1:
                break

        return state


class ReflectionStep:
    name = "reflection"

    def run(self, state: PipelineState) -> PipelineState:
        if not state.config.agents.enable_reflection:
            return state
        from agents.reflective_agent import ReflectiveAgent
        threshold = state.config.agents.reflection_threshold

        for i, agent in enumerate(state.agents):
            if i >= len(state.responses):
                break
            # Only reflect if confidence is below threshold
            if state.responses[i].confidence >= threshold:
                continue
            try:
                reflective = ReflectiveAgent(agent)
                refined, result = reflective.generate_with_reflection(
                    state.query, state.context or None
                )
                if result.improved:
                    state.responses[i] = refined
                    state.reflection_count += 1
            except Exception as e:
                logger.warning("Reflection failed for %s: %s", agent.name, e)
        return state


class OutputSafetyStep:
    name = "output_safety"

    def __init__(self, checker):
        self.checker = checker

    def run(self, state: PipelineState) -> PipelineState:
        safe = [r for r in state.responses
                if self.checker.check_output(r.answer, r.agent_name).safe]
        state.responses = safe if safe else state.responses
        return state


class HallucinationStep:
    name = "hallucination"

    def __init__(self, detector):
        self.detector = detector

    def run(self, state: PipelineState) -> PipelineState:
        from engine.audit_logger import get_logger
        state.hal_report = self.detector.analyze(state.responses)
        get_logger().log_hallucination_check(
            state.hal_report.overall_risk,
            len(state.hal_report.flags),
            state.hal_report.to_dict().get("flags", []),
        )
        return state


class FactCheckStep:
    name = "fact_check"

    def __init__(self, checker):
        self.checker = checker

    def run(self, state: PipelineState) -> PipelineState:
        try:
            state.fact_report = self.checker.verify_responses(state.responses)
        except Exception as e:
            logger.warning("Fact check failed: %s", e)
        return state


class UncertaintyStep:
    name = "uncertainty"

    def __init__(self, decomposer):
        self.decomposer = decomposer

    def run(self, state: PipelineState) -> PipelineState:
        state.uncertainty = self.decomposer.decompose(state.responses)
        return state


class ScoringStep:
    name = "scoring"

    def __init__(self, scorer_factory):
        self.scorer_factory = scorer_factory

    def run(self, state: PipelineState) -> PipelineState:
        scorer = self.scorer_factory()
        state.scored = scorer.score_all(state.responses)
        return state


class GraphOfThoughtsStep:
    name = "graph_of_thoughts"

    def run(self, state: PipelineState) -> PipelineState:
        from engine.got_engine import GraphOfThoughts
        got = GraphOfThoughts()
        got.build_from_agent_responses(state.responses)
        state.got_summary = got.summary()
        return state


class JudgeStep:
    name = "judge"

    def run(self, state: PipelineState) -> PipelineState:
        from engine.audit_logger import get_logger
        if not state.scored:
            state.final_answer = "No scored responses available."
            state.reasoning_trace = "Scoring produced no results."
            return state

        best = state.scored[0]
        parts = [f"Selected {best.agent_name} (score={best.total_score:.3f})."]
        if len(state.scored) > 1:
            parts.append(f"Runner-up: {state.scored[1].agent_name} (score={state.scored[1].total_score:.3f}).")
        if state.rounds_run > 1:
            parts.append(f"Ran {state.rounds_run} debate rounds.")
        if state.convergence_achieved:
            parts.append("Agents converged.")
        if state.tot_reasoning:
            parts.append(f"ToT pre-reasoning: {state.tot_reasoning[:80]}.")

        state.final_answer = best.answer
        state.reasoning_trace = " | ".join(parts)

        # Confidence
        hal_risk = state.hal_report.overall_risk if state.hal_report else 0.0
        credibility = state.fact_report.overall_credibility if state.fact_report else 1.0
        base = best.total_score * 100
        base += 10 if state.convergence_achieved else 0
        base -= hal_risk * 20
        base += (credibility - 0.5) * 10
        state.confidence = round(max(0.0, min(100.0, base)), 1)

        get_logger().log_final_output(
            state.final_answer, state.confidence,
            state.analysis.domain if state.analysis else None,
            state.analysis.complexity if state.analysis else "unknown",
        )
        return state


class PersonaEvolutionStep:
    name = "persona_evolution"

    def __init__(self, evolver):
        self.evolver = evolver

    def run(self, state: PipelineState) -> PipelineState:
        if not state.scored:
            return state
        from agents.persona_evolver import AgentPerformanceRecord
        winner_name = state.scored[0].agent_name
        hal_risk = state.hal_report.overall_risk if state.hal_report else 0.0
        domain = state.analysis.domain if state.analysis else "general"
        for r in state.responses:
            try:
                self.evolver.record_outcome(AgentPerformanceRecord(
                    agent_name=r.agent_name,
                    query_domain=domain,
                    won=(r.agent_name == winner_name),
                    confidence_accuracy=abs(r.confidence - state.confidence / 100.0),
                    contradiction_rate=hal_risk,
                ))
            except Exception as e:
                logger.warning("Persona evolution failed: %s", e)
        return state


class PersistStep:
    name = "persist"

    def __init__(self, memory, kg):
        self.memory = memory
        self.kg = kg

    def run(self, state: PipelineState) -> PipelineState:
        from scoring.metrics import record_output
        output = _state_to_output(state)
        if self.memory:
            try:
                self.memory.save(output)
            except Exception as e:
                logger.warning("Memory save failed: %s", e)
        if self.kg:
            try:
                self.kg.ingest_from_output(output, state.query)
            except Exception as e:
                logger.warning("KG ingest failed: %s", e)
        try:
            record_output(output)
        except Exception as e:
            logger.warning("Metrics record failed: %s", e)
        return state


def _state_to_output(state: PipelineState):
    """Convert PipelineState → FinalOutput for persistence."""
    from engine.debate_manager import FinalOutput
    return FinalOutput(
        query=state.query,
        final_answer=state.final_answer,
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
