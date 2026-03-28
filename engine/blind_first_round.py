"""
Anti-Herding Protocol — Blind First Round with Concurrent Execution.
Round 1: All agents generate independently in parallel (ThreadPoolExecutor).
Round 2+: Agents see prior round context and refine.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Optional
import logging

from agents.base_agent import AgentResponse

logger = logging.getLogger(__name__)

DEFAULT_AGENT_TIMEOUT = 90   # seconds per agent


@dataclass
class BlindRoundResult:
    round_number: int
    blind: bool
    responses: list[AgentResponse]
    diversity_score: float
    failed_agents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "blind": self.blind,
            "diversity_score": round(self.diversity_score, 3),
            "agent_count": len(self.responses),
            "failed_agents": self.failed_agents,
        }


class BlindFirstRound:
    def __init__(self, max_workers: int = 4, agent_timeout: int = DEFAULT_AGENT_TIMEOUT):
        self.max_workers = max_workers
        self.agent_timeout = agent_timeout

    def run_blind_round(
        self,
        query: str,
        agents: list,
        round_number: int,
        prior_context: Optional[str] = None,
    ) -> BlindRoundResult:
        is_blind = round_number == 1
        context = None if is_blind else prior_context

        responses: list[AgentResponse] = []
        failed: list[str] = []

        # Run all agents concurrently
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(agents))) as pool:
            future_to_agent = {
                pool.submit(self._safe_generate, agent, query, context): agent
                for agent in agents
            }
            for future in as_completed(future_to_agent, timeout=self.agent_timeout + 5):
                agent = future_to_agent[future]
                try:
                    result = future.result(timeout=self.agent_timeout)
                    responses.append(result)
                except FuturesTimeout:
                    logger.warning("Agent %s timed out", agent.name)
                    failed.append(agent.name)
                except Exception as e:
                    logger.error("Agent %s raised: %s", agent.name, e)
                    failed.append(agent.name)

        if not responses:
            logger.error("All agents failed in round %d", round_number)

        return BlindRoundResult(
            round_number=round_number,
            blind=is_blind,
            responses=responses,
            diversity_score=self._diversity(responses),
            failed_agents=failed,
        )

    def _safe_generate(self, agent, query: str, context: Optional[str]) -> AgentResponse:
        return agent.generate(query, context=context)

    def _diversity(self, responses: list[AgentResponse]) -> float:
        if len(responses) < 2:
            return 0.0
        confidences = [r.confidence for r in responses]
        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
        lengths = [len(r.answer) for r in responses]
        len_mean = sum(lengths) / len(lengths)
        len_var = sum((l - len_mean) ** 2 for l in lengths) / len(lengths)
        len_var_norm = min(1.0, len_var / 10000)
        return round(min(1.0, variance * 4 * 0.4 + len_var_norm * 0.6), 3)

    def build_context_from_round(self, result: BlindRoundResult) -> str:
        parts = ["Previous round responses — use these to refine your answer:"]
        for r in result.responses:
            parts.append(f"\n[{r.agent_name}]\nAnswer: {r.answer}\nReasoning: {r.reasoning}")
        return "\n".join(parts)
