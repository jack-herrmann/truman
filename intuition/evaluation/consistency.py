"""ConsistencyEvaluator — coherence measurement."""
from __future__ import annotations
import logging
import numpy as np
from intuition.agent.agent import PersonalityAgent
from intuition.evaluation.probes import ProbeBattery, ProbeResult
from intuition.llm.client import LLMClient
from intuition.llm.embeddings import EmbeddingClient
from intuition.training.discriminator import TraceDiscriminator
logger = logging.getLogger(__name__)


class ConsistencyReport:
    def __init__(self, agent_name, probe_stability, embedding_consistency, narrative_consistency, probe_details):
        self.agent_name = agent_name
        self.probe_stability = probe_stability
        self.embedding_consistency = embedding_consistency
        self.narrative_consistency = narrative_consistency
        self.probe_details = probe_details

    @property
    def overall_score(self):
        return 0.4 * self.probe_stability + 0.35 * self.embedding_consistency + 0.25 * self.narrative_consistency

    def summary(self):
        return (f"Consistency: {self.agent_name} overall={self.overall_score:.3f} "
                f"probe={self.probe_stability:.3f} embed={self.embedding_consistency:.3f} "
                f"narr={self.narrative_consistency:.3f}")


class ConsistencyEvaluator:
    def __init__(self, llm: LLMClient, embedding_client: EmbeddingClient,
                 probe_battery: ProbeBattery | None = None, probe_repetitions: int = 3):
        self.llm = llm
        self.embedding_client = embedding_client
        self.probe_battery = probe_battery or ProbeBattery(llm)
        self.probe_repetitions = probe_repetitions
        self.discriminator = TraceDiscriminator(embedding_client)

    async def evaluate(self, agent: PersonalityAgent) -> ConsistencyReport:
        from intuition.agent.prompt_builder import PromptBuilder
        system_prompt = PromptBuilder().build(agent.kernel)
        probe_results = await self.probe_battery.full_assessment(system_prompt, self.probe_repetitions)
        probe_stability, probe_details = self._measure_probe_stability(probe_results)
        embedding_consistency = (await self.discriminator.compute_within_agent_variance(agent.traces)
                                 if len(agent.traces) >= 2 else 1.0)
        narrative_consistency = 1.0
        if len(agent.traces) >= 4:
            from intuition.training.rewards import CoherenceReward
            narrative_consistency = await CoherenceReward(self.embedding_client, self.llm)._narrative_consistency(agent.traces)
        return ConsistencyReport(agent.kernel.name, probe_stability, embedding_consistency,
                                 narrative_consistency, probe_details)

    @staticmethod
    def _measure_probe_stability(probe_results):
        details, all_stabilities = {}, []
        for name, results in probe_results.items():
            if len(results) < 2:
                continue
            dim_across: dict[str, list[float]] = {}
            for r in results:
                for d, s in r.dimension_scores.items():
                    dim_across.setdefault(d, []).append(s)
            dim_stab = []
            for d, scores in dim_across.items():
                std = float(np.std(scores))
                stability = 1.0 / (1.0 + std * 5)
                dim_stab.append({"dim": d, "std": std, "stability": stability})
                all_stabilities.append(stability)
            details[name] = dim_stab
        return (float(np.mean(all_stabilities)) if all_stabilities else 0.5), details
