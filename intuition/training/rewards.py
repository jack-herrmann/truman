"""Reward functions for personality training."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import numpy as np
from intuition.core.traces import BehavioralTrace
from intuition.llm.client import LLMClient
from intuition.llm.embeddings import EmbeddingClient
from intuition.training.discriminator import TraceDiscriminator
if TYPE_CHECKING:
    from intuition.agent.agent import PersonalityAgent
logger = logging.getLogger(__name__)


class CoherenceReward:
    def __init__(self, embedding_client: EmbeddingClient, llm: LLMClient | None = None) -> None:
        self.discriminator = TraceDiscriminator(embedding_client)
        self.llm = llm

    async def compute(self, agent: "PersonalityAgent", traces: list[BehavioralTrace]) -> float:
        if len(traces) < 2:
            return 0.5
        embedding_score = await self.discriminator.compute_within_agent_variance(traces)
        narrative_score = 1.0
        if self.llm and len(traces) >= 4:
            narrative_score = await self._narrative_consistency(traces)
        return 0.6 * embedding_score + 0.4 * narrative_score if self.llm else embedding_score

    async def _narrative_consistency(self, traces: list[BehavioralTrace]) -> float:
        early = "\n".join(f"Situation: {t.situation[:100]}\nAction: {t.action[:150]}" for t in traces[:3])
        late = f"Situation: {traces[-1].situation[:100]}\nAction: {traces[-1].action[:150]}"
        prompt = (f"Early behaviour:\n{early}\n\nLater behaviour:\n{late}\n\n"
                  "On 0-10, how much does later behaviour feel like the SAME PERSON? Number only.")
        try:
            r = await self.llm.generate(system="Personality consistency evaluator.",
                                         messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=10)
            return max(0.0, min(1.0, float(r.strip().split()[0]) / 10.0))
        except (ValueError, IndexError):
            return 0.5


class IndividualityReward:
    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.discriminator = TraceDiscriminator(embedding_client)

    async def compute(self, agents: list["PersonalityAgent"], traces_per_agent: list[list[BehavioralTrace]]) -> float:
        if len(agents) < 2:
            return 0.0
        _, individuality = await self.discriminator.compute_scores(traces_per_agent)
        return individuality

    async def compute_per_agent(self, traces_per_agent: list[list[BehavioralTrace]]) -> list[float]:
        if len(traces_per_agent) < 2:
            return [0.0] * len(traces_per_agent)
        all_texts, agent_indices = [], []
        for i, traces in enumerate(traces_per_agent):
            for t in traces:
                all_texts.append(t.full_text)
                agent_indices.append(i)
        if not all_texts:
            return [0.0] * len(traces_per_agent)
        embeddings = await self.discriminator.embeddings.embed_batch(all_texts)
        n = len(traces_per_agent)
        arr = np.array(agent_indices)
        centroids = np.zeros((n, embeddings.shape[1]))
        for i in range(n):
            mask = arr == i
            if mask.any():
                centroids[i] = embeddings[mask].mean(axis=0)
        scores = []
        for i in range(n):
            dists = [float(np.linalg.norm(centroids[i] - centroids[j])) for j in range(n) if j != i]
            scores.append(float(2.0 / (1.0 + np.exp(-np.mean(dists))) - 1.0) if dists else 0.0)
        return scores
