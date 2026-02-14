"""IndividualityEvaluator — between-agent distinctness."""
from __future__ import annotations
import logging
import numpy as np
from intuition.agent.agent import PersonalityAgent
from intuition.core.traces import BehavioralTrace
from intuition.llm.client import LLMClient
from intuition.llm.embeddings import EmbeddingClient
from intuition.training.discriminator import TraceDiscriminator
logger = logging.getLogger(__name__)


class IndividualityReport:
    def __init__(self, agent_names, discriminability, mean_pairwise_distance,
                 qualitative_score, pairwise_matrix):
        self.agent_names = agent_names
        self.discriminability = discriminability
        self.mean_pairwise_distance = mean_pairwise_distance
        self.qualitative_score = qualitative_score
        self.pairwise_matrix = pairwise_matrix

    @property
    def overall_score(self):
        return 0.4*self.discriminability + 0.3*self.mean_pairwise_distance + 0.3*self.qualitative_score

    def summary(self):
        return (f"Individuality ({len(self.agent_names)} agents): overall={self.overall_score:.3f} "
                f"discrim={self.discriminability:.3f} dist={self.mean_pairwise_distance:.3f} "
                f"qual={self.qualitative_score:.3f}")


class IndividualityEvaluator:
    def __init__(self, llm: LLMClient, embedding_client: EmbeddingClient):
        self.llm = llm
        self.embedding_client = embedding_client
        self.discriminator = TraceDiscriminator(embedding_client)

    async def evaluate(self, agents: list[PersonalityAgent],
                       traces_per_agent: list[list[BehavioralTrace]] | None = None):
        if traces_per_agent is None:
            traces_per_agent = [a.traces for a in agents]
        names = [a.kernel.name for a in agents]
        _, discriminability = await self.discriminator.compute_scores(traces_per_agent)
        matrix, mean_dist = await self._pairwise_distances(traces_per_agent)
        qualitative = await self._qualitative_check(agents, traces_per_agent)
        return IndividualityReport(names, discriminability, mean_dist, qualitative, matrix)

    async def _pairwise_distances(self, traces_per_agent):
        n = len(traces_per_agent)
        centroids = []
        for traces in traces_per_agent:
            if not traces:
                centroids.append(np.zeros(self.embedding_client.dimension))
                continue
            embs = await self.embedding_client.embed_batch([t.full_text for t in traces])
            centroids.append(embs.mean(axis=0))
        matrix = [[0.0]*n for _ in range(n)]
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                d = float(np.linalg.norm(centroids[i]-centroids[j]))
                matrix[i][j] = matrix[j][i] = d
                dists.append(d)
        mean = float(np.mean(dists)) if dists else 0.0
        return matrix, float(2.0/(1.0+np.exp(-mean))-1.0)

    async def _qualitative_check(self, agents, traces_per_agent):
        if len(agents) < 2 or not all(traces_per_agent):
            return 0.5
        samples = [f"Person {i+1}: {t[0].action[:200]}" for i, t in enumerate(traces_per_agent) if t]
        if len(samples) < 2:
            return 0.5
        prompt = ("Responses from different people to same situation:\n" + "\n".join(samples) +
                  "\n\nOn 0-10, how DISTINCT are they? Number only.")
        try:
            r = await self.llm.generate(system="Personality distinctness evaluator.",
                messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=10)
            return max(0.0, min(1.0, float(r.strip().split()[0])/10.0))
        except (ValueError, IndexError):
            return 0.5
