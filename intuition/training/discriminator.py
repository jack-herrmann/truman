"""TraceDiscriminator — embedding-based behavioural trace classification."""
from __future__ import annotations
import numpy as np
from intuition.core.traces import BehavioralTrace
from intuition.llm.embeddings import EmbeddingClient


class TraceDiscriminator:
    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embeddings = embedding_client

    async def compute_scores(self, traces_per_agent: list[list[BehavioralTrace]]) -> tuple[float, float]:
        n_agents = len(traces_per_agent)
        if n_agents < 2:
            return 1.0, 0.0
        all_texts, labels = [], []
        for i, traces in enumerate(traces_per_agent):
            for t in traces:
                all_texts.append(t.full_text)
                labels.append(i)
        if not all_texts:
            return 0.0, 0.0
        embeddings = await self.embeddings.embed_batch(all_texts)
        labels_arr = np.array(labels)
        centroids = np.zeros((n_agents, embeddings.shape[1]))
        for i in range(n_agents):
            mask = labels_arr == i
            if mask.any():
                centroids[i] = embeddings[mask].mean(axis=0)
        correct = sum(1 for idx, emb in enumerate(embeddings)
                      if int(np.argmin(np.linalg.norm(centroids - emb[np.newaxis, :], axis=1))) == labels[idx])
        coherence = correct / len(embeddings)
        pairwise = [float(np.linalg.norm(centroids[i] - centroids[j]))
                     for i in range(n_agents) for j in range(i + 1, n_agents)]
        individuality = float(2.0 / (1.0 + np.exp(-np.mean(pairwise))) - 1.0) if pairwise else 0.0
        return coherence, individuality

    async def compute_within_agent_variance(self, traces: list[BehavioralTrace]) -> float:
        if len(traces) < 2:
            return 1.0
        embeddings = await self.embeddings.embed_batch([t.full_text for t in traces])
        variance = float(np.var(embeddings, axis=0).mean())
        return 1.0 / (1.0 + variance)
