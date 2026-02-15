"""Tests for training rewards (CoherenceReward, IndividualityReward, discriminator)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from intuition.core.traces import BehavioralTrace
from intuition.training.discriminator import TraceDiscriminator
from intuition.training.rewards import CoherenceReward, IndividualityReward


@pytest.fixture
def mock_embedding_client():
    """Embedding client that returns deterministic vectors per text."""
    client = MagicMock()
    client.dimension = 64

    async def embed(text):
        # Deterministic from text hash
        h = hash(text) % (2 ** 32)
        rng = np.random.RandomState(h)
        v = rng.randn(64).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    async def embed_batch(texts):
        out = []
        for t in texts:
            out.append(await embed(t))
        return np.stack(out)

    client.embed = AsyncMock(side_effect=embed)
    client.embed_batch = AsyncMock(side_effect=embed_batch)
    return client


def make_traces(n, prefix="Agent"):
    return [
        BehavioralTrace(
            situation="A situation.",
            perception="Noticed something.",
            cognition="Thought about it.",
            emotion="Felt something.",
            action=f"{prefix} action {i}.",
        )
        for i in range(n)
    ]


async def test_discriminator_compute_scores(mock_embedding_client):
    disc = TraceDiscriminator(mock_embedding_client)
    traces_a = make_traces(3, "A")
    traces_b = make_traces(3, "B")
    coherence, individuality = await disc.compute_scores([traces_a, traces_b])
    assert 0 <= coherence <= 1
    assert 0 <= individuality <= 1


async def test_discriminator_within_agent_variance(mock_embedding_client):
    disc = TraceDiscriminator(mock_embedding_client)
    traces = make_traces(4, "Same")
    score = await disc.compute_within_agent_variance(traces)
    assert 0 <= score <= 1


async def test_individuality_reward_compute_per_agent(mock_embedding_client):
    reward = IndividualityReward(mock_embedding_client)
    traces_per_agent = [make_traces(2, "X"), make_traces(2, "Y")]
    scores = await reward.compute_per_agent(traces_per_agent)
    assert len(scores) == 2
    assert all(0 <= s <= 1 for s in scores)


async def test_coherence_reward_short_traces(mock_embedding_client):
    llm = AsyncMock(return_value="7")
    reward = CoherenceReward(mock_embedding_client, llm)
    agent = MagicMock()
    agent.kernel = MagicMock()
    agent.traces = make_traces(1)
    score = await reward.compute(agent, agent.traces)
    assert score == 0.5  # len(traces) < 2


async def test_coherence_reward_with_narrative(mock_embedding_client):
    llm = AsyncMock(return_value="8")
    reward = CoherenceReward(mock_embedding_client, llm)
    agent = MagicMock()
    agent.kernel = MagicMock()
    agent.traces = make_traces(5, "SamePerson")
    score = await reward.compute(agent, agent.traces)
    assert 0 <= score <= 1
