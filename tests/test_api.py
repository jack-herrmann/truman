"""Tests for public API (save/load, create_personality fallback, config)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intuition.api import (
    _load_config,
    save_personality,
    load_personality,
    create_personality,
    create_agent,
)
from intuition.core.kernel import PersonalityKernel


def test_load_config_finds_yaml():
    config = _load_config()
    assert "llm" in config
    assert "latent" in config or "llm" in config


def test_save_and_load_personality(sample_kernel, temp_dir):
    path = temp_dir / "p.json"
    save_personality(sample_kernel, str(path))
    assert path.exists()
    loaded = load_personality(str(path))
    assert loaded.name == sample_kernel.name
    assert loaded.z == sample_kernel.z


async def test_create_personality_fallback_when_latent_fails(sample_kernel):
    """When latent path raises (e.g. missing VAE or mismatch), fallback to direct LLM."""
    mock_llm = MagicMock()
    sample_kernel.name = "FallbackPerson"
    mock_llm.generate_structured = AsyncMock(return_value=sample_kernel)

    # Force latent path to raise (e.g. FileNotFoundError for VAE)
    with patch("intuition.api._create_from_latent_space", new_callable=AsyncMock) as m:
        m.side_effect = FileNotFoundError("VAE not trained")
        result = await create_personality(z=[0.1] * 32, llm=mock_llm)
    # Should have fallen back to _create_direct and called LLM
    assert result is not None
    assert result.name == "FallbackPerson"
    mock_llm.generate_structured.assert_called_once()


async def test_create_agent_returns_agent(sample_kernel):
    mock_llm = MagicMock()
    agent = await create_agent(sample_kernel, llm=mock_llm)
    from intuition.agent.agent import PersonalityAgent
    assert isinstance(agent, PersonalityAgent)
    assert agent.kernel is sample_kernel
    assert agent.llm is mock_llm
