"""Tests for intuition.agent (StateManager, PromptBuilder, tag inference)."""
from __future__ import annotations

import pytest

from intuition.agent.state import StateManager
from intuition.agent.prompt_builder import PromptBuilder
from intuition.agent.agent import PersonalityAgent
from intuition.core.traces import BehavioralTrace


def test_state_manager_update(sample_kernel, sample_trace):
    sm = StateManager(sample_kernel)
    initial_stress = sm.state.stress_level
    sm.update(sample_trace)
    assert sm.state.current_emotion is not None
    assert len(sm.state.recent_events) <= 5
    sample_kernel.state = sm.state


def test_state_manager_stress_tags(sample_kernel):
    from intuition.core.kernel import AgentState
    sample_kernel.state = AgentState(stress_level=0.2)
    sm = StateManager(sample_kernel)
    high_stakes_trace = BehavioralTrace(
        situation="Emergency.",
        perception="Danger.",
        cognition="Act now.",
        emotion="Fear.",
        action="Ran.",
        context_tags=["high_stakes"],
    )
    sm.update(high_stakes_trace)
    assert sm.state.stress_level >= 0.2


def test_prompt_builder_build(sample_kernel):
    pb = PromptBuilder()
    text = pb.build(sample_kernel)
    assert sample_kernel.name in text
    assert "Honesty" in text or "honesty" in text
    assert "CORE VALUES" in text or "VALUES" in text


def test_prompt_builder_build_with_context(sample_kernel):
    from intuition.core.memory import EpisodicMemory, Episode
    pb = PromptBuilder()
    memory = EpisodicMemory()
    memory.add(Episode(
        situation_summary="Test situation",
        what_i_did="Did something",
        outcome="OK",
        emotional_residue="Fine",
        self_reflection="Learned.",
        importance=0.5,
    ))
    text = pb.build_with_context(sample_kernel, situation="Something happened", memory=memory)
    assert sample_kernel.name in text
    assert "Test situation" in text or "MEMORIES" in text or "Relevant" in text


def test_agent_infer_tags():
    tags = PersonalityAgent._infer_tags("A friend invited you to a party with people you don't know.")
    assert "social" in tags
    tags = PersonalityAgent._infer_tags("You see a beautiful sunset over the ocean.")
    assert "aesthetic" in tags or "mundane" in tags
    tags = PersonalityAgent._infer_tags("Nothing happens. Empty room. Waiting.")
    assert "boredom" in tags or "mundane" in tags
