"""Tests for intuition.core (kernel, traces, memory)."""
from __future__ import annotations

import tempfile

import pytest

from intuition.core.kernel import PersonalityKernel, Value, AgentState
from intuition.core.traces import BehavioralTrace
from intuition.core.memory import EpisodicMemory, Episode


def test_value_validation():
    v = Value(name="X", importance=0.5, stability=0.5, description="Test.")
    assert v.name == "X"
    assert v.importance == 0.5


def test_behavioral_trace_full_text(sample_trace):
    full = sample_trace.full_text
    assert "hesitation" in full
    assert "asked directly" in full


def test_kernel_save_load(sample_kernel, temp_dir):
    path = temp_dir / "kernel.json"
    sample_kernel.save(str(path))
    loaded = PersonalityKernel.load(str(path))
    assert loaded.name == sample_kernel.name
    assert loaded.z == sample_kernel.z
    assert len(loaded.values) == len(sample_kernel.values)


def test_episodic_memory_add_and_relevant():
    mem = EpisodicMemory()
    mem.add(Episode(
        situation_summary="A friend asked for help.",
        what_i_did="I helped them.",
        outcome="Good.",
        emotional_residue="Warm.",
        self_reflection="I care about friends.",
        importance=0.8,
    ))
    mem.add(Episode(
        situation_summary="Someone was rude.",
        what_i_did="I walked away.",
        outcome="Neutral.",
        emotional_residue="Annoyed.",
        self_reflection="",
        importance=0.3,
    ))
    relevant = mem.relevant_to("friend help", k=2)
    assert len(relevant) >= 1
    assert "friend" in relevant[0].situation_summary.lower() or relevant[0].importance >= 0.5


def test_episodic_memory_cap():
    mem = EpisodicMemory()
    for i in range(60):
        mem.add(Episode(
            situation_summary=f"Situation {i}",
            what_i_did="X",
            outcome="Y",
            emotional_residue="Z",
            self_reflection="",
            importance=0.1 + i * 0.01,
        ))
    assert len(mem.episodes) <= 50


def test_episodic_memory_format_for_prompt():
    mem = EpisodicMemory()
    mem.add(Episode(
        situation_summary="Rainy day",
        what_i_did="Stayed in",
        outcome="Relaxed",
        emotional_residue="Calm",
        self_reflection="I like quiet.",
        importance=0.7,
    ))
    text = mem.format_for_prompt("rain", k=2)
    assert "Relevant memories" in text or "Rainy" in text
