"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Enable pytest-asyncio for all tests in tests/
pytest_plugins = ("pytest_asyncio",)


def pytest_configure(config):
    config.addinivalue_line("asyncio_mode", "auto")


@pytest.fixture
def temp_dir(tmp_path):
    """A temporary directory that persists for the test."""
    return tmp_path


@pytest.fixture
def sample_kernel():
    """Minimal valid PersonalityKernel for tests."""
    from intuition.core.kernel import (
        CognitiveStyle,
        EmotionalBaseline,
        FaultLine,
        PerceptualStyle,
        SocialStyle,
        StressProfile,
        Value,
        PersonalityKernel,
    )
    return PersonalityKernel(
        name="TestPerson",
        z=[0.1] * 32,
        sigma=[0.4] * 32,
        values=[
            Value(name="Honesty", importance=0.9, stability=0.8, description="Truth matters."),
        ],
        cognitive_style=CognitiveStyle(
            abstraction_level=0.2,
            decision_mode=-0.3,
            attention_priorities=["safety", "others"],
            complexity_preference=0.5,
            thinking_tempo=0.4,
            inner_monologue_style="questioning",
        ),
        emotional_baseline=EmotionalBaseline(
            default_valence=0.1,
            emotional_range=0.6,
            reactivity=0.5,
            recovery_speed=0.5,
            dominant_emotions=["calm", "curious", "wary"],
            emotional_depth=0.6,
        ),
        social_style=SocialStyle(
            energy_direction=-0.2,
            trust_default=0.5,
            conflict_approach="mediate",
            attachment_style="secure",
            dominance=0.4,
            empathy_type="both",
        ),
        perceptual_style=PerceptualStyle(
            primary_lens="Practical outcomes",
            notices_first=["risk", "other people"],
            blind_spots=["subtle cues"],
            aesthetic_sensibility="understated",
        ),
        fault_lines=[
            FaultLine(
                tension=("duty", "desire"),
                activation_context="moral choice",
                typical_resolution="compromise",
                stress_escalation="withdraw",
            ),
        ],
        stress_profile=StressProfile(
            threshold=0.6,
            primary_response="freeze",
            behavioral_shifts={"trust": "suspicious"},
            breaking_point="shutdown",
        ),
        behavioral_summary="A cautious, thoughtful person.",
        origin_sketch="Grew up in a strict environment.",
    )


@pytest.fixture
def sample_trace():
    """Minimal BehavioralTrace for tests."""
    from intuition.core.traces import BehavioralTrace
    return BehavioralTrace(
        situation="Someone lied to you.",
        perception="I noticed the hesitation.",
        cognition="I thought they might be hiding something.",
        emotion="Disappointed but not surprised.",
        action="I asked directly what was going on.",
    )


@pytest.fixture
def character_data_dir(temp_dir):
    """A directory with exactly 2 character JSON files and matching 2-row embeddings."""
    from intuition.corpus.extractor import CharacterProfile
    profiles = [
        CharacterProfile(
            name="Alice",
            novel="TestNovel",
            author="TestAuthor",
            personality_analysis="Alice is brave and direct. She values truth.",
        ),
        CharacterProfile(
            name="Bob",
            novel="TestNovel",
            author="TestAuthor",
            personality_analysis="Bob is cautious and kind. He avoids conflict.",
        ),
    ]
    for p in profiles:
        path = temp_dir / f"testnovel_{p.name.lower()}.json"
        path.write_text(p.model_dump_json(indent=2))
    # Embeddings: 2 x 512 (match config dimension)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((2, 512)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb = emb / norms
    np.save(str(temp_dir / "embeddings.npy"), emb)
    return temp_dir


@pytest.fixture
def small_vae():
    """A minimal VAE with small dimensions for fast tests."""
    from intuition.latent.vae import PersonalityVAE
    return PersonalityVAE(input_dim=512, latent_dim=8, hidden_dims=[64, 32])
