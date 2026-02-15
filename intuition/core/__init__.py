"""Core data structures for Intuition (PersonalityKernel, traces, memory)."""

from intuition.core.kernel import (
    PersonalityKernel,
    Value,
    CognitiveStyle,
    EmotionalBaseline,
    SocialStyle,
    PerceptualStyle,
    FaultLine,
    StressProfile,
    AgentState,
)
from intuition.core.traces import BehavioralTrace
from intuition.core.memory import EpisodicMemory, Episode

__all__ = [
    "PersonalityKernel",
    "Value",
    "CognitiveStyle",
    "EmotionalBaseline",
    "SocialStyle",
    "PerceptualStyle",
    "FaultLine",
    "StressProfile",
    "AgentState",
    "BehavioralTrace",
    "EpisodicMemory",
    "Episode",
]
