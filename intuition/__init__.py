"""Intuition — AI Personality Engine.

Create truly individual and coherent artificial personalities.

Usage:
    from intuition import create_personality, create_agent

    kernel = await create_personality()
    agent = await create_agent(kernel)
    response = await agent.respond("Tell me about yourself.")
"""

from intuition.core.kernel import PersonalityKernel
from intuition.core.traces import BehavioralTrace
from intuition.core.memory import EpisodicMemory, Episode
from intuition.api import create_personality, create_agent, load_personality, save_personality

__all__ = [
    "PersonalityKernel",
    "BehavioralTrace",
    "EpisodicMemory",
    "Episode",
    "create_personality",
    "create_agent",
    "load_personality",
    "save_personality",
]
