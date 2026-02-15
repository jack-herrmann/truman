"""PersonalityKernel — the deep, structured representation of a personality."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


def _uid() -> str:
    return uuid.uuid4().hex[:12]


class Value(BaseModel):
    """A core value with priority ranking and stability under pressure."""
    name: str
    importance: float = Field(ge=0.0, le=1.0, description="Priority weight")
    stability: float = Field(ge=0.0, le=1.0, description="Resistance to pressure")
    description: str = Field(description="Rich narrative description of this value")
    conflicts_with: list[str] = Field(default_factory=list)


class CognitiveStyle(BaseModel):
    """How this personality thinks."""
    abstraction_level: float = Field(ge=-1.0, le=1.0, description="concrete (-1) to abstract (+1)")
    decision_mode: float = Field(ge=-1.0, le=1.0, description="intuitive (-1) to analytical (+1)")
    attention_priorities: list[str] = Field(description="What they notice first, in priority order")
    complexity_preference: float = Field(ge=0.0, le=1.0, description="simple (0) to complex (1)")
    thinking_tempo: float = Field(ge=0.0, le=1.0, description="deliberate (0) to rapid (1)")
    inner_monologue_style: str = Field(description="e.g. 'questioning', 'narrative', 'imagistic'")


class EmotionalBaseline(BaseModel):
    """The emotional landscape."""
    default_valence: float = Field(ge=-1.0, le=1.0, description="negative (-1) to positive (+1)")
    emotional_range: float = Field(ge=0.0, le=1.0, description="narrow (0) to wide (1)")
    reactivity: float = Field(ge=0.0, le=1.0, description="slow (0) to fast (1)")
    recovery_speed: float = Field(ge=0.0, le=1.0, description="slow (0) to fast (1)")
    dominant_emotions: list[str] = Field(description="Top 3-5 frequent emotions")
    emotional_depth: float = Field(ge=0.0, le=1.0, description="surface (0) to deep (1)")


class SocialStyle(BaseModel):
    """How this personality relates to others."""
    energy_direction: float = Field(ge=-1.0, le=1.0, description="solitary (-1) to social (+1)")
    trust_default: float = Field(ge=0.0, le=1.0, description="suspicious (0) to trusting (1)")
    conflict_approach: str = Field(description="avoid | confront | mediate | deflect")
    attachment_style: str = Field(description="secure | anxious | avoidant | disorganized")
    dominance: float = Field(ge=0.0, le=1.0, description="submissive (0) to dominant (1)")
    empathy_type: str = Field(description="cognitive | emotional | both | limited")


class PerceptualStyle(BaseModel):
    """What this personality notices and how they see the world."""
    primary_lens: str = Field(description="The dominant frame through which they interpret reality")
    notices_first: list[str] = Field(description="Attention priorities when entering a new situation")
    blind_spots: list[str] = Field(description="What they characteristically miss or overlook")
    aesthetic_sensibility: str = Field(description="Relationship to beauty, art, rhythm, and form")


class FaultLine(BaseModel):
    """An internal contradiction."""
    tension: list[str] = Field(min_length=2, max_length=2, description="The two conflicting tendencies")
    activation_context: str = Field(description="What kind of situation brings this tension to the surface")
    typical_resolution: str = Field(description="How they usually navigate or suppress the contradiction")
    stress_escalation: str = Field(description="What happens when they cannot resolve it")


class StressProfile(BaseModel):
    """How this personality responds under increasing pressure."""
    threshold: float = Field(ge=0.0, le=1.0, description="How much pressure before visible change")
    primary_response: str = Field(description="fight | flight | freeze | fawn")
    behavioral_shifts: list[str] = Field(description="How behavior shifts under stress, e.g. 'trust becomes suspicious'")
    breaking_point: str = Field(description="What they look like at maximum stress")


class AgentState(BaseModel):
    """Dynamic emotional/cognitive state — mutable at runtime."""
    current_emotion: str = "neutral"
    stress_level: float = Field(default=0.1, ge=0.0, le=1.0)
    arousal: float = Field(default=0.3, ge=0.0, le=1.0)
    recent_events: list[str] = Field(default_factory=list)
    current_context: str = ""


class PersonalityKernel(BaseModel):
    """The complete personality representation."""
    id: str = Field(default_factory=_uid)
    name: str = Field(description="Generated name for this personality")

    z: list[float] = Field(description="Latent vector from personality space")
    sigma: list[float] = Field(description="Per-dimension stability")

    values: list[Value]
    cognitive_style: CognitiveStyle
    emotional_baseline: EmotionalBaseline
    social_style: SocialStyle
    perceptual_style: PerceptualStyle
    fault_lines: list[FaultLine]
    stress_profile: StressProfile

    behavioral_summary: str = Field(description="2-3 paragraph narrative portrait")
    origin_sketch: str = Field(description="Imagined formative experiences")

    state: AgentState = Field(default_factory=AgentState)

    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str) -> "PersonalityKernel":
        from pathlib import Path
        return cls.model_validate_json(Path(path).read_text())
