"""BehavioralTrace — the atomic unit of personality-in-action."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class BehavioralTrace(BaseModel):
    """One unit of personality-conditioned behaviour in a situation."""
    situation: str = Field(description="The situation that was presented")
    perception: str = Field(description="What this personality noticed")
    cognition: str = Field(description="What they thought about it")
    emotion: str = Field(description="What they felt")
    action: str = Field(description="What they did or said")
    timestamp: float = Field(default_factory=time.time)
    context_tags: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return f"{self.perception} {self.cognition} {self.emotion} {self.action}"
