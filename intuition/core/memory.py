"""EpisodicMemory — self-reinforcing personality coherence through narrative."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Episode(BaseModel):
    """One memorable experience and its psychological residue."""
    situation_summary: str = Field(description="What happened")
    what_i_did: str = Field(description="How I responded")
    outcome: str = Field(description="What resulted")
    emotional_residue: str = Field(description="How it left me feeling")
    self_reflection: str = Field(description="What I learned about myself")
    importance: float = Field(ge=0.0, le=1.0, description="How formative")


class EpisodicMemory(BaseModel):
    """Accumulated experiences and the self-narrative they produce."""
    episodes: list[Episode] = Field(default_factory=list)
    self_narrative: str = Field(default="")
    recurring_themes: list[str] = Field(default_factory=list)

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > 50:
            self.episodes.sort(key=lambda e: e.importance, reverse=True)
            self.episodes = self.episodes[:50]

    def relevant_to(self, situation: str, k: int = 3) -> list[Episode]:
        situation_words = set(situation.lower().split())
        def relevance(ep: Episode) -> float:
            ep_words = set(ep.situation_summary.lower().split())
            return len(situation_words & ep_words) + ep.importance
        ranked = sorted(self.episodes, key=relevance, reverse=True)
        return ranked[:k]

    def format_for_prompt(self, situation: str, k: int = 3) -> str:
        relevant = self.relevant_to(situation, k)
        if not relevant:
            return ""
        lines = ["Relevant memories:"]
        for ep in relevant:
            lines.append(
                f"- {ep.situation_summary} -> I {ep.what_i_did}. "
                f"It left me feeling {ep.emotional_residue}. "
                f"I learned: {ep.self_reflection}"
            )
        if self.self_narrative:
            lines.append(f"\nSelf-narrative: {self.self_narrative}")
        return "\n".join(lines)
