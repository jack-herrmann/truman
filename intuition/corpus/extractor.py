"""CharacterExtractor — LLM-powered extraction of personality profiles from novels."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from intuition.llm.client import LLMClient
from intuition.llm.templates import TemplateEngine

logger = logging.getLogger(__name__)


class CharacterEvidence(BaseModel):
    decisions: list[str] = Field(default_factory=list)
    dialogue: list[str] = Field(default_factory=list)
    monologue: list[str] = Field(default_factory=list)
    reactions: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    stress: list[str] = Field(default_factory=list)

    def merge(self, other: "CharacterEvidence") -> None:
        for field in ["decisions","dialogue","monologue","reactions","notices","relationships","contradictions","stress"]:
            getattr(self, field).extend(getattr(other, field))

    @property
    def total_evidence(self) -> int:
        return sum(len(getattr(self, f)) for f in ["decisions","dialogue","monologue","reactions","notices","relationships","contradictions","stress"])


class CharacterProfile(BaseModel):
    name: str
    novel: str
    author: str
    key_decisions: list[str] = Field(default_factory=list)
    dialogue_samples: list[str] = Field(default_factory=list)
    internal_monologue: list[str] = Field(default_factory=list)
    emotional_reactions: list[str] = Field(default_factory=list)
    what_they_notice: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    under_stress: list[str] = Field(default_factory=list)
    values_expressed: list[str] = Field(default_factory=list)
    growth_arc: str = ""
    personality_analysis: str = Field(default="")

    def save(self, path: str) -> None:
        from pathlib import Path as P
        P(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str) -> "CharacterProfile":
        from pathlib import Path as P
        return cls.model_validate_json(P(path).read_text())


class ChunkCharacterEvidence(BaseModel):
    character_name: str
    decisions: list[str] = Field(default_factory=list)
    dialogue: list[str] = Field(default_factory=list)
    monologue: list[str] = Field(default_factory=list)
    reactions: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    stress: list[str] = Field(default_factory=list)


class ChunkExtractionResult(BaseModel):
    characters: list[ChunkCharacterEvidence] = Field(default_factory=list)


class CharacterList(BaseModel):
    characters: list[str] = Field(description="Names of major characters (3-6)")


class SynthesisResult(BaseModel):
    personality_analysis: str
    values_expressed: list[str]
    growth_arc: str


class CharacterExtractor:
    def __init__(self, llm: LLMClient, templates: TemplateEngine | None = None,
                 min_evidence: int = 5) -> None:
        self.llm = llm
        self.templates = templates or TemplateEngine()
        self.min_evidence = min_evidence

    async def extract_from_novel(self, text: str, title: str, author: str,
                                  max_characters: int = 6) -> list[CharacterProfile]:
        from intuition.corpus.gutenberg import GutenbergCorpus
        chunks = GutenbergCorpus.chunk_text(text)
        characters = await self._identify_characters(chunks[:5], title, author, max_characters)
        evidence: dict[str, CharacterEvidence] = {name: CharacterEvidence() for name in characters}
        for i, chunk in enumerate(chunks):
            chunk_evidence = await self._extract_chunk(chunk, characters, title, author)
            for char_ev in chunk_evidence:
                name_lower = char_ev.character_name.lower()
                for known_name in characters:
                    if known_name.lower() in name_lower or name_lower in known_name.lower():
                        ev = CharacterEvidence(decisions=char_ev.decisions, dialogue=char_ev.dialogue,
                            monologue=char_ev.monologue, reactions=char_ev.reactions,
                            notices=char_ev.notices, relationships=char_ev.relationships,
                            contradictions=char_ev.contradictions, stress=char_ev.stress)
                        evidence[known_name].merge(ev)
                        break
        profiles: list[CharacterProfile] = []
        for name in characters:
            ev = evidence[name]
            if ev.total_evidence < self.min_evidence:
                continue
            profile = await self._synthesize(name, ev, title, author)
            profiles.append(profile)
        return profiles

    async def _identify_characters(self, early_chunks, title, author, max_characters):
        combined = "\n\n---\n\n".join(early_chunks[:5])
        result = await self.llm.generate_structured(
            system=f"Identify the {max_characters} most important characters in this novel excerpt.",
            messages=[{"role": "user", "content": f"Novel: {title} by {author}\n\n{combined[:8000]}"}],
            response_model=CharacterList)
        return result.characters[:max_characters]

    async def _extract_chunk(self, chunk, character_names, title, author):
        prompt = self.templates.render("extract_character.j2",
            novel_title=title, author=author, character_names=character_names, passage=chunk[:6000])
        try:
            result = await self.llm.generate_structured(
                system="Extract character behavioural evidence from this passage.",
                messages=[{"role": "user", "content": prompt}],
                response_model=ChunkExtractionResult, temperature=0.2)
            return result.characters
        except Exception:
            return []

    async def _synthesize(self, name, evidence, title, author):
        ev_obj = type("Ev", (), {
            "decisions": evidence.decisions[:20], "dialogue": evidence.dialogue[:15],
            "monologue": evidence.monologue[:15], "reactions": evidence.reactions[:15],
            "notices": evidence.notices[:10], "relationships": evidence.relationships[:10],
            "contradictions": evidence.contradictions[:10], "stress": evidence.stress[:10]})()
        prompt = self.templates.render("synthesize_character.j2",
            character_name=name, novel_title=title, author=author, evidence=ev_obj)
        synthesis = await self.llm.generate_structured(
            system="Synthesize this character evidence into a deep personality analysis.",
            messages=[{"role": "user", "content": prompt}],
            response_model=SynthesisResult, temperature=0.5)
        return CharacterProfile(
            name=name, novel=title, author=author,
            key_decisions=evidence.decisions[:20], dialogue_samples=evidence.dialogue[:15],
            internal_monologue=evidence.monologue[:15], emotional_reactions=evidence.reactions[:15],
            what_they_notice=evidence.notices[:10], relationships=evidence.relationships[:10],
            contradictions=evidence.contradictions[:10], under_stress=evidence.stress[:10],
            values_expressed=synthesis.values_expressed, growth_arc=synthesis.growth_arc,
            personality_analysis=synthesis.personality_analysis)
