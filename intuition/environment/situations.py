"""SituationBank — generate identity-revealing situations."""

from __future__ import annotations

import random
from enum import Enum

from intuition.core.kernel import PersonalityKernel
from intuition.core.traces import BehavioralTrace
from intuition.llm.client import LLMClient
from intuition.llm.templates import TemplateEngine


class SituationCategory(str, Enum):
    SOCIAL_DILEMMA = "social_dilemma"
    MORAL_AMBIGUITY = "moral_ambiguity"
    MUNDANE_CHOICE = "mundane_choice"
    CONFLICT = "conflict"
    KINDNESS = "kindness"
    BOREDOM = "boredom"
    AESTHETIC = "aesthetic"
    LOSS = "loss"
    ACHIEVEMENT = "achievement"
    CONTRADICTION = "contradiction"
    ESCALATION = "escalation"


class SituationBank:
    def __init__(self, llm: LLMClient, templates: TemplateEngine | None = None) -> None:
        self.llm = llm
        self.templates = templates or TemplateEngine()

    async def generate(self, category: SituationCategory, stakes: float = 0.5,
                       agent_history: list[BehavioralTrace] | None = None) -> str:
        prompt = self.templates.render("situation_generate.j2",
            category=category.value, stakes=stakes, agent_history=agent_history or [])
        return (await self.llm.generate(
            system="You are a narrative situation generator. Create vivid, identity-revealing scenarios. Second person. 2-4 paragraphs.",
            messages=[{"role": "user", "content": prompt}], temperature=0.85)).strip()

    async def generate_contradiction_trigger(self, kernel: PersonalityKernel) -> str:
        if not kernel.fault_lines:
            return await self.generate(SituationCategory.MORAL_AMBIGUITY, stakes=0.7)
        fl = random.choice(kernel.fault_lines)
        prompt = (f"Create a situation forcing a choice between:\n  A: {fl.tension[0]}\n  B: {fl.tension[1]}\n\n"
                  f"Context: {fl.activation_context}\n\nWrite 2-4 paragraphs, second person. Impossible to resolve cleanly.")
        return (await self.llm.generate(
            system="You are a narrative designer specialising in psychological dilemmas.",
            messages=[{"role": "user", "content": prompt}], temperature=0.8)).strip()
