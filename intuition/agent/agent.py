"""PersonalityAgent — an LLM agent conditioned on a PersonalityKernel."""

from __future__ import annotations

import logging

from intuition.agent.prompt_builder import PromptBuilder
from intuition.agent.state import StateManager
from intuition.core.kernel import PersonalityKernel
from intuition.core.memory import Episode, EpisodicMemory
from intuition.core.traces import BehavioralTrace
from intuition.llm.client import LLMClient

logger = logging.getLogger(__name__)


class PersonalityAgent:
    def __init__(self, kernel: PersonalityKernel, llm: LLMClient,
                 prompt_builder: PromptBuilder | None = None) -> None:
        self.kernel = kernel
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.state_manager = StateManager(kernel)
        self.memory = EpisodicMemory()
        self.traces: list[BehavioralTrace] = []

    async def act(self, situation: str) -> BehavioralTrace:
        system_prompt = self.prompt_builder.build_with_context(
            self.kernel, situation=situation, memory=self.memory)
        user_prompt = (f"You encounter this situation:\n\n{situation}\n\n"
                       "Respond as yourself. Describe:\n"
                       "1. What you NOTICE first (perception)\n"
                       "2. What you THINK about it (cognition)\n"
                       "3. What you FEEL (emotion)\n"
                       "4. What you DO or SAY (action)\n\nBe specific, be honest, be you.")
        trace = await self.llm.generate_structured(
            system=system_prompt, messages=[{"role": "user", "content": user_prompt}],
            response_model=BehavioralTrace, temperature=0.7)
        trace.situation = situation
        trace.context_tags = self._infer_tags(situation)
        self.state_manager.update(trace)
        self.traces.append(trace)
        self._maybe_remember(trace)
        return trace

    async def respond(self, prompt: str) -> str:
        system_prompt = self.prompt_builder.build_with_context(
            self.kernel, situation=prompt, memory=self.memory)
        return await self.llm.generate(system=system_prompt,
                                       messages=[{"role": "user", "content": prompt}], temperature=0.7)

    def _maybe_remember(self, trace: BehavioralTrace) -> None:
        importance = self.state_manager.state.stress_level * 0.3
        if set(trace.context_tags) & {"high_stakes", "moral_ambiguity", "loss", "achievement"}:
            importance += 0.3
        if self.state_manager.fault_lines_active():
            importance += 0.2
        if importance >= 0.3:
            self.memory.add(Episode(situation_summary=trace.situation[:200],
                what_i_did=trace.action[:200], outcome="(pending)",
                emotional_residue=trace.emotion[:100], self_reflection="",
                importance=min(1.0, importance)))

    @staticmethod
    def _infer_tags(situation: str) -> list[str]:
        text = situation.lower()
        tag_keywords = {
            "social": ["friend","party","conversation","someone","group","people"],
            "conflict": ["argue","disagree","confront","angry","upset","fight"],
            "moral_ambiguity": ["should you","right thing","honest","lie","steal","cheat"],
            "high_stakes": ["emergency","crisis","danger","risk","everything","life"],
            "loss": ["lost","gone","died","miss","broken","ending"],
            "achievement": ["succeed","won","accomplish","proud","recognition"],
            "boredom": ["nothing","empty","quiet","alone","waiting","idle"],
            "aesthetic": ["beautiful","music","art","sunset","rain","light"],
        }
        tags = [tag for tag, kws in tag_keywords.items() if any(kw in text for kw in kws)]
        return tags or ["mundane"]
