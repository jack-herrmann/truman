"""TrumanWorld — the complete episode runner."""

from __future__ import annotations

import logging

from intuition.agent.agent import PersonalityAgent
from intuition.core.traces import BehavioralTrace
from intuition.environment.curriculum import Curriculum
from intuition.environment.narrator import WorldNarrator
from intuition.environment.situations import SituationBank, SituationCategory
from intuition.llm.client import LLMClient
from intuition.llm.templates import TemplateEngine

logger = logging.getLogger(__name__)


class TrumanWorld:
    def __init__(self, llm: LLMClient, templates: TemplateEngine | None = None) -> None:
        self.llm = llm
        templates = templates or TemplateEngine()
        self.narrator = WorldNarrator(llm, templates)
        self.situation_bank = SituationBank(llm, templates)
        self.curriculum = Curriculum()

    async def run_episode(self, agent: PersonalityAgent, num_steps: int = 12):
        self.narrator.reset()
        plan = self.curriculum.generate_episode(agent.kernel, num_steps)
        traces = []
        for step in plan:
            if step.category == SituationCategory.CONTRADICTION:
                situation = await self.situation_bank.generate_contradiction_trigger(agent.kernel)
            else:
                situation = await self.situation_bank.generate(
                    category=step.category, stakes=step.stakes,
                    agent_history=traces[-3:] if traces else None)
            trace = await agent.act(situation)
            traces.append(trace)
            await self.narrator.respond_to_action(trace)
        return traces

    async def run_evaluation_episode(self, agent: PersonalityAgent, num_steps: int = 8):
        self.narrator.reset()
        plan = self.curriculum.generate_evaluation_episode(num_steps)
        traces = []
        for step in plan:
            situation = await self.situation_bank.generate(
                category=step.category, stakes=step.stakes,
                agent_history=traces[-2:] if traces else None)
            traces.append(await agent.act(situation))
        return traces

    async def run_shared_scenarios(self, agents: list[PersonalityAgent], scenarios: list[str]):
        all_traces = []
        for agent in agents:
            agent_traces = [await agent.act(s) for s in scenarios]
            all_traces.append(agent_traces)
        return all_traces
