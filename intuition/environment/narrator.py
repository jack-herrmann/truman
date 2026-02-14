"""WorldNarrator — the LLM dungeon master for the Truman Show."""

from __future__ import annotations

import logging

from intuition.core.traces import BehavioralTrace
from intuition.environment.situations import SituationCategory
from intuition.llm.client import LLMClient
from intuition.llm.templates import TemplateEngine

logger = logging.getLogger(__name__)


class WorldNarrator:
    def __init__(self, llm: LLMClient, templates: TemplateEngine | None = None) -> None:
        self.llm = llm
        self.templates = templates or TemplateEngine()
        self.history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.history = []

    async def present_situation(self, category: SituationCategory | None = None,
                                agent_history: list[BehavioralTrace] | None = None) -> str:
        system = self.templates.render("narrator_system.j2",
            situation_category=category.value if category else None,
            agent_history=agent_history or [])
        messages = list(self.history)
        messages.append({"role": "user", "content": "Generate the next situation. Vivid, specific, identity-revealing. 2-4 paragraphs, second person."})
        situation = await self.llm.generate(system=system, messages=messages, temperature=0.85)
        self.history.append({"role": "assistant", "content": situation})
        return situation.strip()

    async def respond_to_action(self, trace: BehavioralTrace) -> str:
        action_summary = (f"The person:\n  Noticed: {trace.perception[:200]}\n"
                          f"  Thought: {trace.cognition[:200]}\n  Felt: {trace.emotion[:100]}\n"
                          f"  Did/said: {trace.action[:300]}")
        self.history.append({"role": "user", "content": action_summary})
        response = await self.llm.generate(
            system="You are the world narrator. Describe natural consequences. 1-2 paragraphs, second person.",
            messages=self.history, temperature=0.7)
        self.history.append({"role": "assistant", "content": response})
        return response.strip()
