"""PromptBuilder — translate a PersonalityKernel into an LLM system prompt."""

from __future__ import annotations

from intuition.core.kernel import PersonalityKernel
from intuition.core.memory import EpisodicMemory
from intuition.llm.templates import TemplateEngine


class PromptBuilder:
    def __init__(self, templates: TemplateEngine | None = None) -> None:
        self.templates = templates or TemplateEngine()

    def build(self, kernel: PersonalityKernel) -> str:
        return self.templates.render("agent_system.j2", kernel=kernel,
                                     state_text=self._format_state(kernel), memory_text="")

    def build_with_context(self, kernel: PersonalityKernel, situation: str = "",
                           memory: EpisodicMemory | None = None) -> str:
        state_text = self._format_state(kernel)
        memory_text = memory.format_for_prompt(situation) if memory and memory.episodes else ""
        return self.templates.render("agent_system.j2", kernel=kernel,
                                     state_text=state_text, memory_text=memory_text)

    @staticmethod
    def _format_state(kernel: PersonalityKernel) -> str:
        s = kernel.state
        lines = [f"Current emotion: {s.current_emotion}",
                 f"Stress level: {s.stress_level:.1f}/1.0",
                 f"Arousal: {s.arousal:.1f}/1.0"]
        if s.recent_events:
            lines.append("Recent events: " + "; ".join(s.recent_events[-3:]))
        if s.current_context:
            lines.append(f"Context: {s.current_context}")
        return "\n".join(lines)
