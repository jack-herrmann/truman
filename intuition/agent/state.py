"""StateManager — z_state dynamics for personality agents."""

from __future__ import annotations

from intuition.core.kernel import AgentState, PersonalityKernel
from intuition.core.traces import BehavioralTrace


class StateManager:
    def __init__(self, kernel: PersonalityKernel) -> None:
        self.kernel = kernel
        baseline = kernel.emotional_baseline
        self.state = kernel.state or AgentState(
            current_emotion=baseline.dominant_emotions[0] if baseline.dominant_emotions else "neutral",
            stress_level=0.1, arousal=0.3)

    def update(self, trace: BehavioralTrace) -> AgentState:
        baseline = self.kernel.emotional_baseline
        self.state.current_emotion = trace.emotion[:100]
        is_stressful = any(tag in trace.context_tags
                           for tag in ["high_stakes", "conflict", "loss", "moral_ambiguity"])
        if is_stressful:
            self.state.stress_level = min(1.0, self.state.stress_level + 0.1 * (0.5 + baseline.reactivity))
        else:
            self.state.stress_level = max(0.0, self.state.stress_level - 0.05 * (0.3 + baseline.recovery_speed))
        if baseline.reactivity > 0.5:
            self.state.arousal = min(1.0, self.state.arousal + 0.05)
        else:
            self.state.arousal = max(0.1, self.state.arousal - 0.02)
        self.state.recent_events.append(trace.action[:150])
        self.state.recent_events = self.state.recent_events[-5:]
        self.kernel.state = self.state
        return self.state

    def is_stressed(self) -> bool:
        return self.state.stress_level > self.kernel.stress_profile.threshold

    def fault_lines_active(self) -> bool:
        return self.state.stress_level > (self.kernel.stress_profile.threshold * 0.8)
