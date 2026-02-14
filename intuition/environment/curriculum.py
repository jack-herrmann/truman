"""Curriculum — structured sequences of situations for personality development."""

from __future__ import annotations

import random

from intuition.core.kernel import PersonalityKernel
from intuition.environment.situations import SituationCategory


class CurriculumStep:
    def __init__(self, category: SituationCategory, stakes: float = 0.5) -> None:
        self.category = category
        self.stakes = stakes


class Curriculum:
    def generate_episode(self, kernel: PersonalityKernel | None = None, num_steps: int = 12):
        steps = []
        warmup_cats = [SituationCategory.MUNDANE_CHOICE, SituationCategory.AESTHETIC, SituationCategory.KINDNESS]
        social_cats = [SituationCategory.SOCIAL_DILEMMA, SituationCategory.CONFLICT, SituationCategory.MORAL_AMBIGUITY]
        for i in range(max(2, num_steps // 5)):
            steps.append(CurriculumStep(random.choice(warmup_cats), stakes=0.1 + 0.1 * i))
        for i in range(max(2, num_steps // 5)):
            steps.append(CurriculumStep(random.choice(social_cats), stakes=0.3 + 0.1 * i))
        steps.append(CurriculumStep(SituationCategory.BOREDOM, stakes=0.1))
        remaining = num_steps - len(steps) - 3
        for i in range(max(2, remaining)):
            steps.append(CurriculumStep(random.choice(social_cats), stakes=min(0.9, 0.5 + 0.4 * i / max(1, remaining - 1))))
        steps.append(CurriculumStep(SituationCategory.CONTRADICTION, stakes=0.8))
        steps.append(CurriculumStep(random.choice([SituationCategory.LOSS, SituationCategory.CONFLICT]), stakes=0.95))
        steps.append(CurriculumStep(random.choice([SituationCategory.MUNDANE_CHOICE, SituationCategory.AESTHETIC]), stakes=0.1))
        return steps[:num_steps]

    def generate_evaluation_episode(self, num_steps: int = 8):
        categories = [(SituationCategory.MUNDANE_CHOICE, 0.2), (SituationCategory.SOCIAL_DILEMMA, 0.5),
                       (SituationCategory.MORAL_AMBIGUITY, 0.6), (SituationCategory.CONFLICT, 0.5),
                       (SituationCategory.BOREDOM, 0.1), (SituationCategory.AESTHETIC, 0.3),
                       (SituationCategory.LOSS, 0.7), (SituationCategory.ACHIEVEMENT, 0.4)]
        return [CurriculumStep(cat, stakes) for cat, stakes in categories[:num_steps]]
