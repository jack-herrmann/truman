"""KernelOptimizer — evolutionary optimisation of personality kernels."""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
from intuition.core.kernel import PersonalityKernel
from intuition.latent.decoder import PersonalityDecoder
from intuition.latent.space import PersonalitySpace
logger = logging.getLogger(__name__)


@dataclass
class ScoredKernel:
    kernel: PersonalityKernel
    coherence: float
    individuality: float
    combined: float


class KernelOptimizer:
    def __init__(self, space: PersonalitySpace, decoder: PersonalityDecoder,
                 population_size: int = 12, elite_fraction: float = 0.25,
                 mutation_radius: float = 0.3, exploration_fraction: float = 0.2) -> None:
        self.space = space
        self.decoder = decoder
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_radius = mutation_radius
        self.exploration_fraction = exploration_fraction
        self.rng = np.random.default_rng()
        self.population: list[ScoredKernel] = []
        self.generation = 0

    async def initialise(self) -> list[PersonalityKernel]:
        kernels = []
        for _ in range(self.population_size):
            z, sigma = self.space.sample(temperature=1.0, rng=self.rng)
            kernels.append(await self.decoder.decode(z, sigma))
        self.population = [ScoredKernel(k, 0.0, 0.0, 0.0) for k in kernels]
        return kernels

    def update_scores(self, coherence_scores, individuality_scores,
                      coherence_weight=0.5, individuality_weight=0.5):
        for sk in self.population:
            kid = sk.kernel.id
            sk.coherence = coherence_scores.get(kid, 0.0)
            sk.individuality = individuality_scores.get(kid, 0.0)
            sk.combined = coherence_weight * sk.coherence + individuality_weight * sk.individuality
        self.population.sort(key=lambda x: x.combined, reverse=True)

    async def evolve(self) -> list[PersonalityKernel]:
        self.generation += 1
        n_elite = max(1, int(self.population_size * self.elite_fraction))
        n_explore = max(1, int(self.population_size * self.exploration_fraction))
        n_mutate = self.population_size - n_elite - n_explore
        new_kernels = [sk.kernel for sk in self.population[:n_elite]]
        for i in range(n_mutate):
            parent = self.population[i % n_elite]
            z_p = np.array(parent.kernel.z, dtype=np.float32)
            s_p = np.array(parent.kernel.sigma, dtype=np.float32)
            z_c = z_p + self.rng.standard_normal(len(z_p)).astype(np.float32) * self.mutation_radius
            s_c = s_p * (0.9 + 0.2 * self.rng.random(len(s_p)).astype(np.float32))
            new_kernels.append(await self.decoder.decode(z_c, s_c))
        for _ in range(n_explore):
            z, s = self.space.sample(temperature=1.2, rng=self.rng)
            new_kernels.append(await self.decoder.decode(z, s))
        self.population = [ScoredKernel(k, 0.0, 0.0, 0.0) for k in new_kernels]
        return new_kernels

    @property
    def best(self):
        return max(self.population, key=lambda x: x.combined) if self.population else None

    def current_kernels(self):
        return [sk.kernel for sk in self.population]
