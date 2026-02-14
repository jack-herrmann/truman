"""PersonalityDecoder — translate latent vector z into a full PersonalityKernel."""

from __future__ import annotations

import logging

import numpy as np

from intuition.core.kernel import PersonalityKernel
from intuition.latent.space import PersonalitySpace
from intuition.llm.client import LLMClient
from intuition.llm.templates import TemplateEngine

logger = logging.getLogger(__name__)


class PersonalityDecoder:
    def __init__(self, space: PersonalitySpace, llm: LLMClient,
                 templates: TemplateEngine | None = None, k_neighbours: int = 5) -> None:
        self.space = space
        self.llm = llm
        self.templates = templates or TemplateEngine()
        self.k = k_neighbours

    async def decode(self, z: np.ndarray, sigma: np.ndarray) -> PersonalityKernel:
        neighbours = self.space.nearest_characters(z, k=self.k)
        profiles = [n[0] for n in neighbours]
        distances = [n[1] for n in neighbours]
        weights = self._distance_to_weights(distances)
        high_sigma = self._describe_sigma(sigma, high=True)
        low_sigma = self._describe_sigma(sigma, high=False)
        prompt = self.templates.render("decode_kernel.j2",
            neighbors=profiles, weights=weights,
            high_sigma_dims=high_sigma, low_sigma_dims=low_sigma)
        kernel = await self.llm.generate_structured(
            system=("You are a personality architect. Generate a complete, deeply coherent "
                    "PersonalityKernel. Every field must be filled with rich, specific content."),
            messages=[{"role": "user", "content": prompt}],
            response_model=PersonalityKernel, temperature=0.8)
        kernel.z = z.tolist()
        kernel.sigma = sigma.tolist()
        return kernel

    @staticmethod
    def _distance_to_weights(distances):
        inv = [1.0 / (d + 1e-6) for d in distances]
        total = sum(inv)
        return [w / total for w in inv]

    @staticmethod
    def _describe_sigma(sigma, high=True, top_k=5):
        indices = np.argsort(sigma)
        if high:
            indices = indices[-top_k:][::-1]
            label = "unstable / fault-line"
        else:
            indices = indices[:top_k]
            label = "stable / bedrock"
        parts = [f"dimension {idx} (sigma={float(sigma[idx]):.3f})" for idx in indices]
        return f"{label} dimensions: {', '.join(parts)}"
