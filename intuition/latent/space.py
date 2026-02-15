"""PersonalitySpace — navigate, sample, interpolate in the learned latent space."""

from __future__ import annotations

import logging

import numpy as np

from intuition.corpus.dataset import CharacterDataset
from intuition.corpus.extractor import CharacterProfile
from intuition.latent.vae import PersonalityVAE

logger = logging.getLogger(__name__)


class PersonalitySpace:
    def __init__(self, vae: PersonalityVAE, dataset: CharacterDataset, embeddings: np.ndarray) -> None:
        n_profiles = len(dataset)
        if embeddings.shape[0] != n_profiles:
            raise ValueError(
                "Embeddings row count (%d) must match number of character profiles (%d). "
                "Run: python3 scripts/generate_seed_data.py to regenerate embeddings and VAE."
                % (embeddings.shape[0], n_profiles)
            )
        self.vae = vae
        self.dataset = dataset
        self.embeddings = embeddings
        self._mus, self._sigmas = vae.encode_to_numpy(embeddings)
        self._name_to_idx = {p.name.lower(): i for i, p in enumerate(dataset)}

    def sample(self, temperature: float = 1.0, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        z = (rng.standard_normal(self.vae.latent_dim) * temperature).astype(np.float32)
        base_sigma = self._sigmas.mean(axis=0)
        sigma = base_sigma * (0.8 + 0.4 * rng.random(self.vae.latent_dim).astype(np.float32))
        return z, sigma

    def sample_near(self, character_name: str, radius: float = 0.5, rng=None):
        rng = rng or np.random.default_rng()
        idx = self._name_to_idx.get(character_name.lower())
        if idx is None:
            raise KeyError(f"Character '{character_name}' not found")
        mu, sigma = self._mus[idx], self._sigmas[idx]
        z = mu + rng.standard_normal(self.vae.latent_dim).astype(np.float32) * radius
        return z, sigma

    def interpolate(self, z1, z2, t):
        z1_n = z1 / (np.linalg.norm(z1) + 1e-9)
        z2_n = z2 / (np.linalg.norm(z2) + 1e-9)
        omega = np.arccos(np.clip(np.dot(z1_n, z2_n), -1.0, 1.0))
        if omega < 1e-6:
            return (1 - t) * z1 + t * z2
        so = np.sin(omega)
        return np.sin((1 - t) * omega) / so * z1 + np.sin(t * omega) / so * z2

    def nearest_characters(self, z, k=5):
        dists = np.linalg.norm(self._mus - z[np.newaxis, :], axis=1)
        indices = np.argsort(dists)[:k]
        return [(self.dataset[int(i)], float(dists[i])) for i in indices]

    def distance(self, z1, z2):
        return float(np.linalg.norm(z1 - z2))

    @property
    def latent_dim(self):
        return self.vae.latent_dim

    @property
    def num_characters(self):
        return len(self.dataset)
