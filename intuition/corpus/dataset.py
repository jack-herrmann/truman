"""CharacterDataset — collection of extracted character profiles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from intuition.corpus.extractor import CharacterProfile
from intuition.llm.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class CharacterDataset:
    def __init__(self, data_dir: str = "data/characters") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: list[CharacterProfile] | None = None

    def add(self, profile: CharacterProfile) -> None:
        safe_name = f"{profile.novel}_{profile.name}".lower().replace(" ", "_").replace("'", "")
        path = self.data_dir / f"{safe_name}.json"
        profile.save(str(path))
        self._profiles = None

    def _load_all(self) -> list[CharacterProfile]:
        profiles = []
        for path in sorted(self.data_dir.glob("*.json")):
            try:
                profiles.append(CharacterProfile.load(str(path)))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path.name, exc)
        return profiles

    @property
    def profiles(self) -> list[CharacterProfile]:
        if self._profiles is None:
            self._profiles = self._load_all()
        return self._profiles

    def __len__(self) -> int:
        return len(self.profiles)

    def __iter__(self) -> Iterator[CharacterProfile]:
        return iter(self.profiles)

    def __getitem__(self, idx: int) -> CharacterProfile:
        return self.profiles[idx]

    async def compute_embeddings(self, client: EmbeddingClient) -> np.ndarray:
        texts = [p.personality_analysis for p in self.profiles]
        if not texts:
            raise ValueError("No profiles in dataset")
        return await client.embed_batch(texts)

    def save_embeddings(self, embeddings: np.ndarray, path: str | None = None) -> Path:
        dest = Path(path) if path else self.data_dir / "embeddings.npy"
        np.save(str(dest), embeddings)
        return dest

    def load_embeddings(self, path: str | None = None) -> np.ndarray:
        src = Path(path) if path else self.data_dir / "embeddings.npy"
        if not src.exists():
            raise FileNotFoundError(f"No embeddings at {src}")
        return np.load(str(src))

    def find_by_name(self, name: str) -> CharacterProfile | None:
        name_lower = name.lower()
        for p in self.profiles:
            if p.name.lower() == name_lower:
                return p
        return None
