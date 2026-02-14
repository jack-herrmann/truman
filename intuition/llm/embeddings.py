"""Text embedding backends."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingClient(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray: ...

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        vectors = [await self.embed(t) for t in texts]
        return np.stack(vectors)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return dot / norm if norm > 0 else 0.0

    @staticmethod
    def pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = matrix / norms
        return normed @ normed.T


class OpenAIEmbeddings(EmbeddingClient):
    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        self._model = model
        self._dim = dim
        self._client = None

    @property
    def dimension(self) -> int:
        return self._dim

    def _ensure_client(self):
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI()

    async def embed(self, text: str) -> np.ndarray:
        self._ensure_client()
        response = await self._client.embeddings.create(model=self._model, input=text)
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        self._ensure_client()
        response = await self._client.embeddings.create(model=self._model, input=texts)
        vecs = np.array([d.embedding for d in response.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs / norms


class LocalEmbeddings(EmbeddingClient):
    """Deterministic hash-based embeddings for offline development."""

    def __init__(self, dim: int = 512):
        self._dim = dim
        self._hasher = None

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, text: str) -> np.ndarray:
        return self._hash_embed(text)

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._hash_embed(t) for t in texts])

    def _hash_embed(self, text: str) -> np.ndarray:
        try:
            if self._hasher is None:
                from sklearn.feature_extraction.text import HashingVectorizer
                self._hasher = HashingVectorizer(n_features=self._dim, alternate_sign=False, norm="l2")
            vec = self._hasher.transform([text]).toarray()[0]
            norm = np.linalg.norm(vec)
            return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)
        except ImportError:
            pass
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)
