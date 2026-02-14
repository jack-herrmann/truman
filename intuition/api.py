"""Public API for creating and using personalities."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from intuition.agent.agent import PersonalityAgent
from intuition.core.kernel import PersonalityKernel
from intuition.llm.client import LLMClient
logger = logging.getLogger(__name__)

_llm: LLMClient | None = None
_config: dict | None = None


def _load_config() -> dict:
    global _config
    if _config is not None:
        return _config
    for p in [Path("config.yaml"), Path(__file__).resolve().parent.parent / "config.yaml"]:
        if p.exists():
            _config = yaml.safe_load(p.read_text())
            return _config
    _config = {"llm": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}, "latent": {"dimension": 32}}
    return _config


def _get_llm() -> LLMClient:
    global _llm
    if _llm is not None:
        return _llm
    c = _load_config().get("llm", {})
    _llm = LLMClient(provider=c.get("provider","anthropic"), model=c.get("model","claude-sonnet-4-20250514"),
                      max_tokens=c.get("max_tokens",4096), temperature=c.get("temperature",0.7))
    return _llm


async def create_personality(z=None, like=None, variation=0.5, temperature=1.0, llm=None):
    client = llm or _get_llm()
    config = _load_config()
    latent_dim = config.get("latent", {}).get("dimension", 32)
    try:
        return await _create_from_latent_space(z, like, variation, temperature, client)
    except Exception:
        pass
    return await _create_direct(z, latent_dim, temperature, client)


async def _create_from_latent_space(z, like, variation, temperature, llm):
    from intuition.corpus.dataset import CharacterDataset
    from intuition.latent.decoder import PersonalityDecoder
    from intuition.latent.space import PersonalitySpace
    from intuition.latent.vae import PersonalityVAE
    config = _load_config()
    vae_path = Path(config.get("paths",{}).get("checkpoints","data/checkpoints")) / "vae.pt"
    if not vae_path.exists():
        raise FileNotFoundError("VAE not trained")
    vae = PersonalityVAE.load(str(vae_path))
    dataset = CharacterDataset()
    embeddings = dataset.load_embeddings()
    space = PersonalitySpace(vae, dataset, embeddings)
    decoder = PersonalityDecoder(space, llm)
    if z is not None:
        z_arr = np.array(z, dtype=np.float32)
        _, sigma = space.sample()
        return await decoder.decode(z_arr, sigma)
    elif like is not None:
        z_arr, sigma = space.sample_near(like, radius=variation)
        return await decoder.decode(z_arr, sigma)
    else:
        z_arr, sigma = space.sample(temperature=temperature)
        return await decoder.decode(z_arr, sigma)


async def _create_direct(z, latent_dim, temperature, llm):
    rng = np.random.default_rng()
    z_arr = np.array(z, dtype=np.float32) if z is not None else (rng.standard_normal(latent_dim)*temperature).astype(np.float32)
    sigma = (0.3 + 0.4 * rng.random(latent_dim)).astype(np.float32)
    kernel = await llm.generate_structured(
        system=("You are a personality architect. Generate a deeply coherent, richly individual personality. "
                "Every trait connects to something deeper. Include contradictions. Be specific, not generic."),
        messages=[{"role":"user","content":"Create a complete, unique personality. 3-5 values, 2+ fault lines, "
                   "specific stress profile, 2-3 paragraph summary, formative experiences, perceptual style, name."}],
        response_model=PersonalityKernel, temperature=0.8)
    kernel.z = z_arr.tolist()
    kernel.sigma = sigma.tolist()
    return kernel


async def create_agent(kernel, llm=None):
    return PersonalityAgent(kernel, llm or _get_llm())


def save_personality(kernel, path):
    kernel.save(path)


def load_personality(path):
    return PersonalityKernel.load(path)
