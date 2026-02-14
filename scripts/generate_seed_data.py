#!/usr/bin/env python3
"""Generate seed embeddings and VAE checkpoint from shipped character profiles.

This script reads the character profiles in data/characters/, computes
embeddings using LocalEmbeddings (free, no API key needed), trains a
PersonalityVAE on those embeddings, and saves:

    data/characters/embeddings.npy   — embedding matrix (N × dim)
    data/checkpoints/vae.pt          — trained VAE checkpoint

Run once after cloning the repo to enable latent-space personality creation.
If the files already exist, the script will regenerate them.

Usage:
    python3 scripts/generate_seed_data.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from intuition.corpus.dataset import CharacterDataset
from intuition.llm.embeddings import LocalEmbeddings
from intuition.latent.vae import PersonalityVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load config.yaml from project root."""
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text())
    return {}


async def main() -> None:
    config = load_config()
    emb_cfg = config.get("embeddings", {})
    latent_cfg = config.get("latent", {})
    paths_cfg = config.get("paths", {})

    characters_dir = ROOT / paths_cfg.get("characters", "data/characters")
    checkpoints_dir = ROOT / paths_cfg.get("checkpoints", "data/checkpoints")

    emb_dim = emb_cfg.get("dimension", 512)
    latent_dim = latent_cfg.get("dimension", 32)
    hidden_dims = latent_cfg.get("hidden_dims", [512, 256])
    epochs = latent_cfg.get("epochs", 200)
    lr = latent_cfg.get("learning_rate", 0.001)
    kl_weight = latent_cfg.get("kl_weight", 0.001)

    # ── Step 1: Load character profiles ────────────────────────────────
    logger.info("Loading character profiles from %s", characters_dir)
    dataset = CharacterDataset(str(characters_dir))

    if len(dataset) == 0:
        logger.error("No character profiles found in %s", characters_dir)
        sys.exit(1)

    logger.info("Found %d character profiles", len(dataset))
    for p in dataset:
        logger.info("  • %s (%s)", p.name, p.novel)

    # ── Step 2: Compute embeddings ─────────────────────────────────────
    logger.info("Computing embeddings (LocalEmbeddings, dim=%d)...", emb_dim)
    embedder = LocalEmbeddings(dim=emb_dim)
    embeddings = await dataset.compute_embeddings(embedder)

    emb_path = dataset.save_embeddings(embeddings)
    logger.info("Saved embeddings → %s  (shape: %s)", emb_path, embeddings.shape)

    # ── Step 3: Train VAE ──────────────────────────────────────────────
    logger.info(
        "Training PersonalityVAE (input=%d, latent=%d, epochs=%d)...",
        emb_dim, latent_dim, epochs,
    )
    vae = PersonalityVAE(
        input_dim=emb_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
    history = vae.fit(
        embeddings,
        epochs=epochs,
        lr=lr,
        kl_weight=kl_weight,
    )

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    vae_path = checkpoints_dir / "vae.pt"
    vae.save(str(vae_path))
    logger.info("Saved VAE checkpoint → %s", vae_path)
    logger.info("Final training loss: %.6f", history[-1])

    # ── Step 4: Validate ───────────────────────────────────────────────
    logger.info("Validating: encoding profiles into latent space...")
    mus, sigmas = vae.encode_to_numpy(embeddings)
    logger.info("  Latent means  — shape: %s, range: [%.3f, %.3f]",
                mus.shape, mus.min(), mus.max())
    logger.info("  Latent sigmas — shape: %s, range: [%.3f, %.3f]",
                sigmas.shape, sigmas.min(), sigmas.max())

    # Reconstruction quality
    import torch
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32)
        recon, mu, log_var = vae(x)
        loss, recon_loss, kl_loss = vae.loss(recon, x, mu, log_var, kl_weight)
        logger.info("  Reconstruction loss: %.6f, KL loss: %.6f", recon_loss.item(), kl_loss.item())

    logger.info("")
    logger.info("✓ Seed data generation complete!")
    logger.info("  %d character profiles → embeddings.npy + vae.pt", len(dataset))
    logger.info("  You can now run: python3 demo.py")


if __name__ == "__main__":
    asyncio.run(main())
