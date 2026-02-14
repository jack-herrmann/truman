#!/usr/bin/env python3
"""Train the personality VAE on character profile embeddings."""
import asyncio, logging
from pathlib import Path
import yaml
from rich.console import Console
from rich.logging import RichHandler
from intuition.corpus.dataset import CharacterDataset
from intuition.latent.vae import PersonalityVAE
from intuition.llm.embeddings import LocalEmbeddings, OpenAIEmbeddings
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(console=Console(stderr=True))])

async def main():
    config = yaml.safe_load(Path("config.yaml").read_text())
    dataset = CharacterDataset("data/characters")
    emb_cfg = config.get("embeddings", {})
    emb_client = (OpenAIEmbeddings(model=emb_cfg.get("model","text-embedding-3-small"))
                  if emb_cfg.get("provider") == "openai" else LocalEmbeddings(dim=emb_cfg.get("dimension", 512)))
    emb_path = Path("data/characters/embeddings.npy")
    if emb_path.exists():
        embeddings = dataset.load_embeddings()
    else:
        embeddings = await dataset.compute_embeddings(emb_client)
        dataset.save_embeddings(embeddings)
    lat = config.get("latent", {})
    vae = PersonalityVAE(input_dim=embeddings.shape[1], latent_dim=lat.get("dimension", 32),
                          hidden_dims=lat.get("hidden_dims", [512, 256]))
    vae.fit(embeddings, epochs=lat.get("epochs", 200), lr=lat.get("learning_rate", 0.001),
            kl_weight=lat.get("kl_weight", 0.001))
    cp = Path(config.get("paths", {}).get("checkpoints", "data/checkpoints"))
    cp.mkdir(parents=True, exist_ok=True)
    vae.save(str(cp / "vae.pt"))
    print(f"VAE saved to {cp / 'vae.pt'}")

if __name__ == "__main__":
    asyncio.run(main())
