#!/usr/bin/env python3
"""Run the personality training loop."""
import asyncio, logging
from pathlib import Path
import yaml
from rich.console import Console
from rich.logging import RichHandler
from intuition.corpus.dataset import CharacterDataset
from intuition.latent.decoder import PersonalityDecoder
from intuition.latent.space import PersonalitySpace
from intuition.latent.vae import PersonalityVAE
from intuition.llm.client import LLMClient
from intuition.llm.embeddings import LocalEmbeddings, OpenAIEmbeddings
from intuition.training.optimizer import KernelOptimizer
from intuition.training.trainer import TrainingLoop
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(console=Console(stderr=True))])

async def main():
    config = yaml.safe_load(Path("config.yaml").read_text())
    llm = LLMClient(**config.get("llm", {}))
    emb_cfg = config.get("embeddings", {})
    emb = (OpenAIEmbeddings() if emb_cfg.get("provider") == "openai" else LocalEmbeddings(dim=emb_cfg.get("dimension", 512)))
    cp = Path(config.get("paths", {}).get("checkpoints", "data/checkpoints"))
    vae = PersonalityVAE.load(str(cp / "vae.pt"))
    dataset = CharacterDataset()
    space = PersonalitySpace(vae, dataset, dataset.load_embeddings())
    decoder = PersonalityDecoder(space, llm)
    tc = config.get("training", {})
    optimizer = KernelOptimizer(space, decoder, population_size=tc.get("population_size", 12))
    trainer = TrainingLoop(llm, emb, optimizer, episode_length=tc.get("episode_length", 12), save_dir=str(cp))
    await trainer.train(num_generations=tc.get("num_generations", 10))

if __name__ == "__main__":
    asyncio.run(main())
