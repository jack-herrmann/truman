#!/usr/bin/env python3
"""Run evaluation on trained personalities."""
import asyncio, logging
from pathlib import Path
import yaml
from rich.console import Console
from rich.logging import RichHandler
from intuition.agent.agent import PersonalityAgent
from intuition.core.kernel import PersonalityKernel
from intuition.environment.world import TrumanWorld
from intuition.evaluation.report import run_evaluation
from intuition.llm.client import LLMClient
from intuition.llm.embeddings import LocalEmbeddings, OpenAIEmbeddings
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(console=Console(stderr=True))])

async def main():
    config = yaml.safe_load(Path("config.yaml").read_text())
    llm = LLMClient(**config.get("llm", {}))
    emb_cfg = config.get("embeddings", {})
    emb = (OpenAIEmbeddings() if emb_cfg.get("provider") == "openai" else LocalEmbeddings(dim=emb_cfg.get("dimension", 512)))
    cp = Path(config.get("paths", {}).get("checkpoints", "data/checkpoints"))
    kernel_files = sorted(cp.glob("final_top*.json"))
    if not kernel_files:
        print("No trained personalities found.")
        return
    kernels = [PersonalityKernel.load(str(f)) for f in kernel_files]
    agents = [PersonalityAgent(k, llm) for k in kernels]
    world = TrumanWorld(llm)
    traces = [await world.run_evaluation_episode(a, 8) for a in agents]
    report = await run_evaluation(agents, llm, emb, traces)
    print(report.summary())
    report.save(str(cp / "evaluation_report.json"))

if __name__ == "__main__":
    asyncio.run(main())
