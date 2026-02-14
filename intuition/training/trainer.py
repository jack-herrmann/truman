"""TrainingLoop — the complete personality training orchestrator."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from intuition.agent.agent import PersonalityAgent
from intuition.core.kernel import PersonalityKernel
from intuition.core.traces import BehavioralTrace
from intuition.environment.world import TrumanWorld
from intuition.llm.client import LLMClient
from intuition.llm.embeddings import EmbeddingClient
from intuition.training.optimizer import KernelOptimizer
from intuition.training.rewards import CoherenceReward, IndividualityReward
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    generation: int
    coherence_scores: dict[str, float]
    individuality_scores: dict[str, float]
    combined_scores: dict[str, float]
    best_kernel_id: str
    best_combined_score: float


@dataclass
class TrainingResult:
    generations: list[GenerationResult] = field(default_factory=list)
    best_kernels: list[PersonalityKernel] = field(default_factory=list)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "num_generations": len(self.generations),
            "generations": [{"generation": g.generation, "best_score": g.best_combined_score}
                            for g in self.generations]}, indent=2))


class TrainingLoop:
    def __init__(self, llm: LLMClient, embedding_client: EmbeddingClient,
                 optimizer: KernelOptimizer, episode_length: int = 12,
                 coherence_weight: float = 0.5, individuality_weight: float = 0.5,
                 save_dir: str = "data/checkpoints") -> None:
        self.llm = llm
        self.optimizer = optimizer
        self.episode_length = episode_length
        self.coherence_weight = coherence_weight
        self.individuality_weight = individuality_weight
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.coherence_reward = CoherenceReward(embedding_client, llm)
        self.individuality_reward = IndividualityReward(embedding_client)
        self.world = TrumanWorld(llm)

    async def train(self, num_generations: int = 10) -> TrainingResult:
        result = TrainingResult()
        kernels = await self.optimizer.initialise()
        for gen in range(num_generations):
            agents = [PersonalityAgent(k, self.llm) for k in kernels]
            traces_per_agent = [await self.world.run_episode(a, self.episode_length) for a in agents]
            coh = {a.kernel.id: await self.coherence_reward.compute(a, t) for a, t in zip(agents, traces_per_agent)}
            per_ind = await self.individuality_reward.compute_per_agent(traces_per_agent)
            ind = {agents[i].kernel.id: per_ind[i] for i in range(len(agents))}
            self.optimizer.update_scores(coh, ind, self.coherence_weight, self.individuality_weight)
            best = self.optimizer.best
            result.generations.append(GenerationResult(
                generation=gen+1, coherence_scores=coh, individuality_scores=ind,
                combined_scores={k: self.coherence_weight*coh[k]+self.individuality_weight*ind.get(k,0) for k in coh},
                best_kernel_id=best.kernel.id if best else "",
                best_combined_score=best.combined if best else 0.0))
            if best:
                best.kernel.save(str(self.save_dir / f"gen{gen+1}_best.json"))
            if gen < num_generations - 1:
                kernels = await self.optimizer.evolve()
        result.best_kernels = [sk.kernel for sk in sorted(self.optimizer.population, key=lambda x: x.combined, reverse=True)[:5]]
        for i, k in enumerate(result.best_kernels):
            k.save(str(self.save_dir / f"final_top{i+1}.json"))
        result.save(str(self.save_dir / "training_results.json"))
        return result
