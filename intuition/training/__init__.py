"""Training loop — coherence + individuality optimisation."""
from intuition.training.rewards import CoherenceReward, IndividualityReward
from intuition.training.discriminator import TraceDiscriminator
from intuition.training.optimizer import KernelOptimizer
from intuition.training.trainer import TrainingLoop
__all__ = ["CoherenceReward","IndividualityReward","TraceDiscriminator","KernelOptimizer","TrainingLoop"]
