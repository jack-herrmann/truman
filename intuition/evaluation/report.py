"""EvaluationReport — comprehensive evaluation."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from intuition.agent.agent import PersonalityAgent
from intuition.core.traces import BehavioralTrace
from intuition.evaluation.consistency import ConsistencyEvaluator, ConsistencyReport
from intuition.evaluation.individuality import IndividualityEvaluator, IndividualityReport
from intuition.llm.client import LLMClient
from intuition.llm.embeddings import EmbeddingClient
logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    consistency_reports: list[ConsistencyReport] = field(default_factory=list)
    individuality_report: IndividualityReport | None = None

    @property
    def mean_consistency(self):
        return sum(r.overall_score for r in self.consistency_reports)/max(1,len(self.consistency_reports))

    @property
    def overall_individuality(self):
        return self.individuality_report.overall_score if self.individuality_report else 0.0

    @property
    def combined_score(self):
        return 0.5 * self.mean_consistency + 0.5 * self.overall_individuality

    def summary(self):
        lines = [f"=== EVALUATION ===", f"Combined: {self.combined_score:.3f}",
                 f"Consistency: {self.mean_consistency:.3f}", f"Individuality: {self.overall_individuality:.3f}"]
        for cr in self.consistency_reports:
            lines.append(f"  {cr.summary()}")
        if self.individuality_report:
            lines.append(self.individuality_report.summary())
        return "\n".join(lines)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"combined": self.combined_score,
            "consistency": self.mean_consistency, "individuality": self.overall_individuality}, indent=2))


async def run_evaluation(agents, llm, embedding_client, traces_per_agent=None):
    report = EvaluationReport()
    ce = ConsistencyEvaluator(llm, embedding_client)
    for agent in agents:
        report.consistency_reports.append(await ce.evaluate(agent))
    if len(agents) >= 2:
        ie = IndividualityEvaluator(llm, embedding_client)
        report.individuality_report = await ie.evaluate(agents, traces_per_agent)
    return report
