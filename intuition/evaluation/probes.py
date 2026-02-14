"""PersonalityProbe — assessment instruments for personality measurement."""
from __future__ import annotations
import logging
from collections import defaultdict
from pydantic import BaseModel, Field
from intuition.llm.client import LLMClient
logger = logging.getLogger(__name__)


class ProbeQuestion(BaseModel):
    id: str
    text: str
    dimension: str
    variants: list[str] = Field(default_factory=list)
    scale: str = "agree_disagree"


class ProbeAnswer(BaseModel):
    question_id: str
    answer: str
    score: float = Field(ge=-1.0, le=1.0)


class ProbeResult(BaseModel):
    probe_name: str
    answers: list[ProbeAnswer] = Field(default_factory=list)
    dimension_scores: dict[str, float] = Field(default_factory=dict)


class PersonalityProbe:
    def __init__(self, name: str, questions: list[ProbeQuestion], llm: LLMClient) -> None:
        self.name = name
        self.questions = questions
        self.llm = llm

    async def administer(self, agent_system_prompt: str, variation: int = 0) -> ProbeResult:
        answers = []
        for q in self.questions:
            text = q.variants[(variation-1) % len(q.variants)] if variation > 0 and q.variants else q.text
            prompt = f"{text}\n\nRespond naturally as yourself. Be honest."
            response = await self.llm.generate(system=agent_system_prompt,
                messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=200)
            score = await self._score_response(response, q.dimension)
            answers.append(ProbeAnswer(question_id=q.id, answer=response.strip()[:200], score=score))
        dim_scores: dict[str, list[float]] = defaultdict(list)
        for a in answers:
            dim_scores[a.question_id.split("_")[0]].append(a.score)
        return ProbeResult(probe_name=self.name, answers=answers,
            dimension_scores={d: sum(s)/len(s) for d, s in dim_scores.items()})

    async def _score_response(self, response: str, dimension: str) -> float:
        try:
            r = await self.llm.generate(
                system="Psychometric scoring system.",
                messages=[{"role":"user","content":f"Score on '{dimension}' from -1.0 to 1.0:\n{response[:300]}\nNumber only."}],
                temperature=0.1, max_tokens=10)
            return max(-1.0, min(1.0, float(r.strip().split()[0])))
        except (ValueError, IndexError):
            return 0.0


def _big_five_questions():
    items = {
        "openness": [("o1","I enjoy exploring new ideas.",["I'm drawn to novel thinking.","Unfamiliar approaches appeal to me."]),
                      ("o2","I prefer routine over novelty.",["I like knowing what to expect.","Familiar patterns comfort me."]),
                      ("o3","Art and beauty move me deeply.",["Aesthetic experiences affect me.","Beautiful things resonate emotionally."]),
                      ("o4","I think about abstract concepts.",["I ponder deep questions.","My mind gravitates to the abstract."])],
        "conscientiousness": [("c1","I plan ahead and follow through.",["I tend to be organised.","When I commit, I deliver."]),
                              ("c2","I often act on impulse.",["I sometimes leap before looking.","Spontaneity gets the better of me."]),
                              ("c3","I pay close attention to details.",["Getting things right matters.","I notice small errors."]),
                              ("c4","I work steadily toward goals.",["I push through tedium.","Persistence is my strength."])],
        "extraversion": [("e1","I feel energised in groups.",["Social gatherings give me energy.","Being around others charges me."]),
                         ("e2","I prefer deep one-on-one talks.",["Intimate conversations over parties.","I'd rather talk deeply with one person."]),
                         ("e3","I speak up in group settings.",["I voice my opinion early.","I take an active vocal role."]),
                         ("e4","I need alone time to recharge.",["Solitude is essential.","I need to be alone regularly."])],
        "agreeableness": [("a1","I trust people's intentions.",["I give benefit of the doubt.","I assume good faith."]),
                          ("a2","I challenge positions I disagree with.",["I push back when needed.","Disagreement doesn't bother me."]),
                          ("a3","Helping others comes naturally.",["I put others' needs first.","Self-sacrifice feels instinctive."]),
                          ("a4","I prioritise honesty over kindness.",["Uncomfortable truth over comforting lie.","Honesty matters most."])],
        "neuroticism": [("n1","I worry about things going wrong.",["I anticipate problems.","I'm often anxious about the future."]),
                        ("n2","I stay calm under pressure.",["Stress doesn't rattle me.","I bounce back from setbacks."]),
                        ("n3","My mood shifts from small events.",["Little things affect my mood.","My emotions are responsive."]),
                        ("n4","I experience intense emotions.",["Strong feelings overwhelm me.","I feel things powerfully."])],
    }
    questions = []
    for dim, dim_items in items.items():
        for qid, text, variants in dim_items:
            questions.append(ProbeQuestion(id=f"{dim}_{qid}", text=text, dimension=dim, variants=variants))
    return questions


def _moral_foundations_questions():
    scenarios = [
        ("care","mf_care1","A friend is struggling financially but hiding it. Offer help risking embarrassment?"),
        ("care","mf_care2","Injured stray animal. Stopping makes you late for something important."),
        ("fairness","mf_fair1","A colleague who works less gets the same bonus. How do you feel?"),
        ("fairness","mf_fair2","You find a wallet with cash and ID. What do you do?"),
        ("loyalty","mf_loyal1","A close friend asks you to lie about something minor."),
        ("loyalty","mf_loyal2","Your group is excluding someone annoying. Go along or intervene?"),
        ("authority","mf_auth1","A pointless rule at work. Manager insists. How do you handle it?"),
        ("authority","mf_auth2","An elder gives advice contradicting your experience. Your response?"),
        ("purity","mf_pure1","Someone suggests a legal but ethically grey shortcut. Your reaction?"),
        ("purity","mf_pure2","Something you enjoy was produced in a morally questionable way."),
    ]
    return [ProbeQuestion(id=qid, text=text, dimension=dim, scale="choice") for dim, qid, text in scenarios]


class ProbeBattery:
    def __init__(self, llm: LLMClient) -> None:
        self.probes = {
            "big_five": PersonalityProbe("big_five", _big_five_questions(), llm),
            "moral_foundations": PersonalityProbe("moral_foundations", _moral_foundations_questions(), llm),
        }

    async def full_assessment(self, agent_system_prompt: str, repetitions: int = 3):
        results: dict[str, list[ProbeResult]] = {}
        for name, probe in self.probes.items():
            results[name] = [await probe.administer(agent_system_prompt, variation=r) for r in range(repetitions)]
        return results
