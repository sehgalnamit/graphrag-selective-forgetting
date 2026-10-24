"""The jury: aggregates independent judge votes into one decision.

Rationale (honest, no fabricated benchmarks): a single "LLM-as-judge" call
is sensitive to that one model's idiosyncratic biases and prompt-sensitivity.
Polling a small, diverse *panel* of judges and aggregating their votes is a
well-established mitigation explored broadly in the LLM-evaluation
literature under names like "panel of judges" / "jury of models" - the
core idea being that uncorrelated errors across independent judges tend to
cancel out, and that panels of small/local models are cheaper and faster
than one large proprietary model per decision. We make no claims about
specific accuracy numbers; those depend entirely on which judges you plug
in and must be measured for your own data.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from evaluator.judges import (
    AggressiveForgetJudge,
    ComplianceOverrideJudge,
    ConservativeForgetJudge,
    Judge,
    LexicalOverlapRelevanceJudge,
    OllamaJudge,
)
from evaluator.schemas import JuryDecision, Verdict


@dataclass
class Jury:
    judges: list[Judge]
    abstain_margin: float = 0.15
    aggregation: str = "confidence_weighted"  # "confidence_weighted" | "majority"
    _decision_log: list[JuryDecision] = field(default_factory=list, repr=False)

    def decide(self, case: dict, allowed_verdicts: list[str], case_id: str = "") -> JuryDecision:
        if not self.judges:
            raise ValueError("Jury has no judges configured")
        votes: list[Verdict] = [judge.evaluate(case, allowed_verdicts) for judge in self.judges]

        if self.aggregation == "majority":
            final_verdict, confidence, escalated = self._majority(votes, allowed_verdicts)
        else:
            final_verdict, confidence, escalated = self._confidence_weighted(votes, allowed_verdicts)

        decision = JuryDecision(
            case_id=case_id,
            final_verdict=final_verdict,
            aggregated_confidence=confidence,
            method=self.aggregation,
            escalated=escalated,
            votes=votes,
        )
        self._decision_log.append(decision)
        return decision

    def _confidence_weighted(
        self, votes: list[Verdict], allowed_verdicts: list[str]
    ) -> tuple[str, float, bool]:
        scores = dict.fromkeys(allowed_verdicts, 0.0)
        for vote in votes:
            if vote.verdict in scores:
                scores[vote.verdict] += vote.confidence
        total = sum(scores.values())
        if total <= 0:
            return "abstain", 0.0, True
        normalized = {k: v / total for k, v in scores.items()}
        ranked = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
        top_verdict, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if (top_score - second_score) < self.abstain_margin:
            return "abstain", top_score, True
        return top_verdict, top_score, False

    def _majority(self, votes: list[Verdict], allowed_verdicts: list[str]) -> tuple[str, float, bool]:
        counts = Counter(v.verdict for v in votes if v.verdict in allowed_verdicts)
        if not counts:
            return "abstain", 0.0, True
        ranked = counts.most_common()
        top_verdict, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        confidence = top_count / len(votes)
        if top_count == second_count:
            return "abstain", confidence, True
        return top_verdict, confidence, False

    @property
    def decision_log(self) -> list[JuryDecision]:
        return list(self._decision_log)


def build_default_jury(settings) -> Jury:
    """Builds the panel used by the decay engine / retrieval-quality gate.

    Always includes the deterministic stub judges (so the jury works with
    zero external services). Optionally appends one :class:`OllamaJudge`
    per configured model when ``settings.jury_ollama_enabled`` is true.
    """
    judges: list[Judge] = [
        ConservativeForgetJudge(),
        AggressiveForgetJudge(),
        ComplianceOverrideJudge(),
    ]
    if getattr(settings, "jury_ollama_enabled", False):
        models = [m.strip() for m in getattr(settings, "jury_ollama_models", "").split(",") if m.strip()]
        for model in models:
            judges.append(OllamaJudge(model=model, base_url=settings.jury_ollama_url))
    return Jury(judges=judges, abstain_margin=getattr(settings, "jury_abstain_margin", 0.15))


def build_relevance_jury(settings) -> Jury:
    """A separate small panel specialised for retrieval-relevance checks."""
    judges: list[Judge] = [LexicalOverlapRelevanceJudge()]
    return Jury(judges=judges, abstain_margin=getattr(settings, "jury_abstain_margin", 0.15), aggregation="majority")
