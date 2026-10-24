"""Jury aggregation logic tests using deterministic stub judges (no network,
no model downloads - see evaluator/judges.py docstring for rationale)."""
from __future__ import annotations

from evaluator.judges import (
    AggressiveForgetJudge,
    ComplianceOverrideJudge,
    ConservativeForgetJudge,
    LexicalOverlapRelevanceJudge,
)
from evaluator.jury import Jury, build_default_jury, build_relevance_jury
from evaluator.schemas import Verdict


class FixedJudge:
    def __init__(self, name, verdict, confidence):
        self.name = name
        self._verdict = verdict
        self._confidence = confidence

    def evaluate(self, case, allowed_verdicts):
        return Verdict(judge_name=self.name, verdict=self._verdict, confidence=self._confidence, rationale="fixed")


def test_unanimous_verdict_wins_with_no_escalation():
    jury = Jury(judges=[FixedJudge("a", "keep", 0.9), FixedJudge("b", "keep", 0.8), FixedJudge("c", "keep", 0.95)])
    decision = jury.decide({}, allowed_verdicts=["keep", "forget"], case_id="n1")
    assert decision.final_verdict == "keep"
    assert decision.escalated is False
    assert len(decision.votes) == 3


def test_confidence_weighted_majority_wins_over_single_dissent():
    jury = Jury(
        judges=[FixedJudge("a", "forget", 0.9), FixedJudge("b", "forget", 0.85), FixedJudge("c", "keep", 0.6)],
        abstain_margin=0.15,
    )
    decision = jury.decide({}, allowed_verdicts=["keep", "forget"], case_id="n2")
    assert decision.final_verdict == "forget"
    assert decision.escalated is False


def test_close_split_vote_abstains_and_escalates():
    jury = Jury(
        judges=[FixedJudge("a", "keep", 0.55), FixedJudge("b", "forget", 0.55)],
        abstain_margin=0.15,
    )
    decision = jury.decide({}, allowed_verdicts=["keep", "forget"], case_id="n3")
    assert decision.final_verdict == "abstain"
    assert decision.escalated is True


def test_majority_aggregation_tie_abstains():
    jury = Jury(
        judges=[FixedJudge("a", "keep", 0.9), FixedJudge("b", "forget", 0.9)],
        aggregation="majority",
    )
    decision = jury.decide({}, allowed_verdicts=["keep", "forget"], case_id="n4")
    assert decision.final_verdict == "abstain"
    assert decision.escalated is True


def test_majority_aggregation_clear_winner():
    jury = Jury(
        judges=[
            FixedJudge("a", "forget", 0.6),
            FixedJudge("b", "forget", 0.55),
            FixedJudge("c", "keep", 0.9),
        ],
        aggregation="majority",
    )
    decision = jury.decide({}, allowed_verdicts=["keep", "forget"], case_id="n5")
    assert decision.final_verdict == "forget"
    assert decision.escalated is False
    assert decision.aggregated_confidence == 2 / 3


def test_decision_log_accumulates_across_calls():
    jury = Jury(judges=[FixedJudge("a", "keep", 0.9)])
    jury.decide({}, allowed_verdicts=["keep", "forget"], case_id="x")
    jury.decide({}, allowed_verdicts=["keep", "forget"], case_id="y")
    assert [d.case_id for d in jury.decision_log] == ["x", "y"]


# --- stub judge behavior (deterministic, rule-based) ---------------------


def test_conservative_judge_keeps_when_recently_reinforced():
    judge = ConservativeForgetJudge()
    case = {"decay_score": 0.1, "prune_threshold": 0.2, "access_count": 5, "text": ""}
    verdict = judge.evaluate(case, ["keep", "forget"])
    assert verdict.verdict == "keep"


def test_aggressive_judge_forgets_near_threshold():
    judge = AggressiveForgetJudge()
    case = {"decay_score": 0.18, "prune_threshold": 0.2, "access_count": 0, "text": ""}
    verdict = judge.evaluate(case, ["keep", "forget"])
    assert verdict.verdict == "forget"


def test_compliance_judge_overrides_decay_for_compliance_text():
    judge = ComplianceOverrideJudge()
    case = {"decay_score": 0.01, "prune_threshold": 0.2, "text": "This GDPR compliance policy document..."}
    verdict = judge.evaluate(case, ["keep", "forget"])
    assert verdict.verdict == "keep"
    assert verdict.confidence >= 0.9


def test_relevance_judge_flags_overlapping_text_as_relevant():
    judge = LexicalOverlapRelevanceJudge(relevance_token_overlap_ratio=0.3)
    case = {"query": "quarterly revenue report", "chunk_text": "The quarterly revenue grew 10%."}
    verdict = judge.evaluate(case, ["relevant", "irrelevant"])
    assert verdict.verdict == "relevant"


def test_relevance_judge_flags_unrelated_text_as_irrelevant():
    judge = LexicalOverlapRelevanceJudge(relevance_token_overlap_ratio=0.3)
    case = {"query": "quarterly revenue report", "chunk_text": "The weather today is sunny and warm."}
    verdict = judge.evaluate(case, ["relevant", "irrelevant"])
    assert verdict.verdict == "irrelevant"


def test_build_default_jury_has_deterministic_stub_panel(settings):
    jury = build_default_jury(settings)
    assert len(jury.judges) == 3  # no Ollama models configured in test settings
    names = {j.name for j in jury.judges}
    assert "conservative_forget_judge" in names
    assert "aggressive_forget_judge" in names
    assert "compliance_override_judge" in names


def test_build_relevance_jury_runs_end_to_end(settings):
    jury = build_relevance_jury(settings)
    decision = jury.decide(
        {"query": "revenue", "chunk_text": "annual revenue results"},
        allowed_verdicts=["relevant", "irrelevant"],
        case_id="chunk-1",
    )
    assert decision.final_verdict in {"relevant", "irrelevant"}
