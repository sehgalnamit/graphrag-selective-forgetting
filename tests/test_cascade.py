"""Cascade routing tests for ConfidenceGatedCascade (TypeSafe's documented
"Confidence-Gated Routing" pattern): high-confidence Jev answers should act
automatically without ever calling the jury; ambiguous or high-stakes cases
should escalate to the jury; jury abstention should enqueue human review.
All fully offline (stub Jev clients + stub judges, no network)."""
from __future__ import annotations

from evaluator.human_review import HumanReviewQueue
from evaluator.jev_gate import CascadeConfig, ConfidenceGatedCascade, NoulAnswer
from evaluator.jury import Jury
from evaluator.schemas import Verdict


class FixedNoulJevClient:
    """Stub JevClient that always answers a Noul question with a fixed
    probability, regardless of state/question content."""

    def __init__(self, noul: float):
        self._noul = noul
        self.call_count = 0

    def ask(self, state, questions):
        self.call_count += 1
        return {qid: NoulAnswer(noul=self._noul) for qid in questions}


class CountingJury(Jury):
    """A Jury subclass that tracks how many times it was actually invoked,
    so tests can assert the jury was (or was not) called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    def decide(self, case, allowed_verdicts, case_id=""):
        self.call_count += 1
        return super().decide(case, allowed_verdicts, case_id)


def _fixed_judge(verdict, confidence=0.9):
    class _Judge:
        name = f"fixed_{verdict}"

        def evaluate(self, case, allowed_verdicts):
            return Verdict(judge_name=self.name, verdict=verdict, confidence=confidence, rationale="fixed")

    return _Judge()


def test_high_confidence_forget_acts_automatically_without_calling_jury():
    jev = FixedNoulJevClient(noul=0.95)
    jury = CountingJury(judges=[_fixed_judge("keep")])
    cascade = ConfidenceGatedCascade(jev_client=jev, jury=jury)

    decision = cascade.decide_forget_vs_keep(case_id="n1", state={}, jury_case={}, is_high_stakes=False)

    assert decision.acted_automatically is True
    assert decision.forget is True
    assert decision.escalated_to_jury is False
    assert jury.call_count == 0
    assert jev.call_count == 1


def test_high_confidence_keep_acts_automatically_without_calling_jury():
    jev = FixedNoulJevClient(noul=0.05)
    jury = CountingJury(judges=[_fixed_judge("forget")])
    cascade = ConfidenceGatedCascade(jev_client=jev, jury=jury)

    decision = cascade.decide_forget_vs_keep(case_id="n2", state={}, jury_case={}, is_high_stakes=False)

    assert decision.acted_automatically is True
    assert decision.forget is False
    assert jury.call_count == 0


def test_ambiguous_noul_escalates_to_jury():
    jev = FixedNoulJevClient(noul=0.5)
    jury = CountingJury(judges=[_fixed_judge("forget", 0.9), _fixed_judge("forget", 0.8)])
    cascade = ConfidenceGatedCascade(jev_client=jev, jury=jury, config=CascadeConfig(ambiguous_band=(0.35, 0.65)))

    decision = cascade.decide_forget_vs_keep(case_id="n3", state={}, jury_case={}, is_high_stakes=False)

    assert decision.escalated_to_jury is True
    assert jury.call_count == 1
    assert decision.forget is True
    assert decision.jury_decision is not None
    assert decision.jury_decision.final_verdict == "forget"


def test_high_stakes_escalates_to_jury_even_when_jev_is_confident():
    jev = FixedNoulJevClient(noul=0.98)  # very confident "forget"
    jury = CountingJury(judges=[_fixed_judge("keep", 0.9)])
    cascade = ConfidenceGatedCascade(jev_client=jev, jury=jury)

    decision = cascade.decide_forget_vs_keep(case_id="n4", state={}, jury_case={}, is_high_stakes=True)

    assert decision.escalated_to_jury is True
    assert jury.call_count == 1
    assert decision.forget is False  # jury overturned Jev's confident forget vote


def test_high_stakes_flag_can_be_disabled_via_config():
    jev = FixedNoulJevClient(noul=0.98)
    jury = CountingJury(judges=[_fixed_judge("keep")])
    cascade = ConfidenceGatedCascade(
        jev_client=jev, jury=jury, config=CascadeConfig(always_escalate_high_stakes=False)
    )

    decision = cascade.decide_forget_vs_keep(case_id="n5", state={}, jury_case={}, is_high_stakes=True)

    assert decision.escalated_to_jury is False
    assert jury.call_count == 0


def test_jury_abstention_enqueues_human_review():
    jev = FixedNoulJevClient(noul=0.5)  # ambiguous -> always escalates
    jury = CountingJury(
        judges=[_fixed_judge("keep", 0.55), _fixed_judge("forget", 0.55)],
        abstain_margin=0.9,
    )
    queue = HumanReviewQueue(":memory:")
    cascade = ConfidenceGatedCascade(jev_client=jev, jury=jury, human_review_queue=queue)

    decision = cascade.decide_forget_vs_keep(
        case_id="n6", state={}, jury_case={"tenant_id": "acme"}, is_high_stakes=False
    )

    assert decision.jury_decision.escalated is True
    assert decision.forget is False  # "do not act" when even the jury can't agree
    pending = queue.list_pending(tenant_id="acme")
    assert len(pending) == 1
    assert pending[0].node_id == "n6"
    assert pending[0].case_type == "jev_cascade_jury_abstain"


def test_escalation_without_a_configured_jury_defaults_to_conservative_keep():
    jev = FixedNoulJevClient(noul=0.5)
    cascade = ConfidenceGatedCascade(jev_client=jev, jury=None)

    decision = cascade.decide_forget_vs_keep(case_id="n7", state={}, jury_case={}, is_high_stakes=False)

    assert decision.escalated_to_jury is False
    assert decision.acted_automatically is False
    assert decision.forget is False
    assert decision.jury_decision is None
