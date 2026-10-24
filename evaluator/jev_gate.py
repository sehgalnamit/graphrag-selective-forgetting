"""TypeSafe "Jev" (System One model) integration + Confidence-Gated Routing.

This module implements the **real, documented** TypeSafe Jev API/SDK
contract (verified against the live docs at the URLs below - nothing in
this module is invented beyond what those docs specify):

* Product/intro:               https://typesafe.ai/
* API reference:                https://docs.typesafe.ai/introduction
* Confidence formulas:           https://docs.typesafe.ai/confidence
* Confidence-Gated Routing:      https://docs.typesafe.ai/patterns/confidence-routing
* DeepEval's JevEval integration (weighted combination inspiration):
                                 https://deepeval.com/blog/introducing-jev-as-a-judge

REST contract (``TypeSafeJevClient``):
    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <TYPESAFE_API_KEY>
    Content-Type: application/json
    {"state": <string|object>, "model": "jev-latest",
     "questions": {"<id>": {"type": "noul"|"choice"|"score", ...}}}

Three typed primitives:
    * **Noul** - yes/no proposition -> ``{"type": "noul", "noul": <0..1>}``.
      There is no separate confidence field for Noul: distance from 0.5
      *is* the confidence signal.
    * **Choice** - pick one of an unordered set of named options ->
      ``{"type": "choice", "choice": ..., "probabilities": {...}, "confidence": ...}``.
      ``confidence = (n * p_max - 1) / (n - 1)`` for ``n`` options.
    * **Score** - an ordered rubric ->
      ``{"type": "score", "score": ..., "probabilities": {...}, "confidence": ...}``.
      ``score = sum(k * p_k for k in 0..n-1)`` (expected rubric position);
      ``confidence = max(0, 1 - MAD(p) / MAD(uniform))`` where MAD is the
      mean absolute deviation of the probability mass around the
      highest-probability ("mode") index - 1.0 is fully concentrated on
      one level, 0.0 is spread perfectly uniformly.

The official Python SDK (``pip install typesafe-sdk``,
``from typesafe_sdk import Choice, Noul, Score, TypeSafeClient``) wraps the
exact same REST contract; this repo calls the endpoint directly via
``httpx`` (already a hard dependency here) so `typesafe-sdk` stays an
**optional** extra, never a hard requirement - see ``requirements.txt``.

TypeSafe's own "Confidence-Gated Routing" pattern
(https://docs.typesafe.ai/patterns/confidence-routing) is implemented by
:class:`ConfidenceGatedCascade`: confidence is a *second* decision axis
alongside the verdict itself - "a confidence threshold is not one
number... different actions [are] gated at different levels depending on
the consequences of getting it wrong" (TypeSafe). Here: high confidence ->
act automatically; medium/low confidence (or a high-stakes item) -> escalate
to this repo's existing LLM-jury panel (``evaluator.jury.Jury``) for a
slower, more deliberate second opinion.

TypeSafe reports Jev as dramatically faster/cheaper than frontier LLMs on
their own workflows (they describe multipliers in the tens-to-hundreds-x
range). That is **TypeSafe's own, unverified-by-us claim** on their own
benchmarks and is not repeated here as fact - see the README's Evaluator
section for the exact caveat.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from evaluator.jury import Jury
from evaluator.schemas import JuryDecision

# ---------------------------------------------------------------------------
# Typed question / answer primitives (request & response contracts)
# ---------------------------------------------------------------------------


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict = Field(default_factory=dict)


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]  # option_id -> description


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]  # ordered, lowest-to-highest rubric level descriptions


JevQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)

    @property
    def confidence(self) -> float:
        """The real API returns no separate confidence field for Noul
        answers; per https://docs.typesafe.ai/confidence, distance from 0.5
        *is* the confidence signal (0.5 => 0.0 confidence / maximal
        ambiguity; 0.0 or 1.0 => 1.0 confidence)."""
        return abs(self.noul - 0.5) * 2.0


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


JevAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer


# ---------------------------------------------------------------------------
# Confidence formulas (standalone, unit-testable against the documented spec)
# ---------------------------------------------------------------------------


def choice_confidence(probabilities: dict[str, float]) -> float:
    """``confidence = (n * p_max - 1) / (n - 1)``, equivalently
    ``(p_max - 1/n) / (1 - 1/n)``, per https://docs.typesafe.ai/confidence.

    Worked examples from the spec: n=3, p_max=1.0 -> confidence=1.0;
    n=3, p_max=1/3 (uniform) -> confidence=0.0.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    p_max = max(probabilities.values())
    return max(0.0, min(1.0, (n * p_max - 1) / (n - 1)))


def score_expected_value(probabilities: list[float]) -> float:
    """``score = sum(k * p_k for k in 0..n-1)`` - the expected rubric
    position (0-indexed) under the returned probability distribution."""
    return sum(k * p for k, p in enumerate(probabilities))


def _mean_absolute_deviation(probabilities: list[float], about_index: int) -> float:
    return sum(p * abs(k - about_index) for k, p in enumerate(probabilities))


def score_confidence(probabilities: list[float]) -> float:
    """``confidence = max(0, 1 - MAD(p) / MAD(uniform))``, MAD taken around
    the mode (highest-probability) index. 1.0 = fully concentrated on one
    level, 0.0 = spread perfectly uniformly across levels."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    mode_index = max(range(n), key=lambda i: probabilities[i])
    mad = _mean_absolute_deviation(probabilities, mode_index)
    uniform = [1.0 / n] * n
    mad_uniform = _mean_absolute_deviation(uniform, mode_index)
    if mad_uniform == 0:
        return 1.0
    return max(0.0, 1.0 - mad / mad_uniform)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# JevClient implementations
# ---------------------------------------------------------------------------


class JevClient(Protocol):
    def ask(self, state: object, questions: dict[str, JevQuestion]) -> dict[str, JevAnswer]: ...


def typesafe_available() -> bool:
    """True if a ``TYPESAFE_API_KEY`` is configured in the environment."""
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def _parse_answer(raw: dict) -> JevAnswer:
    answer_type = raw.get("type")
    if answer_type == "noul":
        return NoulAnswer(noul=raw["noul"])
    if answer_type == "choice":
        return ChoiceAnswer(choice=raw["choice"], probabilities=raw["probabilities"], confidence=raw["confidence"])
    if answer_type == "score":
        return ScoreAnswer(score=raw["score"], probabilities=raw["probabilities"], confidence=raw["confidence"])
    raise ValueError(f"Unrecognized Jev answer type: {answer_type!r}")


class TypeSafeJevClient:
    """Real adapter for the hosted TypeSafe Jev "System One" model
    (https://docs.typesafe.ai/introduction).

    Calls the documented REST endpoint directly via ``httpx`` rather than
    requiring the optional ``typesafe-sdk`` package, so this adapter has no
    extra dependency beyond what this repo already needs. The official SDK
    (``pip install typesafe-sdk``) is a drop-in alternative that wraps the
    identical contract with ``Choice``/``Noul``/``Score`` request helpers
    and a ``TypeSafeClient`` context manager, if you'd rather use it.

    Only constructed/used when ``TYPESAFE_API_KEY`` is set - raises
    ``RuntimeError`` immediately otherwise, so callers fail fast instead of
    silently making an unauthenticated request. **No live API key was
    available while building this repo, so this adapter is implemented
    against the documented contract but has not been exercised against the
    real hosted model** - see the README's honest-limitations section.
    """

    API_URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, api_key: str | None = None, model: str = "jev-latest", timeout: float = 15.0) -> None:
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set; cannot construct TypeSafeJevClient. "
                "Use LocalJevStandIn for offline development/CI instead."
            )
        self._model = model
        self._timeout = timeout

    def ask(self, state: object, questions: dict[str, JevQuestion]) -> dict[str, JevAnswer]:
        import httpx

        payload = {
            "state": state,
            "model": self._model,
            "questions": {qid: q.model_dump() for qid, q in questions.items()},
        }
        response = httpx.post(
            self.API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        raw = response.json()
        # Tolerate either {"answers": {...}} or a flat {"<id>": {...}} body.
        answers_raw = raw.get("answers", raw)
        return {question_id: _parse_answer(payload_) for question_id, payload_ in answers_raw.items()}


class LocalJevStandIn:
    """**NOT the real trained Jev model.** A contract-compatible, fully
    offline stand-in for development/CI: implements the identical typed
    primitives, response shapes, and confidence formulas documented at
    https://docs.typesafe.ai/confidence, computed deterministically from
    simple rule-based heuristics over the ``state`` payload instead of a
    trained model. Swapping to the real hosted model later (via
    ``TypeSafeJevClient``, once ``TYPESAFE_API_KEY`` is set) requires zero
    code changes anywhere that depends on the :class:`JevClient` protocol -
    only the client construction in ``decay_engine/service.py`` changes.

    The forget-vs-keep heuristic mirrors this repo's own
    ``ConservativeForgetJudge``/``AggressiveForgetJudge`` intuition: a
    logistic curve centered on ``prune_threshold`` so a ``decay_score`` far
    below the threshold yields a confident ("near 1.0") noul answer, one
    right at the threshold yields a maximally ambiguous ("near 0.5")
    answer, and ``pinned`` items are always answered "should not be
    forgotten" (a low noul) as a defensive floor - though in this repo's
    decay engine, pinned nodes never actually reach this gate at all, since
    they exit earlier via the ``pinned`` check in
    ``decay_engine/pruning.py``.
    """

    def ask(self, state: object, questions: dict[str, JevQuestion]) -> dict[str, JevAnswer]:
        state_dict = state if isinstance(state, dict) else {"state": state}
        return {question_id: self._answer_one(state_dict, question) for question_id, question in questions.items()}

    def _answer_one(self, state: dict, question: JevQuestion) -> JevAnswer:
        if isinstance(question, NoulQuestion):
            return NoulAnswer(noul=self._noul_probability(state))
        if isinstance(question, ChoiceQuestion):
            option_ids = list(question.criteria.keys())
            weights = _deterministic_pseudo_distribution(f"{question.instructions}|{state}", len(option_ids))
            probabilities = dict(zip(option_ids, weights))
            winning_option = max(probabilities, key=probabilities.get)
            return ChoiceAnswer(choice=winning_option, probabilities=probabilities, confidence=choice_confidence(probabilities))
        if isinstance(question, ScoreQuestion):
            weights = _deterministic_pseudo_distribution(f"{question.instructions}|{state}", len(question.criteria))
            probabilities = {str(i): w for i, w in enumerate(weights)}
            return ScoreAnswer(
                score=score_expected_value(weights),
                probabilities=probabilities,
                confidence=score_confidence(weights),
            )
        raise ValueError(f"Unsupported Jev question type: {question!r}")

    @staticmethod
    def _noul_probability(state: dict) -> float:
        if "decay_score" not in state or "prune_threshold" not in state:
            return 0.5  # unrecognized state shape: maximally uncertain, not a guess
        if bool(state.get("pinned", False)):
            return 0.02  # defensive floor; decay_engine never routes pinned nodes here anyway
        decay_score = float(state["decay_score"])
        threshold = float(state["prune_threshold"])
        spread = max(threshold, 1e-6) / 2.0
        x = (threshold - decay_score) / spread
        return _clamp01(1.0 / (1.0 + math.exp(-x)))


def _deterministic_pseudo_distribution(seed_text: str, n: int) -> list[float]:
    """A stable, deterministic (not model-based) probability distribution
    derived by hashing ``seed_text`` once per index, normalized to sum to
    1.0. Used only as a generic Choice/Score fallback for
    :class:`LocalJevStandIn` - the cascade wired into this repo's decay
    engine only actually relies on the Noul path above."""
    if n <= 0:
        return []
    weights = []
    for i in range(n):
        digest = hashlib.sha256(f"{seed_text}::{i}".encode()).digest()
        weights.append(1.0 + (digest[0] / 255.0))  # stays in [1, 2)
    total = sum(weights)
    return [w / total for w in weights]


# ---------------------------------------------------------------------------
# Confidence-Gated Routing cascade
# ---------------------------------------------------------------------------


@dataclass
class CascadeConfig:
    #: Noul values inside this closed interval are treated as ambiguous and
    #: escalated to the jury, per TypeSafe's documented pattern.
    ambiguous_band: tuple[float, float] = (0.35, 0.65)
    #: Escalate high-stakes items to the jury even when Jev is confident,
    #: because "thresholds should scale with the stakes of the decision"
    #: (https://docs.typesafe.ai/confidence).
    always_escalate_high_stakes: bool = True


@dataclass
class CascadeDecision:
    case_id: str
    acted_automatically: bool
    forget: bool
    jev_noul: float
    jev_confidence: float
    escalated_to_jury: bool
    jury_decision: JuryDecision | None
    rationale: str


@dataclass
class ConfidenceGatedCascade:
    """Implements TypeSafe's documented "Confidence-Gated Routing" pattern
    (https://docs.typesafe.ai/patterns/confidence-routing): Jev's single
    fast/cheap Noul probability is the default decision signal, and
    confidence - here, distance from 0.5 - is a second axis gating whether
    to act automatically or escalate:

    * **high confidence** (outside ``ambiguous_band`` and not high-stakes)
      -> act on Jev's verdict automatically.
    * **medium/low confidence** (inside ``ambiguous_band``) **or a
      high-stakes item** (e.g. pinned/compliance-flagged) -> escalate to
      this repo's existing LLM-jury panel (:class:`evaluator.jury.Jury`)
      for a slower, more deliberate second opinion. If the jury itself
      cannot reach consensus (``JuryDecision.escalated``), the case is
      additionally enqueued to ``human_review_queue`` (if provided) -
      TypeSafe/DeepEval's "weighted combination of judges" idea
      (https://deepeval.com/blog/introducing-jev-as-a-judge) is exactly
      what ``evaluator.jury.Jury`` already does one layer up, so no
      separate weighting logic is duplicated here.

    This is the efficiency rationale made concrete: Jev (cheap/fast) is the
    default for the large majority of routine, clear-cut cases; the jury
    (slower, more deliberate, multi-model) is reserved for the minority of
    ambiguous or high-stakes cases.
    """

    jev_client: JevClient
    jury: Jury | None
    config: CascadeConfig = field(default_factory=CascadeConfig)
    human_review_queue: object | None = None

    def decide_forget_vs_keep(
        self, *, case_id: str, state: dict, jury_case: dict, is_high_stakes: bool = False
    ) -> CascadeDecision:
        question = {
            "forget": NoulQuestion(
                instructions=(
                    "Should this memory be forgotten (pruned) given its current decay_score, "
                    "recency, pinned status, and access history?"
                ),
            )
        }
        answers = self.jev_client.ask(state, question)
        noul_answer = answers["forget"]
        if not isinstance(noul_answer, NoulAnswer):  # pragma: no cover - defensive
            raise TypeError(f"Expected a NoulAnswer, got {type(noul_answer)!r}")

        p = noul_answer.noul
        confidence = noul_answer.confidence
        low, high = self.config.ambiguous_band
        is_ambiguous = low <= p <= high
        needs_escalation = is_ambiguous or (is_high_stakes and self.config.always_escalate_high_stakes)

        if not needs_escalation:
            forget = p >= 0.5
            return CascadeDecision(
                case_id=case_id,
                acted_automatically=True,
                forget=forget,
                jev_noul=p,
                jev_confidence=confidence,
                escalated_to_jury=False,
                jury_decision=None,
                rationale=(
                    f"Jev noul={p:.3f} (confidence={confidence:.3f}) is outside the ambiguous band "
                    f"{self.config.ambiguous_band} and the item is not high-stakes; acted automatically."
                ),
            )

        if self.jury is None:
            # Escalation was warranted but no jury is configured to escalate
            # to. Per the "low confidence -> do not act" side of
            # Confidence-Gated Routing, default to the conservative
            # non-destructive outcome rather than guessing.
            return CascadeDecision(
                case_id=case_id,
                acted_automatically=False,
                forget=False,
                jev_noul=p,
                jev_confidence=confidence,
                escalated_to_jury=False,
                jury_decision=None,
                rationale=(
                    f"Jev noul={p:.3f} (confidence={confidence:.3f}) warranted escalation "
                    f"({'ambiguous band' if is_ambiguous else 'high-stakes override'}), but no jury is "
                    f"configured; defaulting to the conservative 'do not act' outcome."
                ),
            )

        decision = self.jury.decide(jury_case, allowed_verdicts=["keep", "forget"], case_id=case_id)
        if decision.escalated and self.human_review_queue is not None:
            self.human_review_queue.enqueue(
                tenant_id=jury_case.get("tenant_id", "unknown"),
                node_id=case_id,
                case_type="jev_cascade_jury_abstain",
                case_payload=jury_case,
                votes=[vote.model_dump() for vote in decision.votes],
            )
        forget = decision.final_verdict == "forget"
        return CascadeDecision(
            case_id=case_id,
            acted_automatically=False,
            forget=forget,
            jev_noul=p,
            jev_confidence=confidence,
            escalated_to_jury=True,
            jury_decision=decision,
            rationale=(
                f"Jev noul={p:.3f} (confidence={confidence:.3f}) triggered escalation "
                f"({'ambiguous band' if is_ambiguous else 'high-stakes override'}); "
                f"jury verdict={decision.final_verdict} (escalated={decision.escalated})."
            ),
        )
