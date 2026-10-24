"""Judge implementations.

A ``Judge`` is anything implementing ``evaluate(case, allowed_verdicts) ->
Verdict``. Judges are deliberately simple, deterministic, rule-based stubs
by default so unit tests and CI never need network access or model
downloads. ``OllamaJudge`` is provided as an opt-in example of how a real
local LLM would plug into the exact same interface.

Two case "shapes" are supported (any dict with the right keys works):

* **Forget-vs-keep case** (used to validate borderline decay-engine pruning
  decisions): ``decay_score``, ``prune_threshold``, ``base_importance``,
  ``access_count``, ``pinned``, ``text``.
* **Relevance case** (used to validate retrieval quality): ``query``,
  ``chunk_text``.
"""
from __future__ import annotations

import re
from typing import Any, Protocol

from evaluator.schemas import Verdict

_COMPLIANCE_KEYWORDS = (
    "contract",
    "compliance",
    "regulation",
    "gdpr",
    "policy",
    "audit",
    "legal",
    "sox",
    "hipaa",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def looks_compliance_sensitive(text: str) -> bool:
    """Lightweight, explainable heuristic flagging compliance/legal-relevant
    text. Used by :class:`ComplianceOverrideJudge` below, and reused by
    ``decay_engine.pruning`` as the "high-stakes" signal for
    :class:`evaluator.jev_gate.ConfidenceGatedCascade` (compliance-flagged
    items are escalated to the jury even when Jev itself is confident)."""
    return any(keyword in text.lower() for keyword in _COMPLIANCE_KEYWORDS)


class Judge(Protocol):
    name: str

    def evaluate(self, case: dict[str, Any], allowed_verdicts: list[str]) -> Verdict: ...


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class ConservativeForgetJudge:
    """Biased toward preserving memory: only votes ``forget`` when the
    decay score is clearly below the prune threshold and access_count is
    low (little reinforcement)."""

    name = "conservative_forget_judge"

    def evaluate(self, case: dict[str, Any], allowed_verdicts: list[str]) -> Verdict:
        decay_score = float(case["decay_score"])
        threshold = float(case["prune_threshold"])
        access_count = int(case.get("access_count", 0))
        margin = threshold - decay_score
        if margin > 0.02 and access_count <= 1:
            confidence = _clamp(0.55 + margin * 2)
            return Verdict(
                judge_name=self.name,
                verdict="forget",
                confidence=confidence,
                rationale=(
                    f"decay_score={decay_score:.3f} is below threshold={threshold:.3f} "
                    f"with low reinforcement (access_count={access_count})."
                ),
            )
        confidence = _clamp(0.55 + max(0.0, -margin) * 1.5 + min(access_count, 5) * 0.05)
        return Verdict(
            judge_name=self.name,
            verdict="keep",
            confidence=confidence,
            rationale=(
                f"Being conservative: decay_score={decay_score:.3f}, threshold={threshold:.3f}, "
                f"access_count={access_count} does not clearly justify forgetting yet."
            ),
        )


class AggressiveForgetJudge:
    """Biased toward reclaiming storage/relevance: votes ``forget`` as soon
    as the decay score is at or below the threshold."""

    name = "aggressive_forget_judge"

    def evaluate(self, case: dict[str, Any], allowed_verdicts: list[str]) -> Verdict:
        decay_score = float(case["decay_score"])
        threshold = float(case["prune_threshold"])
        margin = threshold - decay_score
        if decay_score <= threshold * 1.15:
            confidence = _clamp(0.6 + max(0.0, margin) * 2)
            return Verdict(
                judge_name=self.name,
                verdict="forget",
                confidence=confidence,
                rationale=f"decay_score={decay_score:.3f} is at/near threshold={threshold:.3f}; reclaim it.",
            )
        return Verdict(
            judge_name=self.name,
            verdict="keep",
            confidence=_clamp(0.5 + (decay_score - threshold) * 1.5),
            rationale=f"decay_score={decay_score:.3f} still comfortably above threshold={threshold:.3f}.",
        )


class ComplianceOverrideJudge:
    """Votes ``keep`` with high confidence whenever the memory text looks
    compliance/legal-relevant, regardless of decay score (a lightweight,
    explainable stand-in for a real compliance-classification model)."""

    name = "compliance_override_judge"

    def evaluate(self, case: dict[str, Any], allowed_verdicts: list[str]) -> Verdict:
        text = str(case.get("text", ""))
        if looks_compliance_sensitive(text):
            return Verdict(
                judge_name=self.name,
                verdict="keep",
                confidence=0.95,
                rationale="Text contains compliance/legal-sensitive terminology; preserve regardless of decay.",
            )
        decay_score = float(case["decay_score"])
        threshold = float(case["prune_threshold"])
        verdict = "keep" if decay_score >= threshold else "forget"
        confidence = _clamp(0.5 + abs(decay_score - threshold))
        return Verdict(
            judge_name=self.name,
            verdict=verdict,
            confidence=confidence,
            rationale=f"No compliance terms found; deferring to decay_score={decay_score:.3f} vs threshold={threshold:.3f}.",
        )


class LexicalOverlapRelevanceJudge:
    """Rule-based relevance judge: scores retrieval relevance by token
    (word) overlap between the query and the retrieved chunk text."""

    name = "lexical_overlap_relevance_judge"

    def __init__(self, relevance_token_overlap_ratio: float = 0.15) -> None:
        self._min_ratio = relevance_token_overlap_ratio

    def evaluate(self, case: dict[str, Any], allowed_verdicts: list[str]) -> Verdict:
        query_tokens = set(_TOKEN_RE.findall(str(case.get("query", "")).lower()))
        chunk_tokens = set(_TOKEN_RE.findall(str(case.get("chunk_text", "")).lower()))
        if not query_tokens:
            ratio = 0.0
        else:
            ratio = len(query_tokens & chunk_tokens) / len(query_tokens)
        verdict = "relevant" if ratio >= self._min_ratio else "irrelevant"
        return Verdict(
            judge_name=self.name,
            verdict=verdict,
            confidence=_clamp(0.5 + ratio),
            rationale=f"Query/chunk token overlap ratio={ratio:.2f} (threshold={self._min_ratio:.2f}).",
        )


class OllamaJudge:
    """Optional real-LLM judge backed by a local Ollama server.

    Implements the exact same ``Judge`` protocol as the stub judges above,
    demonstrating how this module can "graduate" from deterministic stubs
    to one or more small local models without changing any jury/aggregation
    code. Not used unless ``JURY_OLLAMA_ENABLED=true`` and the server is
    reachable; network/parse failures raise so the caller can decide
    whether to drop this judge from the panel for that run.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 15.0) -> None:
        self.name = f"ollama_judge::{model}"
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def evaluate(self, case: dict[str, Any], allowed_verdicts: list[str]) -> Verdict:
        import json

        import httpx

        prompt = (
            "You are one judge on a panel evaluating a memory-retention decision. "
            f"Respond ONLY with compact JSON: "
            f'{{"verdict": one of {allowed_verdicts}, "confidence": 0..1, "rationale": "short reason"}}.\n'
            f"Case: {json.dumps(case, default=str)}"
        )
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=self._timeout,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "{}")
        parsed = json.loads(raw_text)
        return Verdict(
            judge_name=self.name,
            verdict=str(parsed["verdict"]),
            confidence=_clamp(float(parsed.get("confidence", 0.5))),
            rationale=str(parsed.get("rationale", "")),
        )
