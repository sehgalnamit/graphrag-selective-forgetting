"""Structured I/O contracts for the LLM-Jury evaluator."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    """One judge's independent opinion on a case."""

    judge_name: str
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class JuryDecision(BaseModel):
    """The panel's aggregated decision for a single case."""

    case_id: str
    final_verdict: str
    aggregated_confidence: float = Field(ge=0.0, le=1.0)
    method: str
    escalated: bool
    votes: list[Verdict]
