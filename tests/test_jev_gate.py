"""Unit tests for the TypeSafe Jev integration (evaluator/jev_gate.py):
the documented confidence formulas, the typed-primitive contract, and the
fully-offline ``LocalJevStandIn``. No network access or API key required -
``TypeSafeJevClient`` itself is only exercised for its fail-fast
construction behavior (see test_typesafe_client_requires_api_key below)."""
from __future__ import annotations

import os

import pytest

from evaluator.jev_gate import (
    ChoiceAnswer,
    ChoiceQuestion,
    LocalJevStandIn,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    TypeSafeJevClient,
    choice_confidence,
    score_confidence,
    score_expected_value,
    typesafe_available,
)

# --- confidence formulas, verified against https://docs.typesafe.ai/confidence ---


def test_choice_confidence_fully_concentrated_is_one():
    # n=3, p_max=1.0 -> confidence = (3*1 - 1) / (3-1) = 1.0
    assert choice_confidence({"a": 1.0, "b": 0.0, "c": 0.0}) == pytest.approx(1.0)


def test_choice_confidence_uniform_is_zero():
    # n=3, p_max=1/3 (uniform) -> confidence = (3*(1/3) - 1) / 2 = 0.0
    assert choice_confidence({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == pytest.approx(0.0)


def test_choice_confidence_matches_documented_formula_directly():
    probabilities = {"a": 0.6, "b": 0.3, "c": 0.1}
    n = len(probabilities)
    p_max = max(probabilities.values())
    expected = (n * p_max - 1) / (n - 1)
    assert choice_confidence(probabilities) == pytest.approx(expected)


def test_choice_confidence_two_options_half_and_half_is_zero():
    assert choice_confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)


def test_score_confidence_fully_concentrated_is_one():
    assert score_confidence([0.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_score_confidence_uniform_is_zero():
    assert score_confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)


def test_score_expected_value_is_weighted_position_sum():
    # score = sum(k * p_k) for k in 0..n-1
    assert score_expected_value([0.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert score_expected_value([0.5, 0.0, 0.5]) == pytest.approx(1.0)
    assert score_expected_value([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_noul_confidence_is_distance_from_half_scaled_to_unit_interval():
    assert NoulAnswer(noul=0.5).confidence == pytest.approx(0.0)
    assert NoulAnswer(noul=1.0).confidence == pytest.approx(1.0)
    assert NoulAnswer(noul=0.0).confidence == pytest.approx(1.0)
    assert NoulAnswer(noul=0.75).confidence == pytest.approx(0.5)


# --- typed primitive contract shapes ---


def test_noul_answer_bounds_are_validated():
    with pytest.raises(Exception):
        NoulAnswer(noul=1.5)


def test_choice_answer_round_trips_fields():
    answer = ChoiceAnswer(choice="a", probabilities={"a": 0.9, "b": 0.1}, confidence=0.8)
    assert answer.type == "choice"
    assert answer.choice == "a"


def test_score_answer_round_trips_fields():
    answer = ScoreAnswer(score=2.1, probabilities={"0": 0.1, "1": 0.2, "2": 0.7}, confidence=0.6)
    assert answer.type == "score"
    assert answer.score == pytest.approx(2.1)


# --- LocalJevStandIn: deterministic, offline, contract-compatible ---


def test_local_jev_standin_noul_is_deterministic():
    client = LocalJevStandIn()
    state = {"decay_score": 0.05, "prune_threshold": 0.2, "access_count": 0, "pinned": False}
    question = {"forget": NoulQuestion(instructions="should this be forgotten?")}
    first = client.ask(state, question)["forget"]
    second = client.ask(state, question)["forget"]
    assert isinstance(first, NoulAnswer)
    assert first.noul == second.noul


def test_local_jev_standin_confident_forget_when_decay_far_below_threshold():
    client = LocalJevStandIn()
    state = {"decay_score": 0.01, "prune_threshold": 0.2, "access_count": 0, "pinned": False}
    answer = client.ask(state, {"forget": NoulQuestion(instructions="x")})["forget"]
    assert answer.noul > 0.65  # confidently "yes, forget"


def test_local_jev_standin_ambiguous_when_decay_at_threshold():
    client = LocalJevStandIn()
    state = {"decay_score": 0.2 - 1e-6, "prune_threshold": 0.2, "access_count": 0, "pinned": False}
    answer = client.ask(state, {"forget": NoulQuestion(instructions="x")})["forget"]
    assert 0.35 <= answer.noul <= 0.65


def test_local_jev_standin_pinned_defensive_floor_is_low():
    client = LocalJevStandIn()
    state = {"decay_score": 0.01, "prune_threshold": 0.2, "access_count": 0, "pinned": True}
    answer = client.ask(state, {"forget": NoulQuestion(instructions="x")})["forget"]
    assert answer.noul < 0.1


def test_local_jev_standin_unrecognized_state_is_maximally_uncertain():
    client = LocalJevStandIn()
    answer = client.ask({"something": "else"}, {"forget": NoulQuestion(instructions="x")})["forget"]
    assert answer.noul == pytest.approx(0.5)


def test_local_jev_standin_choice_question_shape_and_determinism():
    client = LocalJevStandIn()
    question = {"pick": ChoiceQuestion(instructions="pick one", criteria={"a": "desc a", "b": "desc b", "c": "desc c"})}
    state = {"decay_score": 0.3, "prune_threshold": 0.2}
    first = client.ask(state, question)["pick"]
    second = client.ask(state, question)["pick"]
    assert isinstance(first, ChoiceAnswer)
    assert first.choice in {"a", "b", "c"}
    assert set(first.probabilities.keys()) == {"a", "b", "c"}
    assert abs(sum(first.probabilities.values()) - 1.0) < 1e-9
    assert 0.0 <= first.confidence <= 1.0
    assert first.choice == second.choice
    assert first.probabilities == second.probabilities


def test_local_jev_standin_score_question_shape_and_determinism():
    client = LocalJevStandIn()
    question = {"rate": ScoreQuestion(instructions="rate it", criteria=["low", "medium", "high"])}
    state = {"decay_score": 0.3, "prune_threshold": 0.2}
    first = client.ask(state, question)["rate"]
    second = client.ask(state, question)["rate"]
    assert isinstance(first, ScoreAnswer)
    assert 0.0 <= first.score <= 2.0
    assert 0.0 <= first.confidence <= 1.0
    assert first.score == pytest.approx(second.score)


# --- TypeSafeJevClient: real adapter, only its fail-fast behavior is testable offline ---


def test_typesafe_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        TypeSafeJevClient()


def test_typesafe_client_accepts_explicit_api_key_without_env_var(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client = TypeSafeJevClient(api_key="test-key-not-used-over-network")
    assert client._api_key == "test-key-not-used-over-network"


def test_typesafe_available_reflects_env_var(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert typesafe_available() is False
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    assert typesafe_available() is True


def test_typesafe_api_key_not_available_in_this_build_environment():
    """Documents, as an executable assertion, the honest limitation stated
    in the README: this build/test environment has no TYPESAFE_API_KEY, so
    only LocalJevStandIn is ever exercised against real test cases."""
    assert os.environ.get("TYPESAFE_API_KEY") is None
