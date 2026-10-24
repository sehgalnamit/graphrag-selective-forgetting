"""Unit tests for the Ebbinghaus-inspired decay formula."""
from __future__ import annotations

import pytest

from decay_engine.formula import compute_decay_score


def test_pinned_nodes_never_decay():
    score = compute_decay_score(
        base_importance=0.1,
        time_since_last_access_hours=100_000,
        access_count=0,
        pinned=True,
    )
    assert score == 1.0


def test_decay_decreases_monotonically_with_time():
    scores = [
        compute_decay_score(
            base_importance=0.8,
            time_since_last_access_hours=t,
            access_count=0,
            pinned=False,
            half_life_hours=168,
        )
        for t in (0, 24, 168, 500, 5000)
    ]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(0.8)
    assert scores[-1] < 0.01


def test_zero_elapsed_time_equals_base_importance():
    score = compute_decay_score(
        base_importance=0.42, time_since_last_access_hours=0, access_count=0, pinned=False
    )
    assert score == pytest.approx(0.42)


def test_reinforcement_slows_decay():
    """Same elapsed time, higher access_count => slower decay (higher score)."""
    low_reinforcement = compute_decay_score(
        base_importance=0.6, time_since_last_access_hours=300, access_count=0, pinned=False
    )
    high_reinforcement = compute_decay_score(
        base_importance=0.6, time_since_last_access_hours=300, access_count=10, pinned=False
    )
    assert high_reinforcement > low_reinforcement


def test_reinforcement_is_capped():
    just_under_cap = compute_decay_score(
        base_importance=0.6,
        time_since_last_access_hours=300,
        access_count=20,
        pinned=False,
        reinforcement_cap=20,
    )
    way_over_cap = compute_decay_score(
        base_importance=0.6,
        time_since_last_access_hours=300,
        access_count=2000,
        pinned=False,
        reinforcement_cap=20,
    )
    assert just_under_cap == pytest.approx(way_over_cap)


def test_score_bounds_are_always_respected():
    for t in (-10, 0, 1, 10_000_000):
        for importance in (0.0, 0.5, 1.0):
            score = compute_decay_score(
                base_importance=importance, time_since_last_access_hours=t, access_count=0, pinned=False
            )
            assert 0.0 <= score <= 1.0


def test_negative_elapsed_time_is_clamped_to_zero():
    score = compute_decay_score(
        base_importance=0.5, time_since_last_access_hours=-50, access_count=0, pinned=False
    )
    assert score == pytest.approx(0.5)
