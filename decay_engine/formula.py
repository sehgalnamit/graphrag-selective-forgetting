"""The Ebbinghaus-forgetting-curve-inspired decay formula.

The classic Ebbinghaus forgetting curve models retention of a memory trace
as exponential decay over time since it was last reinforced:

    R(t) = e^(-t / S)

where ``R`` is retention (0..1), ``t`` is elapsed time, and ``S`` is the
"stability" of the memory trace - how resistant it is to decay. Ebbinghaus
observed that each additional rehearsal/review of a memory increases its
stability, so the same elapsed time produces less forgetting after more
reinforcement (this is also the basis of spaced-repetition algorithms).

We adapt this directly:

    decay_score = base_importance * e^(-time_since_last_access_hours / stability)
    stability   = half_life_hours * (1 + min(access_count, cap) * reinforcement_factor)

* ``base_importance`` (0..1) scales the ceiling of the curve - an
  intrinsically important fact starts higher and decays from a higher
  baseline than a trivial one.
* ``stability`` grows linearly (and is capped) with ``access_count``: every
  time a memory is retrieved it is reinforced, exactly like rehearsal
  strengthening a trace in the psychological model, which raises its
  effective half-life and slows subsequent decay.
* ``pinned`` nodes (e.g. compliance-critical facts) are completely exempt
  from decay and always evaluate to ``1.0`` - pinning is a hard override of
  the curve, not just a higher base_importance.

The result is clamped to ``[0, 1]``.
"""
from __future__ import annotations

import math

PINNED_DECAY_SCORE = 1.0


def compute_decay_score(
    *,
    base_importance: float,
    time_since_last_access_hours: float,
    access_count: int,
    pinned: bool,
    half_life_hours: float = 168.0,
    reinforcement_factor: float = 0.15,
    reinforcement_cap: int = 20,
) -> float:
    """Compute the current Ebbinghaus-inspired decay_score for a memory node.

    See module docstring for the full formula derivation.
    """
    if pinned:
        return PINNED_DECAY_SCORE

    if time_since_last_access_hours < 0:
        time_since_last_access_hours = 0.0

    capped_access_count = max(0, min(access_count, reinforcement_cap))
    stability = half_life_hours * (1.0 + capped_access_count * reinforcement_factor)
    stability = max(stability, 1e-6)  # guard against div-by-zero on degenerate configs

    decay = base_importance * math.exp(-time_since_last_access_hours / stability)
    return max(0.0, min(1.0, decay))
