"""Fuzzy membership helpers for softening rule-detector fire/no-fire gates.

The rule detectors decide whether to emit an alert with hard cutoffs (e.g.
``amount < smurfing_threshold``, ``count >= min_txns``). A structurer hugging the
boundary (₹49,990 against a ₹50,000 limit, 4 transfers against a 5-transfer
minimum) is dropped entirely. These ramps replace the hard boundary with a soft
one: values within a configurable *margin* of the boundary are admitted at a
membership degree ``μ ∈ (0, 1]`` that the detector folds into its confidence.

The margin is config-driven and defaults to 0. **With margin 0 the ramps reduce
to the original step functions**, so a detector wired through them is byte-for-byte
unchanged until an operator opts in — the zero-regression property.
"""

from __future__ import annotations


def ramp_upper(value: float, hard_limit: float, margin: float = 0.0) -> float:
    """Membership for a "value must be BELOW a cap" gate.

    Returns 1.0 at/below ``hard_limit``, ramps linearly down to 0.0 across
    ``(hard_limit, hard_limit * (1 + margin)]``, and 0.0 beyond. ``margin == 0``
    is the hard step: 1.0 at/below the limit, 0.0 above.
    """
    if value <= hard_limit:
        return 1.0
    if margin <= 0.0:
        return 0.0
    far_edge = hard_limit * (1.0 + margin)
    if value >= far_edge:
        return 0.0
    return (far_edge - value) / (far_edge - hard_limit)


def ramp_lower(value: float, hard_min: float, margin: float = 0.0) -> float:
    """Membership for a "value must be AT LEAST n" gate.

    Returns 1.0 at/above ``hard_min``, ramps linearly down to 0.0 across
    ``[hard_min * (1 - margin), hard_min)``, and 0.0 below. ``margin == 0`` is the
    hard step: 1.0 at/above the minimum, 0.0 below.
    """
    if value >= hard_min:
        return 1.0
    if margin <= 0.0:
        return 0.0
    near_edge = hard_min * (1.0 - margin)
    if value <= near_edge:
        return 0.0
    return (value - near_edge) / (hard_min - near_edge)


def combine(*memberships: float) -> float:
    """Weakest-link aggregation: the membership of the whole is its weakest part.

    No arguments → 1.0 (vacuously full membership), so a detector that has not
    enabled any fuzzy gate is unaffected.
    """
    if not memberships:
        return 1.0
    return min(memberships)
