"""Unit tests for fuzzy membership helpers used to soften rule-detector gates.

The key safety property is the margin=0 step behaviour: with margin 0 the ramps
reduce to the exact hard thresholds the detectors use today, so a detector wired
through them with margin 0 produces identical output (zero regression).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fuzzy import ramp_upper, ramp_lower, combine  # noqa: E402


# ── ramp_upper: membership for "value must be BELOW a cap" ───────────────────

def test_ramp_upper_below_limit_is_full_membership():
    assert ramp_upper(40_000, 50_000, margin=0.1) == 1.0


def test_ramp_upper_at_limit_is_full_membership():
    assert ramp_upper(50_000, 50_000, margin=0.1) == 1.0


def test_ramp_upper_midpoint_is_half():
    # 50_000 .. 55_000 ramp; 52_500 is the midpoint -> 0.5
    assert ramp_upper(52_500, 50_000, margin=0.1) == pytest.approx(0.5)


def test_ramp_upper_at_far_edge_is_zero():
    assert ramp_upper(55_000, 50_000, margin=0.1) == pytest.approx(0.0)


def test_ramp_upper_beyond_margin_is_zero():
    assert ramp_upper(60_000, 50_000, margin=0.1) == 0.0


def test_ramp_upper_margin_zero_is_step_function():
    assert ramp_upper(50_000, 50_000, margin=0.0) == 1.0
    assert ramp_upper(50_001, 50_000, margin=0.0) == 0.0


# ── ramp_lower: membership for "value must be AT LEAST n" ────────────────────

def test_ramp_lower_above_min_is_full_membership():
    assert ramp_lower(6, 5, margin=0.2) == 1.0


def test_ramp_lower_at_min_is_full_membership():
    assert ramp_lower(5, 5, margin=0.2) == 1.0


def test_ramp_lower_midpoint_is_half():
    # 4 .. 5 ramp (5*0.8 = 4); 4.5 is the midpoint -> 0.5
    assert ramp_lower(4.5, 5, margin=0.2) == pytest.approx(0.5)


def test_ramp_lower_at_far_edge_is_zero():
    assert ramp_lower(4, 5, margin=0.2) == pytest.approx(0.0)


def test_ramp_lower_below_margin_is_zero():
    assert ramp_lower(3, 5, margin=0.2) == 0.0


def test_ramp_lower_margin_zero_is_step_function():
    assert ramp_lower(5, 5, margin=0.0) == 1.0
    assert ramp_lower(4, 5, margin=0.0) == 0.0


# ── combine: weakest-link aggregation of several memberships ─────────────────

def test_combine_takes_the_minimum():
    assert combine(0.8, 0.5, 0.9) == 0.5


def test_combine_no_args_is_full_membership():
    assert combine() == 1.0


def test_combine_single_value_passthrough():
    assert combine(0.7) == 0.7
