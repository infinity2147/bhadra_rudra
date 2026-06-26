"""Foundational properties of the shell-funnel detector.

Two flaws the team flagged:
  1. The detector gated on ``type in {shell_company, business}``, so compromised
     *individual* mules — the most common pass-through accounts in the real
     world — were invisible. Type must be a *signal*, not a *gate*.
  2. Holding time was computed against the single most-recent inflow, so a tiny
     decoy deposit placed just before a large exit faked a near-zero holding
     time. Matching must be amount-aware (FIFO), so the large tranche's true
     (long) holding time dominates.
"""

import os
import sys
from datetime import datetime, timedelta

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fraud_detector import FraudDetector  # noqa: E402

_FMT = "%Y-%m-%d %H:%M:%S"


def _add_edge(g, u, v, amount, t_first, t_last=None):
    t_last = t_last or t_first
    g.add_edge(
        u, v,
        total_amount=float(amount), transaction_count=1,
        avg_amount=float(amount), min_amount=float(amount), max_amount=float(amount),
        std_amount=0.0,
        first_seen=t_first.strftime(_FMT), last_seen=t_last.strftime(_FMT),
        fraud_count=0,
    )


def _set_types(g, **types):
    for node, t in types.items():
        g.nodes[node]["type"] = t
        g.nodes[node].setdefault("name", node)


def test_individual_passthrough_mule_is_detected():
    """An individual taking in from 3 sources and instantly forwarding it all
    is a textbook mule — it must be flagged even though it isn't a shell/business."""
    g = nx.DiGraph()
    t0 = datetime(2025, 3, 1, 10, 0, 0)
    for src in ("A", "B", "C"):
        _add_edge(g, src, "M", 2_000_000, t0)
    _add_edge(g, "M", "Z", 6_000_000, t0 + timedelta(minutes=10))
    _set_types(g, A="individual", B="individual", C="individual", M="individual", Z="individual")

    alerts = FraudDetector(g).detect_shell_funnels()
    flagged = {a["funnel_entity"] for a in alerts}

    assert "M" in flagged, f"individual pass-through mule not detected; flagged={flagged}"


def test_holding_time_is_amount_weighted_fifo():
    """A ₹10 decoy deposited 5 min before a ₹10M exit must not fake a 5-minute
    holding time — the ₹10M tranche sat for months and that must dominate."""
    g = nx.DiGraph()
    _add_edge(g, "X", "M", 10_000_000, datetime(2025, 1, 1, 0, 0, 0))      # arrived months ago
    _add_edge(g, "Y", "M", 10, datetime(2025, 4, 1, 9, 55, 0))             # decoy, 5 min pre-exit
    _add_edge(g, "M", "Z", 10_000_000, datetime(2025, 4, 1, 10, 0, 0))     # the real exit
    _set_types(g, X="business", Y="individual", M="business", Z="business")

    holding_seconds = FraudDetector(g)._avg_holding_time("M")

    # The ₹10M tranche was held ~90 days; the decoy must not drag this to minutes.
    assert holding_seconds is not None
    assert holding_seconds > 30 * 86400, \
        f"holding time {holding_seconds/86400:.1f}d should reflect the ₹10M tranche, not the decoy"
