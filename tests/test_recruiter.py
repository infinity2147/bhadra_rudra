"""Recruiter / coordinator detector — names the ORCHESTRATOR funding a fleet of
pass-through mules, not the mules themselves. Distinct from smurfing (which is
about structuring amounts); this is one source -> many accounts that forward."""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fraud_detector import FraudDetector  # noqa: E402


def _add(g, u, v, amt):
    g.add_edge(u, v, total_amount=float(amt), transaction_count=1, avg_amount=float(amt),
               min_amount=float(amt), max_amount=float(amt), std_amount=0.0, fraud_count=0)


def _names(g):
    for n in g.nodes():
        g.nodes[n].setdefault("name", n)
        g.nodes[n].setdefault("type", "individual")


def test_recruiter_coordinator_is_flagged():
    """U funds 6 accounts that each forward ~95% onward — U is the coordinator."""
    g = nx.DiGraph()
    for i in range(6):
        _add(g, "U", f"M{i}", 1_000_000)
        _add(g, f"M{i}", f"S{i}", 950_000)
    _names(g)

    alerts = FraudDetector(g).detect_recruiters()
    coords = {a["coordinator"] for a in alerts}
    assert "U" in coords, f"coordinator not flagged; got {coords}"
    a = next(a for a in alerts if a["coordinator"] == "U")
    assert a["pattern_type"] == "Recruiter / Coordinator"
    assert a["n_recruited"] >= 5
    assert "U" in a["entities"]


def test_normal_disburser_not_flagged():
    """U pays 6 recipients who CONSUME the funds (don't forward) — not a recruiter."""
    g = nx.DiGraph()
    for i in range(6):
        _add(g, "U", f"R{i}", 1_000_000)   # recipients have inflow only, no outflow
    _names(g)

    alerts = FraudDetector(g).detect_recruiters()
    assert all(a["coordinator"] != "U" for a in alerts), \
        "a disburser whose recipients don't forward must not be a recruiter"
