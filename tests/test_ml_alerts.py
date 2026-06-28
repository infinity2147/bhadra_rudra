"""Unit tests for ML-driven alert generation + rule/ML tiering."""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ml_alert_generator import (  # noqa: E402
    generate_ml_alerts,
    apply_tiers,
    tier_summary,
    KEEP_RULE_ONLY_TYPES,
)


def _toy_graph():
    g = nx.DiGraph()
    g.add_node("A", name="Acme")
    g.add_node("B", name="Beta")
    g.add_node("C", name="Gamma")
    g.add_edge("A", "B", total_amount=500000.0, transaction_count=3)
    g.add_edge("B", "C", total_amount=200000.0, transaction_count=1)
    return g


def test_generate_ml_alerts_respects_threshold_and_schema():
    g = _toy_graph()
    scores = {"A->B": {"ensemble": 0.95}, "B->C": {"ensemble": 0.40}}
    alerts = generate_ml_alerts(g, scores, threshold=0.73)
    # only the edge above threshold becomes an alert
    assert len(alerts) == 1
    a = alerts[0]
    assert a["entities"] == ["A", "B"]
    assert a["pattern_type"] == "ML-Detected Anomaly"
    assert a["source"] == "ml"
    assert a["ml_score"] == 0.95
    # schema fields the downstream pipeline relies on
    for key in ("alert_id", "severity", "confidence", "entity_names", "total_flow", "description"):
        assert key in a


def test_generate_ml_alerts_accepts_flat_xgb_scores():
    g = _toy_graph()
    alerts = generate_ml_alerts(g, {"A->B": 0.9, "B->C": 0.1}, threshold=0.5)
    assert len(alerts) == 1 and alerts[0]["entities"] == ["A", "B"]


def test_tiering_corroboration_and_suppression():
    g = _toy_graph()
    ml_alerts = generate_ml_alerts(g, {"A->B": 0.95}, threshold=0.5)  # flags A,B

    rule_alerts = [
        # shares entity A with the ML alert -> both should become Tier 1
        {"alert_id": "R1", "pattern_type": "Circular Transaction", "entities": ["A", "X"]},
        # high-precision typology, NO ML overlap -> kept as Tier 3
        {"alert_id": "R2", "pattern_type": "Rapid Layering", "entities": ["Y", "Z"]},
        # low-precision typology, NO ML overlap -> suppressed
        {"alert_id": "R3", "pattern_type": "Smurfing / Structuring", "entities": ["P", "Q"]},
    ]
    combined = apply_tiers(rule_alerts, ml_alerts)
    by_id = {a.get("alert_id"): a for a in combined}

    # ML alert corroborated by R1 (shared entity A) -> Tier 1
    ml = next(a for a in combined if a.get("source") == "ml")
    assert ml["tier"] == 1
    assert "Circular Transaction" in ml["corroborated_by"]

    # R1 corroborated by ML -> Tier 1
    assert by_id["R1"]["tier"] == 1
    # R2 high-precision typology, rule-only -> Tier 3 kept
    assert by_id["R2"]["tier"] == 3
    # R3 low-precision rule-only -> suppressed (dropped)
    assert "R3" not in by_id

    assert "Rapid Layering" in KEEP_RULE_ONLY_TYPES

    s = tier_summary(combined)
    assert s["tier1"] == 2 and s["tier3"] == 1 and s["total"] == 3
