"""System-level evaluation: does the *combined alert set* catch fraud entities?

The supervised model's own confusion matrix lives in metrics.json. This harness
measures the thing the new lanes actually move — whether the alerts, taken
together, flag the entities that are truly involved in fraud — by scoring alert
coverage against the ground-truth fraud labels on the graph edges.
"""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evaluate_detection import evaluate_alert_entities  # noqa: E402


def _labelled_graph():
    g = nx.DiGraph()
    # fraud edges -> fraud entities {A, B, C, D}
    g.add_edge("A", "B", fraud_count=2)
    g.add_edge("C", "D", fraud_count=1)
    # clean edge -> E is the only fraud-free entity
    g.add_edge("D", "E", fraud_count=0)
    return g


def test_entity_confusion_matrix_counts():
    g = _labelled_graph()
    alerts = [
        {"alert_id": "1", "entities": ["A", "B"]},  # true positives
        {"alert_id": "2", "entities": ["E"]},        # false positive
        # C, D are fraud but unalerted -> false negatives
    ]
    r = evaluate_alert_entities(g, alerts)

    assert r["n_fraud_entities"] == 4      # A, B, C, D
    assert r["tp"] == 2                     # A, B
    assert r["fn"] == 2                     # C, D
    assert r["fp"] == 1                     # E
    assert r["tn"] == 0
    assert r["recall"] == 0.5              # 2 / 4
    assert round(r["precision"], 4) == 0.6667  # 2 / 3
    # sklearn-style confusion matrix [[TN, FP], [FN, TP]]
    assert r["confusion_matrix"] == [[0, 1], [2, 2]]


def test_perfect_coverage_is_full_recall():
    g = _labelled_graph()
    alerts = [{"alert_id": "1", "entities": ["A", "B", "C", "D"]}]
    r = evaluate_alert_entities(g, alerts)
    assert r["recall"] == 1.0
    assert r["fn"] == 0


def test_no_alerts_is_zero_recall():
    g = _labelled_graph()
    r = evaluate_alert_entities(g, [])
    assert r["tp"] == 0 and r["recall"] == 0.0
    assert r["fn"] == r["n_fraud_entities"]
