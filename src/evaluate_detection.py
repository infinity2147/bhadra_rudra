"""System-level detection evaluation — entity coverage vs ground-truth labels.

The per-model confusion matrix (in ``metrics.json``) measures one classifier on a
held-out edge split. It does *not* capture what the rule + ML lanes do
**together**, which is what an analyst actually receives. This module scores the
combined alert set: an entity counts as "detected" if any alert names it, and as
"fraud" if it touches any edge labelled fraudulent. Comparing the before/after
alert sets on the same graph shows the recall the new lanes add.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import networkx as nx


def fraud_entities(graph: nx.DiGraph, fraud_edge_attr: str = "fraud_count") -> set:
    """Entities that touch at least one fraud-labelled edge (either endpoint)."""
    out = set()
    for u, v, data in graph.edges(data=True):
        if float(data.get(fraud_edge_attr, 0) or 0) > 0:
            out.add(u)
            out.add(v)
    return out


def alerted_entities(alerts: Iterable[Dict]) -> set:
    out = set()
    for a in alerts:
        out.update(a.get("entities", []))
    return out


def _fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def evaluate_alert_entities(
    graph: nx.DiGraph,
    alerts: List[Dict],
    fraud_edge_attr: str = "fraud_count",
) -> Dict:
    """Entity-level confusion matrix of an alert set against the graph's labels.

    Positives = entities touching a fraud edge. Predicted positives = entities
    named by any alert. Returns counts, precision/recall/F1/F2, and an
    sklearn-style confusion matrix ``[[TN, FP], [FN, TP]]``.
    """
    all_entities = set(graph.nodes())
    truth = fraud_entities(graph, fraud_edge_attr) & all_entities
    pred = alerted_entities(alerts) & all_entities

    tp = len(truth & pred)
    fp = len(pred - truth)
    fn = len(truth - pred)
    tn = len(all_entities - truth - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "n_entities": len(all_entities),
        "n_fraud_entities": len(truth),
        "n_alerted_entities": len(pred),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_fbeta(precision, recall, 1.0), 4),
        "f2": round(_fbeta(precision, recall, 2.0), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
