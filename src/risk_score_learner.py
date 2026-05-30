"""
Learn the per-node risk score weights from data instead of hand-tuning them.

Background: the pre-T2.10 composite score was
    0.4*(type=shell) + 0.3*degree + 0.2*betweenness + 0.15*imbalance + 0.3*fraud_edges
with weights chosen by intuition. Two problems:

  1. Weights aren't derived from data — they could be wrong for any
     particular dataset (PSB-1's smurfing patterns aren't the same as PSB-2's).
  2. `fraud_edges` is both a *feature* AND essentially a label proxy — the
     score conflates "you have flagged edges now" with "you might do something
     suspicious next". Production wants the latter without leaning on the former.

T2.10 replaces the hand-tuned weights with a Logistic-Regression trained on:

  Features (no leakage):
    * is_shell, is_business           — entity type one-hot
    * degree_centrality
    * betweenness_centrality
    * log_in_strength, log_out_strength
    * flow_imbalance
    * branch_diversity_in              — unique branches feeding this node
    * is_singleton_out                 — node has at least one edge to a
                                          counterparty it never receives from
                                          (mule signature)
    * txn_count_total

  Label: was this node part of any fraud_count > 0 edge?
    — i.e. is the entity *currently* tied to flagged activity. The
    feature set deliberately excludes `fraud_edge_count` so the model has
    to predict the label from behavioural signals, not from itself.

Persisted bundle (data/ml/{variant}/risk_weights.pkl):
    {
      "model": <sklearn LogisticRegression>,
      "feature_columns": [...],
      "metrics": {accuracy, auc, n_train, n_test, positive_rate},
      "coefficients": {feature: weight},
    }

At inference time, FraudDetector.compute_node_risk_scores loads this bundle
if it exists and uses it instead of the hand-tuned weights. Falls back
cleanly when the bundle is missing.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx


FEATURE_COLUMNS = [
    "is_shell",
    "is_business",
    "degree_centrality",
    "betweenness_centrality",
    "log_in_strength",
    "log_out_strength",
    "flow_imbalance",
    "branch_diversity_in",
    "is_singleton_out",
    "log_txn_count",
]


def _extract_node_features(graph: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build a (n_nodes, n_features) matrix + per-node binary label.

    Returns:
        X: feature matrix
        y: 0/1 array — 1 if any adjacent edge has fraud_count > 0
        node_ids: stable order of nodes
    """
    nodes = list(graph.nodes())

    # Cache centrality on the graph object so train + score don't recompute.
    # Without this, score_nodes() after train_risk_weights() re-runs the
    # 60s sampled betweenness for no benefit.
    cached = graph.graph.get("_risk_centrality_cache")
    if cached is not None:
        betweenness, degree_c = cached
    else:
        try:
            n_nodes = len(nodes)
            if n_nodes > 2_000:
                # Unweighted BFS sampling — structural centrality, 10x faster
                # than the weighted Dijkstra variant at this scale.
                betweenness = nx.betweenness_centrality(graph, k=500, seed=42)
            else:
                betweenness = nx.betweenness_centrality(graph, weight="total_amount")
        except Exception:
            betweenness = {n: 0.0 for n in nodes}
        try:
            degree_c = nx.degree_centrality(graph)
        except Exception:
            degree_c = {n: 0.0 for n in nodes}
        graph.graph["_risk_centrality_cache"] = (betweenness, degree_c)

    X_rows = []
    y_rows = []
    for n in nodes:
        nd = graph.nodes[n]
        t = nd.get("type", "individual")

        in_strength = sum(graph[u][n]["total_amount"] for u in graph.predecessors(n))
        out_strength = sum(graph[n][v]["total_amount"] for v in graph.successors(n))
        total = in_strength + out_strength
        imbalance = abs(in_strength - out_strength) / total if total > 0 else 0.0

        # Distinct branches feeding this node (matches the fund-funnel signature)
        in_branches = set()
        for p in graph.predecessors(n):
            br = graph.nodes[p].get("branch", "")
            if br:
                in_branches.add(br)

        # Singleton-out: at least one receiver that never sends back
        receivers = set(graph.successors(n))
        senders = set(graph.predecessors(n))
        singleton_out = int(len(receivers - senders) > 0)

        # Total txn count across all adjacent edges
        txn_count = 0
        fraud_edges = 0
        for u in graph.predecessors(n):
            ed = graph[u][n]
            txn_count += int(ed.get("transaction_count", 0))
            if ed.get("fraud_count", 0) > 0:
                fraud_edges += 1
        for v in graph.successors(n):
            ed = graph[n][v]
            txn_count += int(ed.get("transaction_count", 0))
            if ed.get("fraud_count", 0) > 0:
                fraud_edges += 1

        X_rows.append([
            1.0 if t == "shell_company" else 0.0,
            1.0 if t == "business" else 0.0,
            float(degree_c.get(n, 0.0)),
            float(betweenness.get(n, 0.0)),
            float(np.log1p(in_strength)),
            float(np.log1p(out_strength)),
            float(imbalance),
            float(len(in_branches)),
            float(singleton_out),
            float(np.log1p(txn_count)),
        ])
        y_rows.append(1 if fraud_edges > 0 else 0)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64), nodes


def train_risk_weights(graph: nx.DiGraph, data_dir: str, variant: str = "ibm_aml") -> Dict:
    """Train + persist the per-node risk-score model.

    Returns a metrics dict. Skips silently with `{trained: False}` if the
    graph is too small or has no fraud-touched nodes (can't train a
    classifier with a single class).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y, _ = _extract_node_features(graph)
    if len(X) < 20 or y.sum() == 0 or y.sum() == len(y):
        return {
            "trained": False,
            "reason": f"insufficient data: n={len(X)}, positives={int(y.sum())}",
        }

    # Standardise — LR coefficients are scale-sensitive.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, stratify=y, random_state=42,
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "trained": True,
        "variant": variant,
        "n_nodes": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "coefficients": {
            name: float(coef) for name, coef in zip(FEATURE_COLUMNS, model.coef_[0])
        },
        "intercept": float(model.intercept_[0]),
    }

    out_dir = os.path.join(data_dir, "ml", variant)
    os.makedirs(out_dir, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
    }
    with open(os.path.join(out_dir, "risk_weights.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    with open(os.path.join(out_dir, "risk_weights_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_risk_weights(data_dir: str, variant: str = "ibm_aml") -> Dict:
    path = os.path.join(data_dir, "ml", variant, "risk_weights.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def score_nodes(graph: nx.DiGraph, bundle: Dict) -> Dict[str, float]:
    """Score every node in the graph using the trained risk-weight bundle."""
    if not bundle:
        return {}
    X, _, nodes = _extract_node_features(graph)
    X_scaled = bundle["scaler"].transform(X)
    probs = bundle["model"].predict_proba(X_scaled)[:, 1]
    return {n: float(round(p, 4)) for n, p in zip(nodes, probs)}
