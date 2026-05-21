"""
ML model for edge-level fraud classification.

Builds an explicit feature matrix from the fund flow graph and underlying
transaction frame, then trains XGBoost (or GradientBoosting fallback) with a
stratified 80/20 split. Persists the trained model and a metrics JSON.

Why edge-level: each edge in the graph is a (sender, receiver) pair carrying
aggregated transaction stats. That is the unit a bank's monitoring system
scores — "is this counterparty relationship suspicious?" — and it gives
balanced labels for training without leaking individual-transaction noise.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx


# Feature column order is preserved when scoring new edges, so it must be stable.
FEATURE_COLUMNS = [
    "log_total_amount",
    "log_avg_amount",
    "log_max_amount",
    "log_min_amount",
    "txn_count",
    "amount_cv",                # std / avg (coefficient of variation)
    "time_span_hours",
    "sender_is_individual",
    "sender_is_business",
    "sender_is_shell",
    "receiver_is_individual",
    "receiver_is_business",
    "receiver_is_shell",
    "same_branch",
    "sender_in_degree",
    "sender_out_degree",
    "sender_log_in_strength",
    "sender_log_out_strength",
    "receiver_in_degree",
    "receiver_out_degree",
    "receiver_log_in_strength",
    "receiver_log_out_strength",
    "channel_diversity",
    "rail_diversity",
    "high_value_rail_share",    # RTGS/Wire share of txns on this edge
    "upi_share",
    "near_threshold_score",     # 1.0 when avg approaches ₹2L from below
    "night_ratio",
    "weekend_ratio",
    "in_scc_3plus",             # part of a strongly-connected component ≥ 3
    "neighbor_fraud_density",   # share of edges among neighbors that carry fraud
]

# Per-graph cache so we can call extract_features many times without recomputing
# the global node-level stats.
_CTX_CACHE: Dict[str, Dict] = {}


def _build_context(graph: nx.DiGraph) -> Dict:
    """Pre-compute things we want once per graph, not once per edge."""
    in_strength: Dict[str, float] = {}
    out_strength: Dict[str, float] = {}
    for node in graph.nodes():
        in_strength[node] = sum(graph[u][node]["total_amount"] for u in graph.predecessors(node))
        out_strength[node] = sum(graph[node][v]["total_amount"] for v in graph.successors(node))

    # Strongly-connected components of size >= 3 — used for circular hint
    scc_members = set()
    for comp in nx.strongly_connected_components(graph):
        if len(comp) >= 3:
            scc_members.update(comp)

    return {
        "in_strength": in_strength,
        "out_strength": out_strength,
        "scc_members": scc_members,
    }


def _node_type_flags(graph: nx.DiGraph, node_id: str, prefix: str) -> Dict[str, int]:
    t = graph.nodes[node_id].get("type", "individual")
    return {
        f"{prefix}_is_individual": int(t == "individual"),
        f"{prefix}_is_business": int(t == "business"),
        f"{prefix}_is_shell": int(t == "shell_company"),
    }


def _edge_temporal_features(txns_for_edge: pd.DataFrame) -> Tuple[float, float]:
    """Return (night_ratio, weekend_ratio) for the txns on this edge."""
    if len(txns_for_edge) == 0:
        return 0.0, 0.0
    ts = pd.to_datetime(txns_for_edge["timestamp"])
    hours = ts.dt.hour
    night = ((hours < 6) | (hours >= 22)).mean()
    weekend = (ts.dt.weekday >= 5).mean()
    return float(night), float(weekend)


def extract_features(
    graph: nx.DiGraph,
    transactions: pd.DataFrame,
    edges: List[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """Build the feature matrix.

    Each row corresponds to one edge of the graph. If `edges` is None, all
    edges are used. Result is a DataFrame indexed by (sender, receiver).
    """
    if edges is None:
        edges = list(graph.edges())

    ctx = _build_context(graph)
    in_str = ctx["in_strength"]
    out_str = ctx["out_strength"]
    scc_members = ctx["scc_members"]

    txn_by_edge = {
        key: sub for key, sub in transactions.groupby(["sender_id", "receiver_id"])
    }

    rows = []
    for u, v in edges:
        if not graph.has_edge(u, v):
            continue
        ed = graph[u][v]
        total = float(ed["total_amount"])
        avg = float(ed["avg_amount"])
        mn = float(ed["min_amount"])
        mx = float(ed["max_amount"])
        std = float(ed.get("std_amount", 0) or 0)
        cnt = int(ed["transaction_count"])

        try:
            first = pd.to_datetime(ed["first_seen"])
            last = pd.to_datetime(ed["last_seen"])
            time_span_h = (last - first).total_seconds() / 3600.0
        except Exception:
            time_span_h = 0.0

        sender_branch = graph.nodes[u].get("branch", "")
        receiver_branch = graph.nodes[v].get("branch", "")
        same_branch = int(sender_branch == receiver_branch and sender_branch != "")

        # Channel/rail mix
        rail_mix = ed.get("rail_mix") or {}
        channel_mix = ed.get("channel_mix") or {}
        rail_total = sum(rail_mix.values()) or 1
        channel_total = sum(channel_mix.values()) or 1
        high_value_share = (rail_mix.get("RTGS", 0) + rail_mix.get("Wire Transfer", 0)) / rail_total
        upi_share = rail_mix.get("UPI", 0) / rail_total

        # Near-threshold scoring: 1.0 if avg is just below ₹2L
        near_threshold = max(0.0, 1.0 - abs(avg - 195000) / 195000) if avg < 250000 else 0.0

        # Temporal features from raw txns
        sub = txn_by_edge.get((u, v))
        if sub is not None:
            night_ratio, weekend_ratio = _edge_temporal_features(sub)
        else:
            night_ratio, weekend_ratio = 0.0, 0.0

        # Neighbor fraud density (cheap leakage-free heuristic — counts other edges
        # incident on either endpoint that have fraud > 0, excluding this edge)
        neighbor_edges = (
            [(u, w) for w in graph.successors(u) if w != v]
            + [(w, u) for w in graph.predecessors(u) if w != v]
            + [(v, w) for w in graph.successors(v) if w != u]
            + [(w, v) for w in graph.predecessors(v) if w != u]
        )
        if neighbor_edges:
            frauds = sum(1 for a, b in neighbor_edges if graph[a][b].get("fraud_count", 0) > 0)
            neighbor_fraud_density = frauds / len(neighbor_edges)
        else:
            neighbor_fraud_density = 0.0

        row = {
            "log_total_amount": np.log1p(total),
            "log_avg_amount": np.log1p(avg),
            "log_max_amount": np.log1p(mx),
            "log_min_amount": np.log1p(max(mn, 0)),
            "txn_count": cnt,
            "amount_cv": std / max(avg, 1),
            "time_span_hours": time_span_h,
            "same_branch": same_branch,
            "sender_in_degree": graph.in_degree(u),
            "sender_out_degree": graph.out_degree(u),
            "sender_log_in_strength": np.log1p(in_str.get(u, 0)),
            "sender_log_out_strength": np.log1p(out_str.get(u, 0)),
            "receiver_in_degree": graph.in_degree(v),
            "receiver_out_degree": graph.out_degree(v),
            "receiver_log_in_strength": np.log1p(in_str.get(v, 0)),
            "receiver_log_out_strength": np.log1p(out_str.get(v, 0)),
            "channel_diversity": len(channel_mix),
            "rail_diversity": len(rail_mix),
            "high_value_rail_share": high_value_share,
            "upi_share": upi_share,
            "near_threshold_score": near_threshold,
            "night_ratio": night_ratio,
            "weekend_ratio": weekend_ratio,
            "in_scc_3plus": int(u in scc_members and v in scc_members),
            "neighbor_fraud_density": neighbor_fraud_density,
        }
        row.update(_node_type_flags(graph, u, "sender"))
        row.update(_node_type_flags(graph, v, "receiver"))
        row["__edge__"] = f"{u}->{v}"
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.set_index("__edge__")
    return df[FEATURE_COLUMNS]


def _train_xgb(X_train, y_train, X_test, y_test):
    """Train XGBoost; fall back to GradientBoosting if xgboost isn't installed."""
    try:
        from xgboost import XGBClassifier
        # scale_pos_weight handles class imbalance
        pos = max(int(y_train.sum()), 1)
        neg = max(len(y_train) - pos, 1)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model, "xgboost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
        )
        model.fit(X_train, y_train)
        return model, "gradient_boosting_fallback"


def train_and_save(graph: nx.DiGraph, transactions: pd.DataFrame, data_dir: str) -> Dict:
    """Train fraud classifier and persist model + metrics to data_dir/ml/.

    Returns the metrics dict (also written to data_dir/ml/metrics.json).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, average_precision_score,
    )

    out_dir = os.path.join(data_dir, "ml")
    os.makedirs(out_dir, exist_ok=True)

    X = extract_features(graph, transactions)
    if X.empty:
        raise ValueError("No edges available for training.")

    # Labels: edge is fraudulent if any underlying transaction is fraud
    labels = []
    for edge_str in X.index:
        u, v = edge_str.split("->", 1)
        labels.append(int(graph[u][v].get("fraud_count", 0) > 0))
    y = np.array(labels)

    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Need both fraud and non-fraud edges to train.")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.20, stratify=y, random_state=42,
    )

    model, model_kind = _train_xgb(X_train, y_train, X_test, y_test)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "model_kind": model_kind,
        "n_edges": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_fraud_train": int(y_train.sum()),
        "n_fraud_test": int(y_test.sum()),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "fraud_rate": float(y.mean()),
        "feature_columns": FEATURE_COLUMNS,
    }

    # Feature importances (XGBoost / GB both expose them)
    if hasattr(model, "feature_importances_"):
        imps = list(map(float, model.feature_importances_))
        ranked = sorted(zip(FEATURE_COLUMNS, imps), key=lambda x: x[1], reverse=True)
        metrics["feature_importances"] = [
            {"feature": name, "importance": round(imp, 4)} for name, imp in ranked
        ]

    # Score every edge so we have a per-edge ML score available for UI
    all_scores = model.predict_proba(X.values)[:, 1]
    edge_scores = {edge: float(round(s, 4)) for edge, s in zip(X.index, all_scores)}

    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(out_dir, "edge_scores.json"), "w") as f:
        json.dump(edge_scores, f, indent=2)

    return metrics


def load_model(data_dir: str):
    """Load a previously trained model + feature columns."""
    path = os.path.join(data_dir, "ml", "model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics(data_dir: str) -> Dict:
    path = os.path.join(data_dir, "ml", "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_edge_scores(data_dir: str) -> Dict[str, float]:
    path = os.path.join(data_dir, "ml", "edge_scores.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
