"""
SHAP local explanations for edge-level ML scores.

For each fraud alert, an investigator wants to know *why* the model flagged a
particular counterparty relationship. Global feature importance (the chart on
the ML page) only tells us which features matter on average. SHAP gives
*per-prediction* attributions: for this specific edge, which features pushed
the probability up, and which pulled it down?

We use SHAP's TreeExplainer (works natively with XGBoost / GradientBoosting),
which is exact and fast. For a single edge prediction it returns one
contribution per feature; summing them up plus the base value equals the
log-odds the model outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from ml_model import extract_features


def _import_shap():
    try:
        import shap
        return shap
    except ImportError as e:
        raise ImportError(
            "SHAP not installed. Run `pip install shap`. "
            f"Original error: {e}"
        )


def explain_edge(
    model_bundle: Dict,
    graph: nx.DiGraph,
    transactions: pd.DataFrame,
    sender: str,
    receiver: str,
    top_k: int = 8,
) -> Optional[Dict]:
    """Return per-feature SHAP contributions for a single edge.

    Result schema (so the frontend can render a waterfall):
        {
          "edge": "u->v",
          "predicted_proba": 0.82,
          "base_value": ...,            # model's average log-odds
          "features": [
            {"feature": "log_total_amount", "value": 14.3, "shap": 0.91},
            ...
          ],
          "narrative": [short one-line bullets]
        }
    """
    if not graph.has_edge(sender, receiver):
        return None

    shap = _import_shap()
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_columns"]
    background = model_bundle.get("background")

    # Compute the feature row for this edge
    X = extract_features(graph, transactions, edges=[(sender, receiver)])
    if X.empty:
        return None
    edge_vec = X.values[0]

    # TreeExplainer is fast and exact for XGBoost/sklearn tree models
    explainer = shap.TreeExplainer(model, data=background) if background is not None else shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(edge_vec.reshape(1, -1))

    # shap_values can come back as ndarray or list — normalise to a 1D vector
    if isinstance(shap_vals, list):
        shap_vec = np.asarray(shap_vals[-1]).flatten()
    else:
        shap_vec = np.asarray(shap_vals).flatten()
    if len(shap_vec) > len(feature_cols):
        # newer SHAP returns shape (1, F) — flatten kept all
        shap_vec = shap_vec[:len(feature_cols)]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).flatten()[-1])

    proba = float(model.predict_proba(edge_vec.reshape(1, -1))[0, 1])

    # Sort by absolute contribution
    ranked = sorted(
        zip(feature_cols, edge_vec, shap_vec),
        key=lambda x: abs(x[2]),
        reverse=True,
    )
    top = ranked[:top_k]

    # Narrative: turn the top contributions into plain-English bullets
    narrative = []
    for name, val, contrib in top[:5]:
        direction = "raised" if contrib > 0 else "lowered"
        narrative.append(_narrate_feature(name, val, contrib, direction))

    return {
        "edge": f"{sender}->{receiver}",
        "predicted_proba": round(proba, 4),
        "base_value": float(base_value),
        "features": [
            {
                "feature": name,
                "value": float(val),
                "shap": float(contrib),
            }
            for name, val, contrib in ranked
        ],
        "top_features": [
            {
                "feature": name,
                "value": float(val),
                "shap": float(contrib),
            }
            for name, val, contrib in top
        ],
        "narrative": narrative,
    }


def explain_alert(
    model_bundle: Dict,
    graph: nx.DiGraph,
    transactions: pd.DataFrame,
    alert: Dict,
) -> Optional[Dict]:
    """Explain an alert by picking its highest-amount edge between alert entities."""
    entities = alert.get("entities", [])
    if len(entities) < 2:
        # Single-entity alert (dormant / profile) — fall back to incident-edge
        if not entities:
            return None
        focal = entities[0]
        # Pick the largest outgoing edge from this entity
        best = None
        best_amount = 0
        for v in graph.successors(focal):
            amt = graph[focal][v]["total_amount"]
            if amt > best_amount:
                best_amount = amt
                best = (focal, v)
        if best is None:
            for u in graph.predecessors(focal):
                amt = graph[u][focal]["total_amount"]
                if amt > best_amount:
                    best_amount = amt
                    best = (u, focal)
        if best is None:
            return None
        return explain_edge(model_bundle, graph, transactions, best[0], best[1])

    # Pick the highest-amount edge between any pair of alert entities
    best = None
    best_amount = 0
    for u in entities:
        for v in entities:
            if u != v and graph.has_edge(u, v):
                amt = graph[u][v]["total_amount"]
                if amt > best_amount:
                    best_amount = amt
                    best = (u, v)
    if best is None:
        return None
    return explain_edge(model_bundle, graph, transactions, best[0], best[1])


# ── narrative helpers ────────────────────────────────────────────────────────

_FEATURE_LABEL = {
    "log_total_amount": "Total flow on this edge",
    "log_avg_amount": "Average transaction amount",
    "log_max_amount": "Largest transaction",
    "log_min_amount": "Smallest transaction",
    "txn_count": "Number of transactions",
    "amount_cv": "Variability of amounts (std / mean)",
    "time_span_hours": "Time span between first and last txn",
    "sender_is_individual": "Sender is an individual",
    "sender_is_business": "Sender is a business",
    "sender_is_shell": "Sender is a shell company",
    "receiver_is_individual": "Receiver is an individual",
    "receiver_is_business": "Receiver is a business",
    "receiver_is_shell": "Receiver is a shell company",
    "same_branch": "Same branch on both sides",
    "sender_in_degree": "Sender has many incoming counterparties",
    "sender_out_degree": "Sender has many outgoing counterparties",
    "sender_log_in_strength": "Sender's total inflow",
    "sender_log_out_strength": "Sender's total outflow",
    "receiver_in_degree": "Receiver has many incoming counterparties",
    "receiver_out_degree": "Receiver has many outgoing counterparties",
    "receiver_log_in_strength": "Receiver's total inflow",
    "receiver_log_out_strength": "Receiver's total outflow",
    "channel_diversity": "Number of distinct channels used",
    "rail_diversity": "Number of distinct payment rails",
    "high_value_rail_share": "Share of RTGS / wire-transfer txns",
    "upi_share": "Share of UPI transactions",
    "near_threshold_score": "Closeness of avg amount to the ₹2L reporting limit",
    "night_ratio": "Share of transactions at night (10pm–6am)",
    "weekend_ratio": "Share of transactions on weekends",
    "in_scc_3plus": "Edge is part of a transaction cycle (SCC ≥ 3)",
}


def _narrate_feature(name: str, value: float, contrib: float, direction: str) -> str:
    label = _FEATURE_LABEL.get(name, name)
    return f"{label} {direction} the score (SHAP {contrib:+.3f})"
