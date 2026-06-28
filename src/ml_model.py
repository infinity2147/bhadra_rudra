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
#
# `neighbor_fraud_density` was removed in v3 — on synthetic data fraud
# clusters very tightly by construction (a circular ring shares many
# neighbour edges among its members), so the feature acts as a shortcut
# and made test F1 artificially high. A production model has to score
# new counterparties without using "are your neighbours already flagged"
# as input, otherwise the score becomes circular when the neighbours
# themselves were mis-labelled. Honest features only from here on.
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
]
# NOTE: transit-ratio / velocity-burst node features were trialled here (v4) on
# the theory that the ML was "blind" to the heuristics' temporal signals. They
# were reverted: on IBM AML they gave NO ranking gain on XGBoost (AP 0.661→0.662,
# a clean test since trees are scale-invariant) because the signals are too
# sparse (~100–300 of 119k nodes fire). The real recall lever turned out to be
# the operating point — see fbeta_optimal_threshold (F2). Don't re-add without
# evidence of an AUPRC lift on a held-out set.

# Per-graph cache so we can call extract_features many times without recomputing
# the global node-level stats.
_CTX_CACHE: Dict[str, Dict] = {}


def _build_context(graph: nx.DiGraph) -> Dict:
    """Pre-compute things we want once per graph, not once per edge.

    Memoised on the graph object: a single-edge SHAP explanation (or any
    extract_features call) would otherwise recompute per-node strengths + SCC
    over the whole graph — seconds on a 100k+ node graph, paid on every request.
    A graph rebuild creates a fresh object, so the cache invalidates naturally.
    """
    cached = graph.graph.get("_ml_ctx_cache")
    if cached is not None:
        return cached
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

    ctx = {
        "in_strength": in_strength,
        "out_strength": out_strength,
        "scc_members": scc_members,
    }
    graph.graph["_ml_ctx_cache"] = ctx
    return ctx


def _node_type_flags(graph: nx.DiGraph, node_id: str, prefix: str) -> Dict[str, int]:
    t = graph.nodes[node_id].get("type", "individual")
    return {
        f"{prefix}_is_individual": int(t == "individual"),
        f"{prefix}_is_business": int(t == "business"),
        f"{prefix}_is_shell": int(t == "shell_company"),
    }


def _edge_temporal_table(transactions: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """Pre-aggregate per-edge temporal features in ONE vectorized pass.

    Returns {(sender, receiver): (night_ratio, weekend_ratio, time_span_hours)}.

    Why this exists: the previous code called ``pd.to_datetime`` once PER EDGE
    (~88k times on IBM AML) on tiny sub-frames. With no explicit format each call
    fails inference and falls back to per-element ``dateutil`` parsing — ~56s for
    extract_features and a flood of warnings, and since extract_features also runs
    on the live-scoring / SHAP path it inflated request latency. A consistent
    timestamp column parses vectorized in ~10ms; night/weekend ratios and the
    time span then collapse to a single groupby aggregation.
    """
    if "timestamp" not in transactions.columns or len(transactions) == 0:
        return {}
    ts = pd.to_datetime(transactions["timestamp"], errors="coerce")
    tmp = pd.DataFrame({
        "sender_id": transactions["sender_id"].to_numpy(),
        "receiver_id": transactions["receiver_id"].to_numpy(),
        "is_night": ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).to_numpy(dtype=float),
        "is_weekend": (ts.dt.weekday >= 5).to_numpy(dtype=float),
        "ts": ts.to_numpy(),
    })
    agg = tmp.groupby(["sender_id", "receiver_id"]).agg(
        night=("is_night", "mean"),
        weekend=("is_weekend", "mean"),
        first=("ts", "min"),
        last=("ts", "max"),
    )
    span_h = (agg["last"] - agg["first"]).dt.total_seconds() / 3600.0
    return {
        key: (float(n), float(w), float(s) if pd.notna(s) else 0.0)
        for key, n, w, s in zip(agg.index, agg["night"], agg["weekend"], span_h)
    }


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

    # Detect currency so near_threshold_score uses the correct reporting limit:
    #   USD (IBM AML) → $9,500  just below the US $10,000 CTR reporting threshold
    #   INR (synthetic) → ₹1,95,000  just below the RBI ₹2L reporting threshold
    currency = "INR"
    if "currency" in transactions.columns and len(transactions) > 0:
        top_currency = transactions["currency"].mode()
        if len(top_currency) > 0 and str(top_currency.iloc[0]).upper() == "USD":
            currency = "USD"
    struct_threshold = 9_500   if currency == "USD" else 195_000
    struct_cap       = 11_000  if currency == "USD" else 250_000

    ctx = _build_context(graph)
    in_str = ctx["in_strength"]
    out_str = ctx["out_strength"]
    scc_members = ctx["scc_members"]

    # Per-edge temporal features (night/weekend ratio + time span), pre-computed
    # once and looked up in the loop — see _edge_temporal_table for why.
    edge_temporal = _edge_temporal_table(transactions)

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

        # night/weekend ratio + time span — pre-computed once (see edge_temporal)
        night_ratio, weekend_ratio, time_span_h = edge_temporal.get((u, v), (0.0, 0.0, 0.0))

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

        # Near-threshold scoring: 1.0 if avg amount is just below the reporting limit
        # USD → $9,500 (US CTR threshold $10,000) | INR → ₹1,95,000 (RBI ₹2L threshold)
        near_threshold = max(0.0, 1.0 - abs(avg - struct_threshold) / struct_threshold) if avg < struct_cap else 0.0

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


def fbeta_optimal_threshold(y_true, y_prob, beta: float = 2.0) -> Tuple[float, float]:
    """Pick the decision threshold that maximises F-beta on the PR curve.

    F1 (beta=1) weights precision and recall equally. For AML that is the wrong
    trade-off: a missed launderer costs far more than an analyst reviewing a
    false positive. beta=2 weights recall 2x precision, so the operating point
    moves to catch more fraud at the cost of more (cheap) reviews — which is why
    the prior F1-optimal threshold (~0.85) was leaving ~half the fraud uncaught.

    Shared by all three trainers (XGBoost, ensemble, GNN) so they agree on what
    counts as fraud. Returns (best_threshold, fbeta_at_best).
    """
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, 0.0
    p = precisions[:-1]
    r = recalls[:-1]
    b2 = beta * beta
    denom = b2 * p + r
    # Safe divide: only evaluate where denom > 0 (else F-beta is 0 by definition),
    # avoiding a divide-by-zero RuntimeWarning when both precision and recall are 0.
    fbeta = np.divide(
        (1 + b2) * p * r, denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    i = int(np.argmax(fbeta))
    return float(thresholds[i]), float(fbeta[i])


def train_and_save(
    graph: nx.DiGraph,
    transactions: pd.DataFrame,
    data_dir: str,
    variant: str = "ibm_aml",
    dataset_name: str = "IBM AML",
) -> Dict:
    """Train fraud classifier and persist model + metrics under data/ml/{variant}/.

    A SHAP-ready background sample (100 rows of training features) is stashed
    inside the pickle so the explainer can be loaded later without re-running
    the pipeline.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, average_precision_score,
    )

    out_dir = os.path.join(data_dir, "ml", variant)
    os.makedirs(out_dir, exist_ok=True)

    X = extract_features(graph, transactions)
    if X.empty:
        raise ValueError("No edges available for training.")

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

    y_prob = model.predict_proba(X_test)[:, 1]

    # Operating point: maximise F2 (recall-favouring), not F1. See
    # fbeta_optimal_threshold — a missed launderer costs far more than an
    # analyst review, so we trade some precision for recall on purpose.
    best_threshold, fbeta2 = fbeta_optimal_threshold(y_test, y_prob, beta=2.0)

    y_pred = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "variant": variant,
        "dataset_name": dataset_name,
        "model_kind": model_kind,
        "n_edges": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_fraud_train": int(y_train.sum()),
        "n_fraud_test": int(y_test.sum()),
        "threshold": round(best_threshold, 4),
        "threshold_objective": "f2",
        "fbeta2": round(fbeta2, 4),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "fraud_rate": float(y.mean()),
        "feature_columns": FEATURE_COLUMNS,
    }

    if hasattr(model, "feature_importances_"):
        imps = list(map(float, model.feature_importances_))
        ranked = sorted(zip(FEATURE_COLUMNS, imps), key=lambda x: x[1], reverse=True)
        metrics["feature_importances"] = [
            {"feature": name, "importance": round(imp, 4)} for name, imp in ranked
        ]

    all_scores = model.predict_proba(X.values)[:, 1]
    edge_scores = {edge: float(round(s, 4)) for edge, s in zip(X.index, all_scores)}

    # Background sample for SHAP — keep it small so the pickle stays light
    bg_idx = np.random.RandomState(0).choice(len(X_train), size=min(100, len(X_train)), replace=False)
    background = X_train[bg_idx]

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "background": background,
        "threshold": best_threshold,   # F2-optimal (recall-favouring), not default 0.5
    }
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(out_dir, "edge_scores.json"), "w") as f:
        json.dump(edge_scores, f, indent=2)

    return metrics


def load_model(data_dir: str, variant: str = "ibm_aml"):
    """Load a previously trained model bundle."""
    path = os.path.join(data_dir, "ml", variant, "model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics(data_dir: str, variant: str = "ibm_aml") -> Dict:
    path = os.path.join(data_dir, "ml", variant, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_edge_scores(data_dir: str, variant: str = "ibm_aml") -> Dict[str, float]:
    path = os.path.join(data_dir, "ml", variant, "edge_scores.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def list_variants(data_dir: str) -> List[Dict]:
    """Return all trained variants on disk, each with their dataset metadata."""
    base = os.path.join(data_dir, "ml")
    if not os.path.isdir(base):
        return []
    out = []
    for entry in sorted(os.listdir(base)):
        full = os.path.join(base, entry)
        metrics_path = os.path.join(full, "metrics.json")
        if os.path.isdir(full) and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            out.append({
                "variant": entry,
                "dataset_name": m.get("dataset_name", entry),
                "f1": m.get("f1"),
                "auc": m.get("auc"),
                "n_edges": m.get("n_edges"),
                "fraud_rate": m.get("fraud_rate"),
            })
    return out


def predict_one(model_bundle: Dict, feature_row: Dict) -> float:
    """Score a single edge given its feature dict — used by live mode."""
    cols = model_bundle["feature_columns"]
    vec = np.array([[feature_row.get(c, 0.0) for c in cols]])
    return float(model_bundle["model"].predict_proba(vec)[0, 1])


def get_threshold(model_bundle: Dict) -> float:
    """Return the recall-optimal threshold stored in this bundle.

    Falls back to 0.40 (conservative F1-balanced estimate) if bundle
    pre-dates threshold saving. Always re-train to get the real value.
    """
    return float(model_bundle.get("threshold", 0.40))
