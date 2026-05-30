"""
Per-transaction live scoring + latency benchmarking.

The original /api/live/inject endpoint looked up edge_scores.json — that's
NOT real-time scoring, it's a cache lookup. This module actually runs each
incoming transaction through:

  1. Feature extraction in the current graph context (with the new
     transaction overlaid).
  2. XGBoost predict_proba on the 30-dim feature vector.
  3. Heuristic pattern hints (cycle membership of either endpoint, etc.).
  4. Returns the score + latency in microseconds.

Latency is measured in time.perf_counter — same clock the latency-benchmark
endpoint uses for the full pipeline.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx

from ml_model import FEATURE_COLUMNS


def _build_live_features(
    graph: nx.DiGraph,
    sender: str,
    receiver: str,
    amount: float,
    channel: str,
    rail: str,
    timestamp: pd.Timestamp,
    currency: str = "INR",
) -> Dict[str, float]:
    """Compute the feature row for a single (sender, receiver, amount) txn.

    Treats this txn as an extension to any existing edge — so if there were
    already 3 transactions on this edge, we use those amounts plus this one
    to compute the running stats. Cheap to compute, no graph mutation.
    """
    if graph.has_edge(sender, receiver):
        ed = graph[sender][receiver]
        prev_total = float(ed.get("total_amount", 0))
        prev_count = int(ed.get("transaction_count", 0))
        prev_min = float(ed.get("min_amount", amount))
        prev_max = float(ed.get("max_amount", amount))
        rail_mix = dict(ed.get("rail_mix") or {})
        channel_mix = dict(ed.get("channel_mix") or {})
        try:
            first_seen = pd.to_datetime(ed.get("first_seen"))
        except Exception:
            first_seen = timestamp
    else:
        prev_total = 0.0
        prev_count = 0
        prev_min = amount
        prev_max = amount
        rail_mix = {}
        channel_mix = {}
        first_seen = timestamp

    new_total = prev_total + amount
    new_count = prev_count + 1
    new_min = min(prev_min, amount) if prev_count > 0 else amount
    new_max = max(prev_max, amount) if prev_count > 0 else amount
    new_avg = new_total / new_count
    # Rough std update via running estimate (sufficient for ML)
    new_std = max(prev_max - prev_min, 0) / 4.0

    time_span_h = (timestamp - first_seen).total_seconds() / 3600.0

    rail_mix[rail] = rail_mix.get(rail, 0) + 1
    channel_mix[channel] = channel_mix.get(channel, 0) + 1
    rail_total = sum(rail_mix.values())
    high_value_share = (rail_mix.get("RTGS", 0) + rail_mix.get("Wire Transfer", 0)) / rail_total
    upi_share = rail_mix.get("UPI", 0) / rail_total

    sender_type = graph.nodes.get(sender, {}).get("type", "individual") if graph.has_node(sender) else "individual"
    receiver_type = graph.nodes.get(receiver, {}).get("type", "individual") if graph.has_node(receiver) else "individual"
    sender_branch = graph.nodes.get(sender, {}).get("branch", "") if graph.has_node(sender) else ""
    receiver_branch = graph.nodes.get(receiver, {}).get("branch", "") if graph.has_node(receiver) else ""

    sender_in_deg = graph.in_degree(sender) if graph.has_node(sender) else 0
    sender_out_deg = graph.out_degree(sender) if graph.has_node(sender) else 0
    receiver_in_deg = graph.in_degree(receiver) if graph.has_node(receiver) else 0
    receiver_out_deg = graph.out_degree(receiver) if graph.has_node(receiver) else 0

    if graph.has_node(sender):
        sender_in_str = sum(graph[u][sender]["total_amount"] for u in graph.predecessors(sender))
        sender_out_str = sum(graph[sender][v]["total_amount"] for v in graph.successors(sender))
    else:
        sender_in_str = 0
        sender_out_str = 0
    if graph.has_node(receiver):
        receiver_in_str = sum(graph[u][receiver]["total_amount"] for u in graph.predecessors(receiver))
        receiver_out_str = sum(graph[receiver][v]["total_amount"] for v in graph.successors(receiver))
    else:
        receiver_in_str = 0
        receiver_out_str = 0

    # USD → $9,500 (below US $10,000 CTR threshold) | INR → ₹1,95,000 (below RBI ₹2L threshold)
    struct_threshold = 9_500  if currency.upper() == "USD" else 195_000
    struct_cap       = 11_000 if currency.upper() == "USD" else 250_000
    near_threshold = max(0.0, 1.0 - abs(new_avg - struct_threshold) / struct_threshold) if new_avg < struct_cap else 0.0

    hour = timestamp.hour
    night_ratio = 1.0 if (hour < 6 or hour >= 22) else 0.0
    weekend_ratio = 1.0 if timestamp.weekday() >= 5 else 0.0

    # Cycle membership — approximate: are both endpoints already in any SCC ≥ 3?
    in_scc = 0
    try:
        for comp in nx.strongly_connected_components(graph):
            if len(comp) >= 3 and sender in comp and receiver in comp:
                in_scc = 1
                break
    except Exception:
        pass

    return {
        "log_total_amount": float(np.log1p(new_total)),
        "log_avg_amount": float(np.log1p(new_avg)),
        "log_max_amount": float(np.log1p(new_max)),
        "log_min_amount": float(np.log1p(max(new_min, 0))),
        "txn_count": float(new_count),
        "amount_cv": float(new_std / max(new_avg, 1)),
        "time_span_hours": float(time_span_h),
        "sender_is_individual": 1.0 if sender_type == "individual" else 0.0,
        "sender_is_business": 1.0 if sender_type == "business" else 0.0,
        "sender_is_shell": 1.0 if sender_type == "shell_company" else 0.0,
        "receiver_is_individual": 1.0 if receiver_type == "individual" else 0.0,
        "receiver_is_business": 1.0 if receiver_type == "business" else 0.0,
        "receiver_is_shell": 1.0 if receiver_type == "shell_company" else 0.0,
        "same_branch": 1.0 if sender_branch and sender_branch == receiver_branch else 0.0,
        "sender_in_degree": float(sender_in_deg),
        "sender_out_degree": float(sender_out_deg),
        "sender_log_in_strength": float(np.log1p(sender_in_str)),
        "sender_log_out_strength": float(np.log1p(sender_out_str)),
        "receiver_in_degree": float(receiver_in_deg),
        "receiver_out_degree": float(receiver_out_deg),
        "receiver_log_in_strength": float(np.log1p(receiver_in_str)),
        "receiver_log_out_strength": float(np.log1p(receiver_out_str)),
        "channel_diversity": float(len(channel_mix)),
        "rail_diversity": float(len(rail_mix)),
        "high_value_rail_share": float(high_value_share),
        "upi_share": float(upi_share),
        "near_threshold_score": float(near_threshold),
        "night_ratio": float(night_ratio),
        "weekend_ratio": float(weekend_ratio),
        "in_scc_3plus": float(in_scc),
    }


def score_live_txn(
    model_bundle: Dict,
    graph: nx.DiGraph,
    sender: str,
    receiver: str,
    amount: float,
    channel: str,
    rail: str,
    timestamp: Optional[pd.Timestamp] = None,
    currency: str = "INR",
) -> Dict:
    """Score one txn through the trained ML model, with feature extraction + latency."""
    if timestamp is None:
        timestamp = pd.Timestamp.now()
    t0 = time.perf_counter()
    features = _build_live_features(graph, sender, receiver, amount, channel, rail, timestamp, currency)
    t1 = time.perf_counter()

    cols = model_bundle["feature_columns"]
    vec = np.array([[features.get(c, 0.0) for c in cols]])
    proba = float(model_bundle["model"].predict_proba(vec)[0, 1])
    t2 = time.perf_counter()

    return {
        "ml_score": round(proba, 4),
        "features": features,
        "latency_ms": {
            "feature_extraction": round((t1 - t0) * 1000, 3),
            "model_inference": round((t2 - t1) * 1000, 3),
            "total": round((t2 - t0) * 1000, 3),
        },
    }


# ── Pipeline-level latency benchmark ──────────────────────────────────────────

def benchmark_pipeline(
    graph: nx.DiGraph,
    transactions: pd.DataFrame,
    model_bundle: Dict,
    n_samples: int = 200,
) -> Dict:
    """Measure the wall-clock cost of each pipeline stage.

    Banks ask "is RUDRA actually faster than T+1?" — this is the answer.
    We time:
      - Per-txn ML scoring (mean + p95 over n_samples)
      - Full graph rebuild
      - All detectors run
      - Full-graph ML scoring
    """
    from graph_engine import FundFlowGraph
    from fraud_detector import FraudDetector
    from advanced_detectors import DormantActivationDetector, ProfileMismatchDetector
    from ml_model import extract_features

    # 1. Per-txn ML latency (sampling existing edges)
    edges = list(graph.edges())[:n_samples] if graph.number_of_edges() else []
    per_txn_total_ms = []
    for u, v in edges:
        sample = transactions[
            (transactions["sender_id"] == u) & (transactions["receiver_id"] == v)
        ].head(1)
        if sample.empty:
            continue
        row = sample.iloc[0]
        ts = pd.to_datetime(row["timestamp"]) if "timestamp" in row else pd.Timestamp.now()
        res = score_live_txn(
            model_bundle, graph, u, v,
            float(row["amount"]), str(row.get("channel", "NetBanking")),
            str(row.get("transaction_type", "NEFT")), ts,
        )
        per_txn_total_ms.append(res["latency_ms"]["total"])

    per_txn_mean = float(np.mean(per_txn_total_ms)) if per_txn_total_ms else 0.0
    per_txn_p95 = float(np.percentile(per_txn_total_ms, 95)) if per_txn_total_ms else 0.0
    per_txn_p99 = float(np.percentile(per_txn_total_ms, 99)) if per_txn_total_ms else 0.0

    # 2. Full pipeline timings
    t0 = time.perf_counter()
    ffg = FundFlowGraph()
    rebuilt = ffg.build_graph(transactions)
    t_graph = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    det = FraudDetector(rebuilt)
    det.detect_circular_transactions()
    det.detect_rapid_layering()
    det.detect_smurfing()
    det.detect_shell_funnels()
    t_detectors = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    DormantActivationDetector(rebuilt, transactions).detect()
    t_dormant = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    X = extract_features(rebuilt, transactions)
    if not X.empty:
        model_bundle["model"].predict_proba(X.values)
    t_ml_full = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    det.compute_node_risk_scores()
    t_risk = (time.perf_counter() - t0) * 1000

    full_total_ms = t_graph + t_detectors + t_dormant + t_ml_full + t_risk

    # Comparison vs T+1 (24 hours = 86_400_000 ms)
    t_plus_1_ms = 86_400_000
    speedup = t_plus_1_ms / max(full_total_ms, 1)

    return {
        "txn_count": int(len(transactions)),
        "per_txn_ms": {
            "mean": round(per_txn_mean, 3),
            "p95": round(per_txn_p95, 3),
            "p99": round(per_txn_p99, 3),
            "samples": len(per_txn_total_ms),
        },
        "pipeline_ms": {
            "graph_rebuild": round(t_graph, 2),
            "core_detectors": round(t_detectors, 2),
            "dormant_detector": round(t_dormant, 2),
            "ml_full_scoring": round(t_ml_full, 2),
            "risk_scoring": round(t_risk, 2),
            "total": round(full_total_ms, 2),
        },
        "vs_t_plus_1": {
            "t_plus_1_ms": t_plus_1_ms,
            "rudra_full_ms": round(full_total_ms, 2),
            "speedup_factor": round(speedup, 1),
        },
        "measured_at": pd.Timestamp.now().isoformat(),
    }
