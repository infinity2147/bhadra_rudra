"""
RUDRA — FastAPI Backend (v3)

The investigator-facing API. Wires together:
  - synthetic + real-data ML model variants
  - 6 heuristic detectors + GraphSAGE GNN
  - SHAP local explanations per alert
  - Case workflow on SQLite with hash-chain audit log
  - Threshold-config layer + RBAC roles
  - Live-mode per-txn ML scoring + latency benchmark
  - Fund journey tracer + incident clustering
  - FIU evidence-package generator (zip with STR XML, SAR PDF, etc.)
  - Account Aggregator + DiliSense KYC mocks
  - LLM copilot
"""

import os
import sys
import json
import random
import time
from io import BytesIO
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import networkx as nx

from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Make sibling src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_generator import TransactionGenerator, save_data
from graph_engine import FundFlowGraph
from fraud_detector import FraudDetector
from advanced_detectors import DormantActivationDetector, ProfileMismatchDetector
from sar_generator import SARGenerator
from llm_copilot import LLMCopilot
from ml_model import (
    train_and_save as ml_train_and_save,
    load_model as ml_load_model,
    load_metrics as ml_load_metrics,
    load_edge_scores as ml_load_edge_scores,
    list_variants as ml_list_variants,
)
from gnn_model import load_gnn_metrics, load_gnn_edge_scores
from shap_explainer import explain_alert as shap_explain_alert
from fund_tracer import trace_journey, trace_for_alert
from case_manager import CaseStore, VALID_STATUSES
from fiu_package import build_package as build_fiu_package
from incident_clustering import cluster_alerts, alert_to_incident_map
from config_store import ConfigStore, DEFAULT_CONFIG
from rbac import get_role, require, role_capabilities, VALID_ROLES
from live_scoring import score_live_txn, benchmark_pipeline
from aa_kyc_mock import (
    aa_create_consent, aa_pull_data, aa_revoke_consent, aa_list_consents,
    dilisense_screen,
)


app = FastAPI(title="RUDRA API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── State ──────────────────────────────────────────────────────────────────────
state = {
    "transactions": None,        # pd.DataFrame, timestamp parsed
    "alerts": None,              # list[dict]
    "incidents": None,           # list[dict] grouped alerts
    "risk_scores": None,
    "summary": None,
    "fraud_cases": None,
    "ffg": None,                 # FundFlowGraph
    "graph": None,               # nx.DiGraph
    "copilot": None,
    "sar_gen": None,
    "cases": None,               # CaseStore
    "config": None,              # ConfigStore
    "ml_bundle": None,           # current synthetic model bundle (used for SHAP + live scoring)
    "ml_metrics": None,
    "edge_scores": None,
    "loaded": False,
}


# ── Pipeline integration ──────────────────────────────────────────────────────

def run_pipeline():
    """Trigger the full pipeline (used by /api/pipeline/run + on cold start)."""
    generator = TransactionGenerator(seed=42)
    df, fraud_cases = generator.generate_all_data()
    save_data(df, fraud_cases, DATA_DIR, entities=generator.entities)

    ffg = FundFlowGraph()
    graph = ffg.build_graph(df)

    detector = FraudDetector(graph)
    results = detector.run_all_detections()
    dormant_alerts = DormantActivationDetector(graph, df).detect()
    risk_scores_data = []
    for node_id, score in results["node_risk_scores"].items():
        nd = dict(graph.nodes[node_id])
        risk_scores_data.append({
            "entity_id": node_id, "name": nd.get("name", ""),
            "type": nd.get("type", ""), "risk_score": score,
            "risk_level": ("CRITICAL" if score >= 0.7 else "HIGH" if score >= 0.5
                            else "MEDIUM" if score >= 0.3 else "LOW"),
        })
    profile_alerts = ProfileMismatchDetector(graph, df, risk_scores_data).detect()
    all_alerts = results["all_alerts"] + dormant_alerts + profile_alerts
    results["all_alerts"] = all_alerts
    results["dormant_activation"] = dormant_alerts
    results["profile_mismatch"] = profile_alerts
    results["summary"]["total_alerts"] = len(all_alerts)
    results["summary"]["dormant_count"] = len(dormant_alerts)
    results["summary"]["profile_count"] = len(profile_alerts)
    results["summary"]["critical_alerts"] = sum(1 for a in all_alerts if a["severity"] == "CRITICAL")
    results["summary"]["high_alerts"] = sum(1 for a in all_alerts if a["severity"] == "HIGH")
    results["summary"]["medium_alerts"] = sum(1 for a in all_alerts if a["severity"] == "MEDIUM")
    detector.save_results(results, DATA_DIR)

    incidents = cluster_alerts(all_alerts, graph=graph)
    with open(os.path.join(DATA_DIR, "incidents.json"), "w") as f:
        json.dump(incidents, f, indent=2, default=str)

    try:
        ml_train_and_save(graph, df, DATA_DIR, variant="synthetic",
                           dataset_name="RUDRA Synthetic Generator")
    except Exception as e:
        print(f"ML training failed: {e}")

    sar_gen = SARGenerator(graph, df, all_alerts, fraud_cases)
    sar_dir = os.path.join(DATA_DIR, "sar_reports")
    os.makedirs(sar_dir, exist_ok=True)
    for sar in sar_gen.generate_all_sars(min_severity="HIGH"):
        sar_gen.export_sar_pdf(sar, sar_dir)

    state["loaded"] = False


def load_or_generate():
    if state["loaded"]:
        return

    alerts_path = os.path.join(DATA_DIR, "fraud_alerts.json")
    txn_path = os.path.join(DATA_DIR, "transactions.csv")
    if not os.path.exists(alerts_path) or not os.path.exists(txn_path):
        print("[backend] Data not found — running full pipeline...")
        run_pipeline()

    df = pd.read_csv(txn_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    state["transactions"] = df

    with open(os.path.join(DATA_DIR, "fraud_alerts.json")) as f:
        state["alerts"] = json.load(f)
    with open(os.path.join(DATA_DIR, "risk_scores.json")) as f:
        state["risk_scores"] = json.load(f)
    with open(os.path.join(DATA_DIR, "detection_summary.json")) as f:
        state["summary"] = json.load(f)
    with open(os.path.join(DATA_DIR, "fraud_cases.json")) as f:
        state["fraud_cases"] = json.load(f)

    incidents_path = os.path.join(DATA_DIR, "incidents.json")
    if os.path.exists(incidents_path):
        with open(incidents_path) as f:
            state["incidents"] = json.load(f)
    else:
        state["incidents"] = []

    state["ffg"] = FundFlowGraph()
    state["graph"] = state["ffg"].build_graph(df)
    state["copilot"] = LLMCopilot(
        state["graph"], df, state["alerts"], state["risk_scores"], state["fraud_cases"],
    )
    state["sar_gen"] = SARGenerator(
        state["graph"], df, state["alerts"], state["fraud_cases"],
    )

    # Case store on SQLite (auto-migrates from old cases.json if present)
    case_store = CaseStore(DATA_DIR)
    case_store.bulk_open_all(state["alerts"])
    # Backfill incident_id on cases
    a2i = alert_to_incident_map(state["incidents"])
    for aid, inc_id in a2i.items():
        case_store.set_incident(aid, inc_id)
    state["cases"] = case_store

    # Config store (same SQLite file)
    state["config"] = ConfigStore(os.path.join(DATA_DIR, "rudra.db"))

    # ML artefacts
    state["ml_metrics"] = ml_load_metrics(DATA_DIR, variant="synthetic")
    state["edge_scores"] = ml_load_edge_scores(DATA_DIR, variant="synthetic")
    state["ml_bundle"] = ml_load_model(DATA_DIR, variant="synthetic")
    if state["ml_bundle"] is None:
        try:
            ml_train_and_save(state["graph"], df, DATA_DIR, variant="synthetic")
            state["ml_metrics"] = ml_load_metrics(DATA_DIR, variant="synthetic")
            state["edge_scores"] = ml_load_edge_scores(DATA_DIR, variant="synthetic")
            state["ml_bundle"] = ml_load_model(DATA_DIR, variant="synthetic")
        except Exception as e:
            print(f"[backend] inline ML training skipped: {e}")

    state["loaded"] = True
    print(f"[backend] ready: {len(df)} txns, {len(state['alerts'])} alerts, "
          f"{len(state['incidents'])} incidents, "
          f"{state['graph'].number_of_nodes()} entities, "
          f"ML F1={state['ml_metrics'].get('f1', 0):.3f}")


@app.on_event("startup")
def startup():
    load_or_generate()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _alerts_with_case_status() -> List[Dict]:
    """Decorate alerts with case status + ML score + incident id."""
    cases = state["cases"]
    edge_scores = state["edge_scores"] or {}
    graph = state["graph"]
    a2i = alert_to_incident_map(state["incidents"] or [])
    out = []
    for a in state["alerts"]:
        entities = a.get("entities", [])
        case = cases.get(a.get("alert_id"))
        ml_score = None
        if len(entities) >= 2 and graph is not None:
            best = 0.0
            for u in entities:
                for v in entities:
                    if u != v and graph.has_edge(u, v):
                        s = edge_scores.get(f"{u}->{v}")
                        if s is not None and s > best:
                            best = s
            if best > 0:
                ml_score = round(best, 3)
        decorated = dict(a)
        decorated["case_status"] = case.get("status") if case else "OPEN"
        decorated["assigned_to"] = case.get("assigned_to") if case else None
        decorated["ml_score"] = ml_score
        decorated["incident_id"] = a2i.get(a.get("alert_id"))
        out.append(decorated)
    return out


def _filter_by_time_window(df: pd.DataFrame, until: Optional[str]) -> pd.DataFrame:
    """Time-travel slicing — return txns up to `until` (inclusive)."""
    if not until:
        return df
    try:
        cutoff = pd.to_datetime(until)
    except Exception:
        return df
    return df[df["timestamp"] <= cutoff]


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "RUDRA API",
        "version": "3.0",
        "ml_trained": bool(state["ml_metrics"]),
        "alerts": len(state["alerts"] or []),
        "incidents": len(state["incidents"] or []),
    }


@app.get("/api/me")
def me(role: str = Depends(get_role)):
    """Whoami — returns the role passed via X-User-Role header + permissions."""
    return {
        "role": role,
        "valid_roles": sorted(VALID_ROLES),
        "permissions": role_capabilities(role),
    }


# ── Dashboard ──────────────────────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard(until: Optional[str] = None):
    df = _filter_by_time_window(state["transactions"], until)
    alerts = state["alerts"]
    summary = state["summary"]

    fraud_txns = df[df["is_fraud"]]
    total_volume = float(df["amount"].sum())
    fraud_volume = float(fraud_txns["amount"].sum())

    daily = df.groupby(df["timestamp"].dt.date).agg(
        count=("amount", "count"),
        volume=("amount", "sum"),
        fraud_count=("is_fraud", "sum"),
        fraud_volume=("amount", lambda x: x[df.loc[x.index, "is_fraud"]].sum()),
    ).reset_index()
    daily.columns = ["date", "count", "volume", "fraud_count", "fraud_volume"]
    daily["date"] = daily["date"].astype(str)

    pattern_breakdown = (
        fraud_txns.groupby("fraud_pattern").agg(
            count=("amount", "count"),
            total=("amount", "sum"),
        ).reset_index().to_dict("records")
    )

    risk_dist = {}
    for r in state["risk_scores"]:
        level = r["risk_level"]
        risk_dist[level] = risk_dist.get(level, 0) + 1

    case_status_counts = state["cases"].status_counts(alerts)
    ml = state["ml_metrics"] or {}

    buckets = [
        ("<₹50K", 0, 50_000),
        ("₹50K-2L", 50_000, 200_000),
        ("₹2L-10L", 200_000, 1_000_000),
        ("₹10L-50L", 1_000_000, 5_000_000),
        (">₹50L", 5_000_000, float("inf")),
    ]
    amount_distribution = []
    for label, lo, hi in buckets:
        mask = (df["amount"] >= lo) & (df["amount"] < hi)
        sub = df.loc[mask]
        amount_distribution.append({
            "bucket": label,
            "normal_count": int((~sub["is_fraud"]).sum()),
            "fraud_count": int(sub["is_fraud"].sum()),
        })

    time_window = {
        "start": df["timestamp"].min().isoformat() if len(df) else None,
        "end": df["timestamp"].max().isoformat() if len(df) else None,
        "applied_until": until,
    }

    return {
        "kpis": {
            "total_transactions": len(df),
            "total_volume": round(total_volume, 2),
            "fraud_transactions": len(fraud_txns),
            "fraud_volume": round(fraud_volume, 2),
            "fraud_rate": round(len(fraud_txns) / max(len(df), 1) * 100, 1),
            "total_alerts": summary["total_alerts"],
            "critical_alerts": summary.get("critical_alerts", 0),
            "high_risk_entities": sum(1 for r in state["risk_scores"] if r["risk_score"] >= 0.5),
            "incidents": len(state["incidents"] or []),
            "model_f1": ml.get("f1"),
            "model_auc": ml.get("auc"),
        },
        "daily_data": daily.to_dict("records"),
        "pattern_breakdown": pattern_breakdown,
        "risk_distribution": risk_dist,
        "case_status_counts": case_status_counts,
        "amount_distribution": amount_distribution,
        "time_window": time_window,
    }


# ── Graph ──────────────────────────────────────────────────────────────────

@app.get("/api/graph")
def get_graph(
    fraud_only: bool = False,
    min_amount: float = 0,
    high_risk_only: bool = False,
):
    graph = state["graph"]
    risk_map = {r["entity_id"]: r["risk_score"] for r in state["risk_scores"]}
    edge_scores = state["edge_scores"] or {}

    if high_risk_only:
        nodes_to_include = {r["entity_id"] for r in state["risk_scores"] if r["risk_score"] >= 0.3}
    else:
        nodes_to_include = set(graph.nodes())

    fraud_edges = {(u, v) for u, v, d in graph.edges(data=True) if d.get("fraud_count", 0) > 0}
    fraud_nodes = set()
    for u, v in fraud_edges:
        fraud_nodes.add(u); fraud_nodes.add(v)

    if fraud_only:
        nodes_to_include = nodes_to_include & fraud_nodes
        expanded = set(nodes_to_include)
        for n in nodes_to_include:
            expanded.update(graph.predecessors(n))
            expanded.update(graph.successors(n))
        nodes_to_include = expanded

    sub = graph.subgraph(nodes_to_include).copy()
    edges_to_remove = [(u, v) for u, v, d in sub.edges(data=True) if d["total_amount"] < min_amount]
    sub.remove_edges_from(edges_to_remove)
    sub.remove_nodes_from(list(nx.isolates(sub)))

    if sub.number_of_nodes() == 0:
        return {"nodes": [], "links": []}

    nodes = []
    for n in sub.nodes():
        nd = dict(sub.nodes[n])
        in_deg = sub.in_degree(n); out_deg = sub.out_degree(n)
        nodes.append({
            "id": n,
            "name": nd.get("name", n),
            "type": nd.get("type", "individual"),
            "branch": nd.get("branch", ""),
            "product": nd.get("product", ""),
            "isFraud": n in fraud_nodes,
            "riskScore": round(float(risk_map.get(n, 0)), 3),
            "degree": in_deg + out_deg,
            "val": max(3, min(20, (in_deg + out_deg) * 0.5)),
        })

    links = []
    for u, v, d in sub.edges(data=True):
        links.append({
            "source": u, "target": v,
            "amount": round(float(d["total_amount"]), 2),
            "txCount": int(d["transaction_count"]),
            "avgAmount": round(float(d["avg_amount"]), 2),
            "isFraud": (u, v) in fraud_edges,
            "fraudCount": int(d.get("fraud_count", 0)),
            "mlScore": edge_scores.get(f"{u}->{v}"),
        })

    return {"nodes": nodes, "links": links}


@app.get("/api/graph/{entity_id}")
def get_subgraph(entity_id: str, hops: int = 2):
    graph = state["graph"]
    if not graph.has_node(entity_id):
        raise HTTPException(404, "Entity not found")

    sub = state["ffg"].extract_subgraph([entity_id], hops=hops)
    risk_map = {r["entity_id"]: r["risk_score"] for r in state["risk_scores"]}
    fraud_edges = {(u, v) for u, v, d in sub.edges(data=True) if d.get("fraud_count", 0) > 0}
    fraud_nodes = set()
    for u, v in fraud_edges:
        fraud_nodes.add(u); fraud_nodes.add(v)

    nodes = []
    for n in sub.nodes():
        nd = dict(sub.nodes[n])
        nodes.append({
            "id": n,
            "name": nd.get("name", n),
            "type": nd.get("type", "individual"),
            "branch": nd.get("branch", ""),
            "isFraud": n in fraud_nodes,
            "riskScore": round(float(risk_map.get(n, 0)), 3),
            "degree": sub.degree(n),
            "val": max(3, min(15, sub.degree(n) * 0.5)),
        })

    edges = []
    for u, v, d in sub.edges(data=True):
        edges.append({
            "source": u, "target": v,
            "amount": round(float(d["total_amount"]), 2),
            "txCount": int(d["transaction_count"]),
            "isFraud": (u, v) in fraud_edges,
        })

    nd = dict(graph.nodes[entity_id])
    in_str = sum(graph[u][entity_id]["total_amount"] for u in graph.predecessors(entity_id))
    out_str = sum(graph[entity_id][v]["total_amount"] for v in graph.successors(entity_id))
    return {
        "entity": {
            "id": entity_id, "name": nd.get("name", entity_id),
            "type": nd.get("type", ""), "branch": nd.get("branch", ""),
            "riskScore": round(float(risk_map.get(entity_id, 0)), 3),
            "inflow": round(float(in_str), 2),
            "outflow": round(float(out_str), 2),
            "netFlow": round(float(in_str - out_str), 2),
        },
        "nodes": nodes,
        "links": edges,
    }


# ── Alerts ─────────────────────────────────────────────────────────────────

@app.get("/api/alerts")
def get_alerts(
    severity: Optional[str] = None,
    pattern: Optional[str] = None,
):
    alerts = _alerts_with_case_status()
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    if pattern:
        alerts = [a for a in alerts if pattern.lower() in a.get("pattern_type", "").lower()]
    return {"alerts": alerts, "total": len(alerts)}


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: str):
    alert = next((a for a in _alerts_with_case_status() if a.get("alert_id") == alert_id), None)
    if not alert:
        raise HTTPException(404, "Alert not found")
    return alert


@app.get("/api/alerts/{alert_id}/explain")
def explain_alert(alert_id: str):
    """SHAP local explanation for the highest-amount edge in the alert."""
    alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
    if not alert:
        raise HTTPException(404, "Alert not found")
    if state["ml_bundle"] is None:
        return {"error": "ML model not loaded yet."}
    try:
        explanation = shap_explain_alert(
            state["ml_bundle"], state["graph"], state["transactions"], alert,
        )
        if explanation is None:
            return {"error": "No edge available to explain for this alert."}
        return explanation
    except ImportError as e:
        return {"error": f"SHAP unavailable: {e}"}
    except Exception as e:
        return {"error": f"Explanation failed: {e}"}


# ── Incidents ──────────────────────────────────────────────────────────────

@app.get("/api/incidents")
def get_incidents(severity: Optional[str] = None):
    inc = state["incidents"] or []
    if severity:
        inc = [i for i in inc if i.get("severity") == severity]
    return {"incidents": inc, "total": len(inc)}


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: str):
    inc = next((i for i in (state["incidents"] or []) if i.get("incident_id") == incident_id), None)
    if not inc:
        raise HTTPException(404, "Incident not found")
    # Include the underlying alerts for this incident
    inc_full = dict(inc)
    inc_full["alerts"] = [a for a in _alerts_with_case_status() if a.get("alert_id") in inc.get("alert_ids", [])]
    return inc_full


# ── Patterns ───────────────────────────────────────────────────────────────

@app.get("/api/patterns/{pattern_type}")
def get_pattern(pattern_type: str):
    df = state["transactions"]
    fraud_txns = df[df["is_fraud"]]
    pattern_map = {
        "circular": "circular_transaction",
        "layering": "rapid_layering",
        "smurfing": "smurfing",
        "funnel": "shell_funnel",
        "dormant": "dormant_activation",
        "profile": "profile_mismatch",
    }
    pattern_fraud = pattern_map.get(pattern_type)
    if pattern_fraud and pattern_fraud in [
        "circular_transaction", "rapid_layering", "smurfing", "shell_funnel", "dormant_activation"
    ]:
        txns = fraud_txns[fraud_txns["fraud_pattern"] == pattern_fraud]
        alerts = [a for a in state["alerts"]
                  if pattern_fraud.replace("_", " ").title() in a.get("pattern_type", "")
                  or pattern_fraud in a.get("pattern_type", "").lower()]
        cols = [c for c in ["timestamp", "sender_name", "receiver_name", "amount", "transaction_type",
                             "channel", "sender_type", "receiver_type", "sender_branch", "fraud_case_id"]
                 if c in txns.columns]
        txn_list = txns[cols].sort_values(["fraud_case_id", "timestamp"]).to_dict("records") if cols else []
        return {
            "pattern": pattern_type,
            "alerts": alerts,
            "transactions": txn_list,
            "total_volume": round(float(txns["amount"].sum()), 2) if len(txns) > 0 else 0,
            "total_transactions": len(txns),
        }
    alert_type = "Dormant Activation" if pattern_type == "dormant" else "Profile Mismatch"
    alerts = [a for a in state["alerts"] if a.get("pattern_type") == alert_type]
    return {"pattern": pattern_type, "alerts": alerts, "transactions": [], "total_volume": 0, "total_transactions": 0}


# ── Entities ───────────────────────────────────────────────────────────────

@app.get("/api/entities")
def get_entities(search: Optional[str] = None, risk_level: Optional[str] = None):
    entities = state["risk_scores"]
    if search:
        entities = [e for e in entities if search.lower() in e["name"].lower()]
    if risk_level:
        entities = [e for e in entities if e["risk_level"] == risk_level]
    return {"entities": entities, "total": len(entities)}


@app.get("/api/entities/{entity_id}")
def get_entity(entity_id: str):
    df = state["transactions"]
    graph = state["graph"]
    if not graph.has_node(entity_id):
        raise HTTPException(404, "Entity not found")

    nd = dict(graph.nodes[entity_id])
    risk_info = next((r for r in state["risk_scores"] if r["entity_id"] == entity_id), {})
    entity_txns = df[(df["sender_id"] == entity_id) | (df["receiver_id"] == entity_id)]
    sent = entity_txns[entity_txns["sender_id"] == entity_id]
    received = entity_txns[entity_txns["receiver_id"] == entity_id]
    keep_cols = [c for c in ["timestamp", "sender_name", "receiver_name", "amount",
                              "transaction_type", "channel", "is_fraud", "fraud_pattern"]
                  if c in entity_txns.columns]
    txn_history = (
        entity_txns[keep_cols].sort_values("timestamp", ascending=False).head(50).to_dict("records")
        if keep_cols else []
    )
    in_str = sum(graph[u][entity_id]["total_amount"] for u in graph.predecessors(entity_id))
    out_str = sum(graph[entity_id][v]["total_amount"] for v in graph.successors(entity_id))
    return {
        "id": entity_id,
        "name": nd.get("name", entity_id),
        "type": nd.get("type", ""),
        "branch": nd.get("branch", ""),
        "product": nd.get("product", ""),
        "riskScore": risk_info.get("risk_score", 0),
        "riskLevel": risk_info.get("risk_level", "N/A"),
        "inflow": round(float(in_str), 2),
        "outflow": round(float(out_str), 2),
        "netFlow": round(float(in_str - out_str), 2),
        "totalTransactions": len(entity_txns),
        "fraudTransactions": len(entity_txns[entity_txns["is_fraud"]]),
        "sentVolume": round(float(sent["amount"].sum()), 2),
        "receivedVolume": round(float(received["amount"].sum()), 2),
        "transactionHistory": txn_history,
    }


# ── Copilot ────────────────────────────────────────────────────────────────

@app.post("/api/copilot/query")
async def copilot_query(body: dict):
    query = body.get("query", "")
    if not query:
        raise HTTPException(400, "Query is required")
    result = state["copilot"].query(query)
    return {
        "response": result["response"],
        "source": result.get("source", "local"),
        "tool_calls": result.get("tool_calls", []),
    }


# ── SAR Reports ────────────────────────────────────────────────────────────

@app.get("/api/sar/generate/{alert_id}")
def generate_sar(alert_id: str):
    alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
    if not alert:
        raise HTTPException(404, "Alert not found")
    sar = state["sar_gen"].generate_sar(alert)
    return {
        "report_id": sar["report_id"],
        "alert_id": sar.get("alert_id", alert_id),
        "report_text": sar["report_text"],
        "severity": sar["severity"],
        "pattern_type": sar["pattern_type"],
        "entities": sar["entities"],
        "total_flow": sar["total_flow"],
        "confidence": sar["confidence"],
    }


# ── ML metrics + variants ──────────────────────────────────────────────────

@app.get("/api/ml/variants")
def list_variants():
    """Return all trained model variants with summary metrics."""
    out = ml_list_variants(DATA_DIR)
    # Add GNN metrics where present
    enriched = []
    for v in out:
        gnn = load_gnn_metrics(DATA_DIR, variant=v["variant"])
        if gnn:
            v["gnn"] = {
                "f1": gnn.get("f1"),
                "auc": gnn.get("auc"),
                "precision": gnn.get("precision"),
                "recall": gnn.get("recall"),
            }
        enriched.append(v)
    return {"variants": enriched}


@app.get("/api/ml/metrics")
def get_ml_metrics(variant: str = "synthetic"):
    m = ml_load_metrics(DATA_DIR, variant=variant)
    if not m:
        return {"trained": False, "variant": variant,
                "message": f"Model variant '{variant}' not trained."}
    gnn_m = load_gnn_metrics(DATA_DIR, variant=variant)
    return {"trained": True, **m, "gnn": gnn_m or None}


@app.get("/api/ml/tabular")
def get_tabular_baseline():
    """IEEE-CIS tabular baseline metrics (separate from graph models)."""
    path = os.path.join(DATA_DIR, "ml", "ieee_cis_tabular", "metrics.json")
    if not os.path.exists(path):
        from real_data_loader import available_datasets
        avail = available_datasets().get("ieee_cis", {})
        return {
            "trained": False,
            "message": "IEEE-CIS dataset not present.",
            "download_url": avail.get("download_url"),
            "expected_dir": avail.get("expected_dir"),
        }
    with open(path) as f:
        return {"trained": True, **json.load(f)}


@app.post("/api/ml/retrain")
def retrain_ml(role: str = Depends(get_role)):
    require("ml.retrain", role)
    m = ml_train_and_save(state["graph"], state["transactions"], DATA_DIR,
                          variant="synthetic", dataset_name="RUDRA Synthetic Generator")
    state["ml_metrics"] = ml_load_metrics(DATA_DIR, variant="synthetic")
    state["edge_scores"] = ml_load_edge_scores(DATA_DIR, variant="synthetic")
    state["ml_bundle"] = ml_load_model(DATA_DIR, variant="synthetic")
    return {"status": "ok", "metrics": m}


# ── Journey Tracer ─────────────────────────────────────────────────────────

@app.get("/api/journey/{entity_id}")
def get_journey(
    entity_id: str,
    direction: str = "both",
    hops: int = 3,
    min_amount: float = 0,
):
    if not state["graph"].has_node(entity_id):
        raise HTTPException(404, "Entity not found")
    return trace_journey(
        state["graph"], state["transactions"], state["risk_scores"],
        entity_id=entity_id, direction=direction, max_hops=hops,
        min_amount=min_amount, edge_ml_scores=state["edge_scores"],
    )


@app.get("/api/journey/alert/{alert_id}")
def get_journey_for_alert(alert_id: str, include_neighbors: bool = False):
    alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
    if not alert:
        raise HTTPException(404, "Alert not found")
    return trace_for_alert(
        state["graph"], state["transactions"], state["risk_scores"],
        alert=alert, edge_ml_scores=state["edge_scores"],
        include_neighbors=include_neighbors,
    )


# ── Case Workbench ─────────────────────────────────────────────────────────

@app.get("/api/cases")
def list_cases(status: Optional[str] = None):
    cases = state["cases"].list(status=status)
    return {"cases": cases, "total": len(cases)}


@app.get("/api/cases/{alert_id}")
def get_case(alert_id: str):
    case = state["cases"].get(alert_id)
    if not case:
        alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
        if not alert:
            raise HTTPException(404, "Case / alert not found")
        case = state["cases"].open_case(alert)
    return case


@app.post("/api/cases/{alert_id}/dispose")
def dispose_case(alert_id: str, body: dict, role: str = Depends(get_role)):
    status = (body.get("status") or "").upper()
    if status not in VALID_STATUSES:
        raise HTTPException(400, f"Invalid status. Must be one of {sorted(VALID_STATUSES)}")
    require(f"case.move.{status}", role)

    if not state["cases"].get(alert_id):
        alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
        if not alert:
            raise HTTPException(404, "Alert not found")
        state["cases"].open_case(alert)
    case = state["cases"].dispose(
        alert_id, status,
        note=body.get("note", ""),
        author=body.get("author", role.lower()),
        assigned_to=body.get("assigned_to"),
    )
    return case


@app.post("/api/cases/{alert_id}/note")
def add_case_note(alert_id: str, body: dict, role: str = Depends(get_role)):
    require("case.note", role)
    note = body.get("note", "")
    if not note:
        raise HTTPException(400, "Note text is required")
    if not state["cases"].get(alert_id):
        alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
        if not alert:
            raise HTTPException(404, "Alert not found")
        state["cases"].open_case(alert)
    return state["cases"].add_note(alert_id, note=note, author=body.get("author", role.lower()))


@app.get("/api/cases/{alert_id}/verify")
def verify_case_chain(alert_id: str, role: str = Depends(get_role)):
    require("audit.verify", role)
    if not state["cases"].get(alert_id):
        raise HTTPException(404, "Case not found")
    return state["cases"].verify_chain(alert_id)


# ── FIU Evidence Package ───────────────────────────────────────────────────

@app.get("/api/fiu/package/{alert_id}")
def download_fiu_package(alert_id: str, role: str = Depends(get_role)):
    require("fiu.download", role)
    alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
    if not alert:
        raise HTTPException(404, "Alert not found")
    sar_dir = os.path.join(DATA_DIR, "sar_reports")
    sar_pdf_path = None
    if os.path.isdir(sar_dir):
        sar = state["sar_gen"].generate_sar(alert)
        sar_pdf_path = state["sar_gen"].export_sar_pdf(sar, sar_dir)
    case = state["cases"].get(alert_id)
    zip_bytes = build_fiu_package(
        state["graph"], state["transactions"], alert, sar_pdf_path, case=case,
    )
    return StreamingResponse(
        BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="FIU_Package_{alert_id}.zip"'},
    )


# ── Config (threshold tuning) ──────────────────────────────────────────────

@app.get("/api/config/thresholds")
def get_thresholds(role: str = Depends(get_role)):
    require("config.read", role)
    return {
        "current": state["config"].get_all(),
        "defaults": DEFAULT_CONFIG,
    }


@app.post("/api/config/thresholds")
def update_thresholds(body: dict, role: str = Depends(get_role)):
    require("config.write", role)
    try:
        updated = state["config"].set_many(body or {})
        return {"status": "ok", "current": updated}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/config/thresholds/reset")
def reset_thresholds(role: str = Depends(get_role)):
    require("config.write", role)
    return {"status": "ok", "current": state["config"].reset()}


@app.post("/api/config/rerun")
def rerun_detection(role: str = Depends(get_role)):
    """Re-run all detectors using the current config thresholds. Heavy operation."""
    require("config.write", role)
    cfg = state["config"].get_all()
    det = FraudDetector(state["graph"], config=cfg)
    results = det.run_all_detections()
    dormant_alerts = DormantActivationDetector(state["graph"], state["transactions"], config=cfg).detect()
    risk_scores_data = []
    for node_id, score in results["node_risk_scores"].items():
        nd = dict(state["graph"].nodes[node_id])
        risk_scores_data.append({
            "entity_id": node_id, "name": nd.get("name", ""),
            "type": nd.get("type", ""), "risk_score": score,
            "risk_level": ("CRITICAL" if score >= 0.7 else "HIGH" if score >= 0.5
                            else "MEDIUM" if score >= 0.3 else "LOW"),
        })
    profile_alerts = ProfileMismatchDetector(state["graph"], state["transactions"], risk_scores_data).detect()
    all_alerts = results["all_alerts"] + dormant_alerts + profile_alerts
    results["all_alerts"] = all_alerts
    results["dormant_activation"] = dormant_alerts
    results["profile_mismatch"] = profile_alerts
    results["summary"]["total_alerts"] = len(all_alerts)
    results["summary"]["critical_alerts"] = sum(1 for a in all_alerts if a["severity"] == "CRITICAL")
    results["summary"]["high_alerts"] = sum(1 for a in all_alerts if a["severity"] == "HIGH")
    results["summary"]["medium_alerts"] = sum(1 for a in all_alerts if a["severity"] == "MEDIUM")
    det.save_results(results, DATA_DIR)

    incidents = cluster_alerts(all_alerts, graph=state["graph"])
    with open(os.path.join(DATA_DIR, "incidents.json"), "w") as f:
        json.dump(incidents, f, indent=2, default=str)
    state["alerts"] = all_alerts
    state["incidents"] = incidents
    state["risk_scores"] = risk_scores_data
    return {
        "status": "ok",
        "alert_count": len(all_alerts),
        "incident_count": len(incidents),
        "summary": results["summary"],
    }


# ── Analytics ──────────────────────────────────────────────────────────────

@app.get("/api/analytics/channels")
def analytics_channels():
    df = state["transactions"]
    if "channel" not in df.columns:
        return {"by_channel": [], "by_rail": [], "by_hour": []}

    by_channel = (
        df.groupby("channel").agg(
            count=("amount", "count"),
            volume=("amount", "sum"),
            fraud_count=("is_fraud", "sum"),
            fraud_volume=("amount", lambda x: x[df.loc[x.index, "is_fraud"]].sum()),
        ).reset_index()
    )
    by_channel["fraud_rate"] = (by_channel["fraud_count"] / by_channel["count"] * 100).round(2)
    by_channel = by_channel.sort_values("volume", ascending=False).to_dict("records")

    by_rail = (
        df.groupby("transaction_type").agg(
            count=("amount", "count"),
            volume=("amount", "sum"),
            fraud_count=("is_fraud", "sum"),
        ).reset_index()
    )
    by_rail["fraud_rate"] = (by_rail["fraud_count"] / by_rail["count"] * 100).round(2)
    by_rail = by_rail.sort_values("volume", ascending=False).to_dict("records")

    by_hour = df.groupby(df["timestamp"].dt.hour).agg(
        count=("amount", "count"),
        fraud_count=("is_fraud", "sum"),
    ).reset_index()
    by_hour.columns = ["hour", "count", "fraud_count"]
    by_hour = by_hour.to_dict("records")
    return {"by_channel": by_channel, "by_rail": by_rail, "by_hour": by_hour}


@app.get("/api/analytics/branches")
def analytics_branches():
    df = state["transactions"]
    fraud_df = df[df["is_fraud"]]
    sender_view = df.groupby("sender_branch").agg(
        out_volume=("amount", "sum"),
        out_count=("amount", "count"),
        out_fraud=("is_fraud", "sum"),
    ).rename_axis("branch").reset_index()
    receiver_view = df.groupby("receiver_branch").agg(
        in_volume=("amount", "sum"),
        in_count=("amount", "count"),
        in_fraud=("is_fraud", "sum"),
    ).rename_axis("branch").reset_index()
    sender_fraud = fraud_df.groupby("sender_branch")["amount"].sum().rename("out_fraud_volume").reset_index().rename(columns={"sender_branch": "branch"})
    receiver_fraud = fraud_df.groupby("receiver_branch")["amount"].sum().rename("in_fraud_volume").reset_index().rename(columns={"receiver_branch": "branch"})

    merged = (
        sender_view.merge(receiver_view, on="branch", how="outer")
        .merge(sender_fraud, on="branch", how="left")
        .merge(receiver_fraud, on="branch", how="left")
        .fillna(0)
    )
    merged["total_volume"] = merged["in_volume"] + merged["out_volume"]
    merged["total_fraud_count"] = merged["in_fraud"] + merged["out_fraud"]
    merged["total_fraud_volume"] = merged["in_fraud_volume"] + merged["out_fraud_volume"]
    merged["fraud_rate"] = (
        merged["total_fraud_count"]
        / (merged["in_count"] + merged["out_count"]).clip(lower=1)
        * 100
    ).round(2)
    merged = merged.sort_values("total_volume", ascending=False)
    return {"branches": merged.to_dict("records")}


@app.get("/api/analytics/products")
def analytics_products():
    df = state["transactions"]
    if "sender_product" not in df.columns:
        return {"by_product": []}
    sender_view = df.groupby("sender_product").agg(
        out_volume=("amount", "sum"),
        out_count=("amount", "count"),
        out_fraud=("is_fraud", "sum"),
    ).rename_axis("product").reset_index()
    receiver_view = df.groupby("receiver_product").agg(
        in_volume=("amount", "sum"),
        in_count=("amount", "count"),
        in_fraud=("is_fraud", "sum"),
    ).rename_axis("product").reset_index()
    merged = pd.merge(sender_view, receiver_view, on="product", how="outer").fillna(0)
    merged["total_volume"] = merged["in_volume"] + merged["out_volume"]
    merged["total_fraud"] = merged["in_fraud"] + merged["out_fraud"]
    return {"by_product": merged.sort_values("total_volume", ascending=False).to_dict("records")}


# ── Live Mode ──────────────────────────────────────────────────────────────

@app.get("/api/live/inject")
def inject_transactions(count: int = 10):
    """Simulate `count` incoming transactions and score each one through the
    live ML model. Per-transaction latency is included in the response."""
    graph = state["graph"]
    df = state["transactions"]
    bundle = state["ml_bundle"]
    entities = list(graph.nodes())

    feed = []
    for _ in range(count):
        sender = random.choice(entities)
        receiver = random.choice(entities)
        while receiver == sender:
            receiver = random.choice(entities)

        is_fraud = random.random() < 0.20
        if is_fraud:
            amount = round(random.uniform(500_000, 5_000_000), 2)
        else:
            amount = round(random.lognormvariate(np.log(50_000), 1.2), 2)

        sdata = graph.nodes[sender]; rdata = graph.nodes[receiver]
        rail = random.choice(["NEFT", "RTGS", "IMPS", "UPI"])
        channel = random.choice(["MobileApp", "NetBanking", "Branch", "ATM"])

        scoring = None
        ml_score = None
        latency = None
        if bundle is not None:
            try:
                scoring = score_live_txn(bundle, graph, sender, receiver, amount,
                                           channel, rail, pd.Timestamp.now())
                ml_score = scoring["ml_score"]
                latency = scoring["latency_ms"]
            except Exception as e:
                latency = {"error": str(e)}

        pattern = "none"
        if is_fraud:
            if amount > 2_000_000 and (sdata.get("type") == "shell_company"
                                         or rdata.get("type") == "shell_company"):
                pattern = "shell_funnel"
            elif amount > 1_000_000:
                pattern = random.choice(["rapid_layering", "circular_transaction"])
            elif amount < 200_000:
                pattern = "smurfing"

        severity = None
        if is_fraud or (ml_score and ml_score > 0.5):
            score_for_sev = ml_score if ml_score is not None else 0
            if amount > 3_000_000 or score_for_sev > 0.8:
                severity = "CRITICAL"
            elif amount > 1_000_000 or score_for_sev > 0.6:
                severity = "HIGH"
            else:
                severity = "MEDIUM"

        feed.append({
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sdata.get("name", sender),
            "receiver": rdata.get("name", receiver),
            "sender_id": sender, "receiver_id": receiver,
            "amount": amount,
            "transaction_type": rail, "channel": channel,
            "isFraud": is_fraud,
            "pattern": pattern, "severity": severity,
            "mlScore": ml_score,
            "latency_ms": latency,
        })

    return {"transactions": feed, "count": len(feed)}


@app.get("/api/benchmark/latency")
def benchmark_latency():
    """Time the full pipeline + per-txn ML scoring."""
    if state["ml_bundle"] is None:
        return {"error": "ML model not loaded."}
    return benchmark_pipeline(state["graph"], state["transactions"], state["ml_bundle"])


# ── Account Aggregator + KYC mocks ─────────────────────────────────────────

@app.post("/api/aa/consent")
def aa_consent(body: dict):
    return aa_create_consent(
        customer_id=body.get("customer_id", "CUST-000"),
        fip_ids=body.get("fip_ids", ["FIP-HDFC", "FIP-AXIS"]),
        purpose_code=body.get("purpose_code", "103"),
        duration_days=body.get("duration_days", 30),
    )


@app.get("/api/aa/consents")
def aa_consents():
    return {"consents": aa_list_consents()}


@app.get("/api/aa/pull/{consent_handle}")
def aa_pull(consent_handle: str, days_back: int = 30):
    return aa_pull_data(consent_handle, days_back=days_back)


@app.post("/api/aa/revoke/{consent_handle}")
def aa_revoke(consent_handle: str):
    return aa_revoke_consent(consent_handle)


@app.get("/api/kyc/screen")
def kyc_screen(name: str, entity_type: str = "individual"):
    return dilisense_screen(name, entity_type)


# ── Pipeline trigger ───────────────────────────────────────────────────────

@app.post("/api/pipeline/run")
def trigger_pipeline(role: str = Depends(get_role)):
    require("pipeline.run", role)
    run_pipeline()
    state["loaded"] = False
    load_or_generate()
    return {"status": "ok", "message": "Pipeline completed"}
