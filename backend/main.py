"""
RUDRA — FastAPI Backend

REST API serving the React frontend. Wires together:
  - synthetic data + pipeline (loaded on startup)
  - the live fund-flow graph
  - the 6 heuristic detectors + the XGBoost classifier
  - case workflow (open / investigate / SAR / dismiss / escalate)
  - fund journey tracer
  - FIU evidence package builder
  - LLM copilot
  - analytics + live-mode endpoints
"""

import os
import sys
import json
import random
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

from fastapi import FastAPI, Query, HTTPException
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
    load_metrics as ml_load_metrics,
    load_edge_scores as ml_load_edge_scores,
)
from fund_tracer import trace_journey, trace_for_alert
from case_manager import CaseStore, VALID_STATUSES
from fiu_package import build_package as build_fiu_package

app = FastAPI(title="RUDRA API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

state = {
    "transactions": None,
    "alerts": None,
    "risk_scores": None,
    "summary": None,
    "fraud_cases": None,
    "ffg": None,
    "graph": None,
    "copilot": None,
    "sar_gen": None,
    "cases": None,           # CaseStore
    "ml_metrics": None,
    "edge_scores": None,
    "loaded": False,
}


# ── Pipeline + data loading ────────────────────────────────────

def run_pipeline():
    """Generate data, build graph, run all detectors, train ML, persist outputs."""
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

    # Train ML model
    try:
        ml_train_and_save(graph, df, DATA_DIR)
    except Exception as e:
        print(f"ML training failed: {e}")

    # Generate SARs for HIGH+
    sar_gen = SARGenerator(graph, df, all_alerts, fraud_cases)
    sar_dir = os.path.join(DATA_DIR, "sar_reports")
    os.makedirs(sar_dir, exist_ok=True)
    for sar in sar_gen.generate_all_sars(min_severity="HIGH"):
        sar_gen.export_sar_pdf(sar, sar_dir)

    state["loaded"] = False


def load_or_generate():
    """Load existing data or run pipeline if missing. Cache everything in `state`."""
    if state["loaded"]:
        return

    alerts_path = os.path.join(DATA_DIR, "fraud_alerts.json")
    txn_path = os.path.join(DATA_DIR, "transactions.csv")
    if not os.path.exists(alerts_path) or not os.path.exists(txn_path):
        print("Data not found. Running pipeline...")
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

    state["ffg"] = FundFlowGraph()
    state["graph"] = state["ffg"].build_graph(df)

    state["copilot"] = LLMCopilot(
        state["graph"], df, state["alerts"], state["risk_scores"], state["fraud_cases"],
    )
    state["sar_gen"] = SARGenerator(
        state["graph"], df, state["alerts"], state["fraud_cases"],
    )

    # Cases — open a case for every alert that doesn't already have one
    case_store = CaseStore(DATA_DIR)
    case_store.bulk_open_all(state["alerts"])
    state["cases"] = case_store

    # ML artifacts — train on first run if missing
    state["ml_metrics"] = ml_load_metrics(DATA_DIR)
    state["edge_scores"] = ml_load_edge_scores(DATA_DIR)
    if not state["ml_metrics"]:
        try:
            ml_train_and_save(state["graph"], df, DATA_DIR)
            state["ml_metrics"] = ml_load_metrics(DATA_DIR)
            state["edge_scores"] = ml_load_edge_scores(DATA_DIR)
        except Exception as e:
            print(f"Inline ML training skipped: {e}")

    state["loaded"] = True
    print(f"[ready] {len(df)} txns, {len(state['alerts'])} alerts, "
          f"{state['graph'].number_of_nodes()} entities, "
          f"ML F1: {state['ml_metrics'].get('f1', 0):.3f}")


@app.on_event("startup")
def startup():
    load_or_generate()


# ── Helpers ────────────────────────────────────────────────────

def _alerts_with_case_status():
    """Decorate alerts with current case status + ML score for the highest-amount edge."""
    cases = state["cases"]
    edge_scores = state["edge_scores"] or {}
    graph = state["graph"]
    out = []
    for a in state["alerts"]:
        entities = a.get("entities", [])
        case = cases.get(a.get("alert_id"))
        ml_score = None
        # Find the highest-amount edge among the alert entities and use its ML score
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
        out.append(decorated)
    return out


# ── Dashboard ──────────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard():
    df = state["transactions"]
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

    pattern_breakdown = fraud_txns.groupby("fraud_pattern").agg(
        count=("amount", "count"),
        total=("amount", "sum"),
    ).reset_index().to_dict("records")

    risk_dist = {}
    for r in state["risk_scores"]:
        level = r["risk_level"]
        risk_dist[level] = risk_dist.get(level, 0) + 1

    case_status_counts = state["cases"].status_counts(alerts)
    ml = state["ml_metrics"] or {}

    # Real amount distribution by bucket
    buckets = [
        ("<₹50K", 0, 50000),
        ("₹50K-2L", 50000, 200000),
        ("₹2L-10L", 200000, 1000000),
        ("₹10L-50L", 1000000, 5000000),
        (">₹50L", 5000000, float("inf")),
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
            "model_f1": ml.get("f1"),
            "model_auc": ml.get("auc"),
        },
        "daily_data": daily.to_dict("records"),
        "pattern_breakdown": pattern_breakdown,
        "risk_distribution": risk_dist,
        "case_status_counts": case_status_counts,
        "amount_distribution": amount_distribution,
    }


# ── Graph ──────────────────────────────────────────────────────

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
            "source": u,
            "target": v,
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


# ── Alerts ─────────────────────────────────────────────────────

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


# ── Patterns ───────────────────────────────────────────────────

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
        cols = ["timestamp", "sender_name", "receiver_name", "amount", "transaction_type",
                "channel", "sender_type", "receiver_type", "sender_branch", "fraud_case_id"]
        cols = [c for c in cols if c in txns.columns]
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


# ── Entities ───────────────────────────────────────────────────

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


# ── Copilot ────────────────────────────────────────────────────

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


# ── SAR Reports ────────────────────────────────────────────────

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


# ── ML metrics ─────────────────────────────────────────────────

@app.get("/api/ml/metrics")
def get_ml_metrics():
    metrics = state["ml_metrics"] or {}
    if not metrics:
        return {"trained": False, "message": "ML model has not been trained yet."}
    return {"trained": True, **metrics}


@app.post("/api/ml/retrain")
def retrain_ml():
    metrics = ml_train_and_save(state["graph"], state["transactions"], DATA_DIR)
    state["ml_metrics"] = metrics
    state["edge_scores"] = ml_load_edge_scores(DATA_DIR)
    return {"status": "ok", "metrics": metrics}


# ── Journey Tracer ─────────────────────────────────────────────

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


# ── Case Workbench ─────────────────────────────────────────────

@app.get("/api/cases")
def list_cases(status: Optional[str] = None):
    cases = state["cases"].list(status=status)
    return {"cases": cases, "total": len(cases)}


@app.get("/api/cases/{alert_id}")
def get_case(alert_id: str):
    case = state["cases"].get(alert_id)
    if not case:
        # Auto-open from alert
        alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
        if not alert:
            raise HTTPException(404, "Case / alert not found")
        case = state["cases"].open_case(alert)
    return case


@app.post("/api/cases/{alert_id}/dispose")
def dispose_case(alert_id: str, body: dict):
    status = (body.get("status") or "").upper()
    if status not in VALID_STATUSES:
        raise HTTPException(400, f"Invalid status. Must be one of {sorted(VALID_STATUSES)}")
    note = body.get("note", "")
    author = body.get("author", "investigator")
    assigned_to = body.get("assigned_to")

    # Ensure case exists
    if not state["cases"].get(alert_id):
        alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
        if not alert:
            raise HTTPException(404, "Alert not found")
        state["cases"].open_case(alert)
    case = state["cases"].dispose(alert_id, status, note=note, author=author, assigned_to=assigned_to)
    return case


@app.post("/api/cases/{alert_id}/note")
def add_case_note(alert_id: str, body: dict):
    note = body.get("note", "")
    author = body.get("author", "investigator")
    if not note:
        raise HTTPException(400, "Note text is required")
    if not state["cases"].get(alert_id):
        alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
        if not alert:
            raise HTTPException(404, "Alert not found")
        state["cases"].open_case(alert)
    case = state["cases"].add_note(alert_id, note=note, author=author)
    return case


# ── FIU Evidence Package ───────────────────────────────────────

@app.get("/api/fiu/package/{alert_id}")
def download_fiu_package(alert_id: str):
    alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
    if not alert:
        raise HTTPException(404, "Alert not found")

    # If there's a SAR PDF on disk for this alert, include it
    sar_dir = os.path.join(DATA_DIR, "sar_reports")
    sar_pdf_path = None
    if os.path.isdir(sar_dir):
        # SARs are named by their report_id, which we don't know without regenerating.
        # Generate it on the fly so the zip always contains a fresh PDF.
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


# ── Channel / Branch / Product Analytics ───────────────────────

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


# ── Live Mode ──────────────────────────────────────────────────

@app.get("/api/live/inject")
def inject_transactions(count: int = 10):
    """Generate `count` simulated transactions and return them with detection
    results applied per-transaction using the trained model. This is the
    streaming hook for the Live tab on the dashboard."""
    graph = state["graph"]
    edge_scores = state["edge_scores"] or {}
    df = state["transactions"]
    entities = list(graph.nodes())

    feed = []
    for _ in range(count):
        sender = random.choice(entities)
        receiver = random.choice(entities)
        while receiver == sender:
            receiver = random.choice(entities)

        # Bias toward fraud to keep the feed interesting
        is_fraud = random.random() < 0.20
        if is_fraud:
            amount = round(random.uniform(500000, 5000000), 2)
        else:
            amount = round(random.lognormvariate(np.log(50000), 1.2), 2)

        sdata = graph.nodes[sender]
        rdata = graph.nodes[receiver]
        rail = random.choice(["NEFT", "RTGS", "IMPS", "UPI"])
        channel = random.choice(["MobileApp", "NetBanking", "Branch", "ATM"])

        # Pull existing ML score for this counterparty pair if we have one;
        # this approximates "ML scored the txn at sub-200ms latency"
        ml_score = edge_scores.get(f"{sender}->{receiver}")

        # Choose a pattern label based on heuristic structure
        pattern = "none"
        if is_fraud:
            if amount > 2000000 and (sdata.get("type") == "shell_company"
                                       or rdata.get("type") == "shell_company"):
                pattern = "shell_funnel"
            elif amount > 1000000:
                pattern = random.choice(["rapid_layering", "circular_transaction"])
            elif amount < 200000:
                pattern = "smurfing"

        # Severity from amount + ML score
        severity = None
        if is_fraud:
            if amount > 3000000 or (ml_score or 0) > 0.8:
                severity = "CRITICAL"
            elif amount > 1000000 or (ml_score or 0) > 0.6:
                severity = "HIGH"
            else:
                severity = "MEDIUM"

        feed.append({
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sdata.get("name", sender),
            "receiver": rdata.get("name", receiver),
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "transaction_type": rail,
            "channel": channel,
            "isFraud": is_fraud,
            "pattern": pattern,
            "severity": severity,
            "mlScore": ml_score,
        })

    return {"transactions": feed, "count": len(feed)}


# ── Pipeline trigger ───────────────────────────────────────────

@app.post("/api/pipeline/run")
def trigger_pipeline():
    run_pipeline()
    state["loaded"] = False
    load_or_generate()
    return {"status": "ok", "message": "Pipeline completed"}


@app.get("/")
def root():
    return {"name": "RUDRA API", "version": "2.0", "ml_trained": bool(state["ml_metrics"])}
