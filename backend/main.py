"""
RUDRA — FastAPI Backend
REST API serving the React frontend with data from the fraud detection pipeline.
"""

import sys
import os
import json
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional
import pandas as pd
import networkx as nx

from data_generator import TransactionGenerator, save_data
from graph_engine import FundFlowGraph
from fraud_detector import FraudDetector
from advanced_detectors import DormantActivationDetector, ProfileMismatchDetector
from sar_generator import SARGenerator
from llm_copilot import LLMCopilot

app = FastAPI(title="RUDRA API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Global State ──────────────────────────────────────────────
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
    "loaded": False,
}


def load_or_generate():
    """Load existing data or run pipeline."""
    if state["loaded"]:
        return

    alerts_path = os.path.join(DATA_DIR, "fraud_alerts.json")
    txn_path = os.path.join(DATA_DIR, "transactions.csv")

    if not os.path.exists(alerts_path) or not os.path.exists(txn_path):
        print("Data not found. Running pipeline...")
        run_pipeline()

    # Load data
    state["transactions"] = pd.read_csv(txn_path)
    state["transactions"]["timestamp"] = pd.to_datetime(state["transactions"]["timestamp"])

    with open(os.path.join(DATA_DIR, "fraud_alerts.json")) as f:
        state["alerts"] = json.load(f)
    with open(os.path.join(DATA_DIR, "risk_scores.json")) as f:
        state["risk_scores"] = json.load(f)
    with open(os.path.join(DATA_DIR, "detection_summary.json")) as f:
        state["summary"] = json.load(f)
    with open(os.path.join(DATA_DIR, "fraud_cases.json")) as f:
        state["fraud_cases"] = json.load(f)

    # Build graph
    state["ffg"] = FundFlowGraph()
    state["graph"] = state["ffg"].build_graph(state["transactions"])

    # Copilot & SAR generator
    state["copilot"] = LLMCopilot(
        state["graph"], state["transactions"], state["alerts"],
        state["risk_scores"], state["fraud_cases"],
    )
    state["sar_gen"] = SARGenerator(
        state["graph"], state["transactions"], state["alerts"], state["fraud_cases"],
    )

    state["loaded"] = True
    print(f"Loaded: {len(state['transactions'])} txns, {len(state['alerts'])} alerts, "
          f"{state['graph'].number_of_nodes()} entities")


def run_pipeline():
    """Run full data generation + detection pipeline."""
    generator = TransactionGenerator(seed=42)
    df, fraud_cases = generator.generate_all_data()
    save_data(df, fraud_cases, DATA_DIR)

    ffg = FundFlowGraph()
    graph = ffg.build_graph(df)

    detector = FraudDetector(graph)
    results = detector.run_all_detections()

    dormant_detector = DormantActivationDetector(graph, df)
    dormant_alerts = dormant_detector.detect()

    risk_scores_data = []
    for node_id, score in results["node_risk_scores"].items():
        node_data = dict(graph.nodes[node_id])
        risk_scores_data.append({
            "entity_id": node_id, "name": node_data.get("name", ""),
            "type": node_data.get("type", ""), "risk_score": score,
            "risk_level": "CRITICAL" if score >= 0.7 else "HIGH" if score >= 0.5 else "MEDIUM" if score >= 0.3 else "LOW",
        })

    profile_detector = ProfileMismatchDetector(graph, df, risk_scores_data)
    profile_alerts = profile_detector.detect()

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

    # Reset state so next request reloads
    state["loaded"] = False


@app.on_event("startup")
def startup():
    load_or_generate()


# ── Dashboard ─────────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard():
    df = state["transactions"]
    alerts = state["alerts"]
    summary = state["summary"]

    fraud_txns = df[df["is_fraud"]]
    total_volume = float(df["amount"].sum())
    fraud_volume = float(fraud_txns["amount"].sum())

    # Daily stats for charts
    daily = df.groupby(df["timestamp"].dt.date).agg(
        count=("amount", "count"),
        volume=("amount", "sum"),
        fraud_count=("is_fraud", "sum"),
        fraud_volume=("amount", lambda x: x[df.loc[x.index, "is_fraud"]].sum()),
    ).reset_index()
    daily.columns = ["date", "count", "volume", "fraud_count", "fraud_volume"]
    daily["date"] = daily["date"].astype(str)

    # Pattern breakdown
    pattern_breakdown = fraud_txns.groupby("fraud_pattern").agg(
        count=("amount", "count"),
        total=("amount", "sum"),
    ).reset_index()
    pattern_breakdown = pattern_breakdown.to_dict("records")

    # Risk distribution
    risk_dist = {}
    for r in state["risk_scores"]:
        level = r["risk_level"]
        risk_dist[level] = risk_dist.get(level, 0) + 1

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
        },
        "daily_data": daily.to_dict("records"),
        "pattern_breakdown": pattern_breakdown,
        "risk_distribution": risk_dist,
        "amount_stats": {
            "normal": df[~df["is_fraud"]]["amount"].describe().to_dict(),
            "fraud": fraud_txns["amount"].describe().to_dict(),
        },
    }


# ── Graph ─────────────────────────────────────────────────────

@app.get("/api/graph")
def get_graph(
    fraud_only: bool = False,
    min_amount: float = 0,
    high_risk_only: bool = False,
):
    graph = state["graph"]
    risk_map = {r["entity_id"]: r["risk_score"] for r in state["risk_scores"]}

    # Determine which nodes to include
    if high_risk_only:
        nodes_to_include = {r["entity_id"] for r in state["risk_scores"] if r["risk_score"] >= 0.3}
    else:
        nodes_to_include = set(graph.nodes())

    # Build fraud edges/nodes sets
    fraud_edges = {(u, v) for u, v, d in graph.edges(data=True) if d.get("fraud_count", 0) > 0}
    fraud_nodes = set()
    for u, v in fraud_edges:
        fraud_nodes.add(u)
        fraud_nodes.add(v)

    if fraud_only:
        nodes_to_include = nodes_to_include & fraud_nodes
        # Expand to include direct neighbors
        expanded = set(nodes_to_include)
        for n in nodes_to_include:
            expanded.update(graph.predecessors(n))
            expanded.update(graph.successors(n))
        nodes_to_include = expanded

    # Build filtered graph
    sub = graph.subgraph(nodes_to_include).copy()

    # Remove edges below min_amount
    edges_to_remove = [(u, v) for u, v, d in sub.edges(data=True) if d["total_amount"] < min_amount]
    sub.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    isolates = list(nx.isolates(sub))
    sub.remove_nodes_from(isolates)

    if sub.number_of_nodes() == 0:
        return {"nodes": [], "edges": []}

    # Format for react-force-graph
    nodes = []
    for node in sub.nodes():
        ndata = dict(sub.nodes[node])
        is_fraud = node in fraud_nodes
        risk = risk_map.get(node, 0)
        in_deg = sub.in_degree(node)
        out_deg = sub.out_degree(node)
        nodes.append({
            "id": node,
            "name": ndata.get("name", node),
            "type": ndata.get("type", "individual"),
            "branch": ndata.get("branch", ""),
            "isFraud": is_fraud,
            "riskScore": round(risk, 3),
            "degree": in_deg + out_deg,
            "val": max(3, min(20, (in_deg + out_deg) * 0.5)),
        })

    edges = []
    for u, v, data in sub.edges(data=True):
        is_fraud_edge = (u, v) in fraud_edges
        edges.append({
            "source": u,
            "target": v,
            "amount": round(data["total_amount"], 2),
            "txCount": data["transaction_count"],
            "avgAmount": round(data["avg_amount"], 2),
            "isFraud": is_fraud_edge,
            "fraudCount": data.get("fraud_count", 0),
        })

    return {"nodes": nodes, "links": edges}


@app.get("/api/graph/{entity_id}")
def get_subgraph(entity_id: str, hops: int = 2):
    graph = state["graph"]
    if not graph.has_node(entity_id):
        return JSONResponse({"error": "Entity not found"}, status_code=404)

    sub = state["ffg"].extract_subgraph([entity_id], hops=hops)
    risk_map = {r["entity_id"]: r["risk_score"] for r in state["risk_scores"]}

    fraud_edges = {(u, v) for u, v, d in sub.edges(data=True) if d.get("fraud_count", 0) > 0}
    fraud_nodes = set()
    for u, v in fraud_edges:
        fraud_nodes.add(u)
        fraud_nodes.add(v)

    nodes = []
    for node in sub.nodes():
        ndata = dict(sub.nodes[node])
        nodes.append({
            "id": node,
            "name": ndata.get("name", node),
            "type": ndata.get("type", "individual"),
            "branch": ndata.get("branch", ""),
            "isFraud": node in fraud_nodes,
            "riskScore": round(risk_map.get(node, 0), 3),
            "degree": sub.degree(node),
            "val": max(3, min(15, sub.degree(node) * 0.5)),
        })

    edges = []
    for u, v, data in sub.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "amount": round(data["total_amount"], 2),
            "txCount": data["transaction_count"],
            "isFraud": (u, v) in fraud_edges,
        })

    # Entity detail
    ndata = dict(graph.nodes[entity_id])
    in_str = sum(graph[u][entity_id]["total_amount"] for u in graph.predecessors(entity_id))
    out_str = sum(graph[entity_id][v]["total_amount"] for v in graph.successors(entity_id))

    return {
        "entity": {
            "id": entity_id,
            "name": ndata.get("name", entity_id),
            "type": ndata.get("type", ""),
            "branch": ndata.get("branch", ""),
            "riskScore": round(risk_map.get(entity_id, 0), 3),
            "inflow": round(in_str, 2),
            "outflow": round(out_str, 2),
            "netFlow": round(in_str - out_str, 2),
        },
        "nodes": nodes,
        "links": edges,
    }


# ── Alerts ────────────────────────────────────────────────────

@app.get("/api/alerts")
def get_alerts(
    severity: Optional[str] = None,
    pattern: Optional[str] = None,
):
    alerts = state["alerts"]
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    if pattern:
        alerts = [a for a in alerts if pattern.lower() in a.get("pattern_type", "").lower()]
    return {"alerts": alerts, "total": len(alerts)}


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: str):
    alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
    if not alert:
        return JSONResponse({"error": "Alert not found"}, status_code=404)
    return alert


# ── Patterns ──────────────────────────────────────────────────

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
    if pattern_fraud and pattern_fraud in ["circular_transaction", "rapid_layering", "smurfing", "shell_funnel"]:
        txns = fraud_txns[fraud_txns["fraud_pattern"] == pattern_fraud]
        alerts = [a for a in state["alerts"] if pattern_fraud.replace("_", " ").title() in a.get("pattern_type", "")
                  or pattern_fraud in a.get("pattern_type", "").lower()]

        txn_list = txns[["timestamp", "sender_name", "receiver_name", "amount",
                         "transaction_type", "sender_type", "receiver_type",
                         "sender_branch", "fraud_case_id"]].sort_values(
            ["fraud_case_id", "timestamp"]).to_dict("records")

        return {
            "pattern": pattern_type,
            "alerts": alerts,
            "transactions": txn_list,
            "total_volume": round(float(txns["amount"].sum()), 2) if len(txns) > 0 else 0,
            "total_transactions": len(txns),
        }

    # Dormant / Profile patterns
    alert_type = "Dormant Activation" if pattern_type == "dormant" else "Profile Mismatch"
    alerts = [a for a in state["alerts"] if a.get("pattern_type") == alert_type]
    return {"pattern": pattern_type, "alerts": alerts, "transactions": [], "total_volume": 0, "total_transactions": 0}


# ── Entities ──────────────────────────────────────────────────

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
        return JSONResponse({"error": "Entity not found"}, status_code=404)

    ndata = dict(graph.nodes[entity_id])
    risk_info = next((r for r in state["risk_scores"] if r["entity_id"] == entity_id), {})

    entity_txns = df[(df["sender_id"] == entity_id) | (df["receiver_id"] == entity_id)]
    sent = entity_txns[entity_txns["sender_id"] == entity_id]
    received = entity_txns[entity_txns["receiver_id"] == entity_id]

    txn_history = entity_txns[["timestamp", "sender_name", "receiver_name", "amount",
                                "transaction_type", "is_fraud", "fraud_pattern"]].sort_values(
        "timestamp", ascending=False).head(50).to_dict("records")

    in_str = sum(graph[u][entity_id]["total_amount"] for u in graph.predecessors(entity_id))
    out_str = sum(graph[entity_id][v]["total_amount"] for v in graph.successors(entity_id))

    return {
        "id": entity_id,
        "name": ndata.get("name", entity_id),
        "type": ndata.get("type", ""),
        "branch": ndata.get("branch", ""),
        "riskScore": risk_info.get("risk_score", 0),
        "riskLevel": risk_info.get("risk_level", "N/A"),
        "inflow": round(in_str, 2),
        "outflow": round(out_str, 2),
        "netFlow": round(in_str - out_str, 2),
        "totalTransactions": len(entity_txns),
        "fraudTransactions": len(entity_txns[entity_txns["is_fraud"]]),
        "sentVolume": round(float(sent["amount"].sum()), 2),
        "receivedVolume": round(float(received["amount"].sum()), 2),
        "transactionHistory": txn_history,
    }


# ── Copilot ───────────────────────────────────────────────────

@app.post("/api/copilot/query")
async def copilot_query(body: dict):
    query = body.get("query", "")
    if not query:
        return {"error": "Query is required"}

    result = state["copilot"].query(query)
    return {
        "response": result["response"],
        "source": result.get("source", "local"),
        "tool_calls": result.get("tool_calls", []),
    }


# ── SAR Reports ───────────────────────────────────────────────

@app.get("/api/sar/generate/{alert_id}")
def generate_sar(alert_id: str):
    alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
    if not alert:
        return JSONResponse({"error": "Alert not found"}, status_code=404)

    sar = state["sar_gen"].generate_sar(alert)
    return {
        "report_id": sar["report_id"],
        "report_text": sar["report_text"],
        "severity": sar["severity"],
        "pattern_type": sar["pattern_type"],
        "entities": sar["entities"],
        "total_flow": sar["total_flow"],
        "confidence": sar["confidence"],
    }


# ── Pipeline ──────────────────────────────────────────────────

@app.post("/api/pipeline/run")
def trigger_pipeline():
    run_pipeline()
    state["loaded"] = False
    load_or_generate()
    return {"status": "ok", "message": "Pipeline completed"}


# ── Live Simulation ───────────────────────────────────────────

@app.get("/api/live/inject")
def inject_transactions(count: int = 10):
    graph = state["graph"]
    entities_list = list(graph.nodes())
    txns = []

    for _ in range(count):
        sender = random.choice(entities_list)
        receiver = random.choice(entities_list)
        while receiver == sender:
            receiver = random.choice(entities_list)

        is_fraud = random.random() < 0.15
        amount = round(random.lognormvariate(np.log(100000), 1.5), 2)
        if is_fraud:
            amount = round(random.uniform(500000, 5000000), 2)

        sdata = graph.nodes[sender]
        rdata = graph.nodes[receiver]

        txns.append({
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
            "sender": sdata.get("name", sender),
            "receiver": rdata.get("name", receiver),
            "amount": amount,
            "type": random.choice(["NEFT", "RTGS", "IMPS", "UPI"]),
            "isFraud": is_fraud,
            "pattern": random.choice(["circular", "layering", "smurfing", "none"]) if is_fraud else "none",
            "severity": "CRITICAL" if amount > 3000000 else "HIGH" if is_fraud else None,
        })

    return {"transactions": txns, "count": len(txns)}
