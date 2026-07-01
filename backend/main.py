"""
RUDRA — FastAPI Backend

The investigator-facing API. Wires together (per active RUDRA_DATASET):
  - 6 heuristic detectors + stacked ensemble (XGBoost + GraphSAGE + GAT)
  - SHAP local explanations per alert
  - Case workflow on SQLite with hash-chain audit log
  - Threshold-config layer + RBAC roles
  - Live-mode per-txn ML scoring + latency benchmark
  - Fund journey tracer + incident clustering
  - FIU evidence-package generator (zip with STR XML, SAR PDF, etc.)
  - Sahamati AA + DiliSense KYC adapters (real when creds set, mock otherwise)
  - Kafka stream ingestor (real broker when reachable, in-process fallback)
  - LLM copilot (Claude Haiku when ANTHROPIC_API_KEY set, quick-commands otherwise)
"""

import os
import sys
import json
import time
from io import BytesIO
from typing import Optional, List, Dict

# Load .env from the backend directory before anything else reads os.environ
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

import pandas as pd
import networkx as nx

from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Make sibling src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
from train_tgn import load_tgn_metrics, load_tgn_predictions
from shap_explainer import explain_alert as shap_explain_alert, explain_edge
from fund_tracer import trace_journey, trace_for_alert
from rca_engine import build_rca
from taint_store import TaintStore
from fatf_typology import tag_alert
from geo import city_flows
from case_manager import CaseStore, VALID_STATUSES
from fiu_package import build_package as build_fiu_package
from incident_clustering import cluster_alerts, alert_to_incident_map
from config_store import ConfigStore, DEFAULT_CONFIG
from rbac import get_role, require, role_capabilities, VALID_ROLES
from live_scoring import score_live_txn
from integrations import AAClient, DilisenseClient
from streaming import get_ingestor, StreamTxn
from streaming.kafka_producer import replay_transactions
from dataset_config import (
    get_active_variant, variant_data_dir, required_artefacts, regenerate_command,
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

# Active dataset (RUDRA_DATASET env, default ibm_aml). VARIANT_DIR is where
# all per-dataset operational artefacts live: transactions.csv, fraud_alerts.json,
# risk_scores.json, fraud_cases.json, incidents.json, rudra.db, sar_reports/...
ACTIVE_VARIANT = get_active_variant()
VARIANT_DIR = variant_data_dir(DATA_DIR, ACTIVE_VARIANT)


# ── State ──────────────────────────────────────────────────────────────────────
state = {
    "variant": ACTIVE_VARIANT,   # which dataset the backend is bound to
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
    "ml_bundle": None,           # active-variant model bundle (used for SHAP + live scoring)
    "ml_metrics": None,
    "edge_scores": None,
    "aa_client": None,           # Sahamati Account Aggregator client (real or mock-backed)
    "dilisense_client": None,    # DiliSense KYC client (real or mock-backed)
    "ingestor": None,            # Real Kafka stream ingestor (or in-process fallback)
    "taint": None,               # TaintStore — persistent decaying taint that floors risk
    "ensemble_edge_scores": None,  # lazily-loaded per-edge {xgb,sage,gat,ensemble}
    "loaded": False,
}


# ── Pipeline integration ──────────────────────────────────────────────────────

def run_pipeline():
    """Regenerate everything for the active dataset variant.

    Driven by `python src/run_pipeline.py --dataset <variant>`. We shell out
    rather than reimplementing the orchestration here so this endpoint and
    the CLI stay in lock-step. The variant comes from RUDRA_DATASET; the
    pipeline writes to data/<variant>/ and data/ml/<variant>/.
    """
    import subprocess

    here = os.path.dirname(__file__)
    pipeline = os.path.join(here, "..", "src", "run_pipeline.py")
    env = dict(os.environ)
    env["RUDRA_DATASET"] = ACTIVE_VARIANT
    res = subprocess.run(
        [sys.executable, pipeline, "--dataset", ACTIVE_VARIANT],
        env=env,
        cwd=os.path.join(here, ".."),
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed for variant={ACTIVE_VARIANT}. "
            f"Run it manually to see the full error: {regenerate_command(ACTIVE_VARIANT)}"
        )
    state["loaded"] = False


def load_or_generate():
    if state["loaded"]:
        return

    os.makedirs(VARIANT_DIR, exist_ok=True)

    # Fail loud if the configured variant's artefacts are missing. The user
    # picked RUDRA_DATASET; they should see exactly which files are missing
    # and the command to regenerate them.
    missing = [p for p in required_artefacts(VARIANT_DIR) if not os.path.exists(p)]
    if missing:
        msg = (
            f"\n[backend] Variant '{ACTIVE_VARIANT}' is missing required files under {VARIANT_DIR}:\n"
            + "\n".join(f"    - {p}" for p in missing)
            + f"\n\nRegenerate with:\n    {regenerate_command(ACTIVE_VARIANT)}\n"
        )
        raise RuntimeError(msg)

    txn_path = os.path.join(VARIANT_DIR, "transactions.csv")
    df = pd.read_csv(txn_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    state["transactions"] = df

    with open(os.path.join(VARIANT_DIR, "fraud_alerts.json")) as f:
        state["alerts"] = json.load(f)
    with open(os.path.join(VARIANT_DIR, "risk_scores.json")) as f:
        state["risk_scores"] = json.load(f)
    with open(os.path.join(VARIANT_DIR, "detection_summary.json")) as f:
        state["summary"] = json.load(f)
    with open(os.path.join(VARIANT_DIR, "fraud_cases.json")) as f:
        state["fraud_cases"] = json.load(f)

    incidents_path = os.path.join(VARIANT_DIR, "incidents.json")
    if os.path.exists(incidents_path):
        with open(incidents_path) as f:
            state["incidents"] = json.load(f)
    else:
        state["incidents"] = []

    state["ffg"] = FundFlowGraph()
    state["graph"] = state["ffg"].build_graph(df)

    # ML artefacts — load BEFORE constructing the copilot so it can wire
    # SHAP-driven explanations into the explain_alert tool.
    state["ml_metrics"] = ml_load_metrics(DATA_DIR, variant=ACTIVE_VARIANT)
    state["edge_scores"] = ml_load_edge_scores(DATA_DIR, variant=ACTIVE_VARIANT)
    state["ml_bundle"] = ml_load_model(DATA_DIR, variant=ACTIVE_VARIANT)
    if state["ml_bundle"] is None:
        try:
            ml_train_and_save(state["graph"], df, DATA_DIR, variant=ACTIVE_VARIANT)
            state["ml_metrics"] = ml_load_metrics(DATA_DIR, variant=ACTIVE_VARIANT)
            state["edge_scores"] = ml_load_edge_scores(DATA_DIR, variant=ACTIVE_VARIANT)
            state["ml_bundle"] = ml_load_model(DATA_DIR, variant=ACTIVE_VARIANT)
        except Exception as e:
            print(f"[backend] inline ML training skipped: {e}")

    state["copilot"] = LLMCopilot(
        state["graph"], df, state["alerts"], state["risk_scores"], state["fraud_cases"],
        model_bundle=state.get("ml_bundle"),
    )
    state["sar_gen"] = SARGenerator(
        state["graph"], df, state["alerts"], state["fraud_cases"],
    )

    # Case store on SQLite — per-variant so cases belong to the dataset they
    # were opened against.
    case_store = CaseStore(VARIANT_DIR)
    case_store.bulk_open_all(state["alerts"])
    # Backfill incident_id on cases
    a2i = alert_to_incident_map(state["incidents"])
    for aid, inc_id in a2i.items():
        case_store.set_incident(aid, inc_id)
    state["cases"] = case_store

    # Config store (same SQLite file)
    state["config"] = ConfigStore(os.path.join(VARIANT_DIR, "rudra.db"))

    # Persistent taint memory (same SQLite file) — decaying taint seeded from
    # confirmed-fraud cases, floors future risk scores across pipeline runs.
    state["taint"] = TaintStore(os.path.join(VARIANT_DIR, "rudra.db"))

    # AA + DiliSense clients — pick up env-driven creds; fall back to mock when absent.
    state["aa_client"] = AAClient()
    state["dilisense_client"] = DilisenseClient()

    # Pre-build tracer auxiliary structures so per-entity flag lookups are O(1)
    # instead of recomputing from scratch on every /api/entities/{id} request.
    try:
        from fund_tracer import (
            _build_txn_index, _build_baseline_stats,
            _build_burst_counts, _build_transit_ratios,
        )
        state["_txn_index"] = _build_txn_index(df)
        state["_baseline_stats"] = _build_baseline_stats(df)
        state["_burst_counts"] = _build_burst_counts(df)
        state["_transit_ratios"] = _build_transit_ratios(df)
    except Exception as e:
        print(f"[backend] tracer cache build skipped: {e}")

    # Warm the caches that otherwise make the FIRST request slow: the memoised
    # SCC set (journey/replay), the ML feature context (SHAP in Simulation
    # Studio), and the ensemble edge-score file. One-time cost at boot instead
    # of a multi-second stall on the first user interaction.
    try:
        from fund_tracer import _scc3_members
        _scc3_members(state["graph"])
        from ml_model import _build_context as _warm_ml_ctx
        _warm_ml_ctx(state["graph"])
        _ensemble_edge_scores()
        if state.get("ml_bundle"):
            from shap_explainer import get_explainer
            get_explainer(state["ml_bundle"]["model"], state["ml_bundle"].get("background"))
        # Pre-warm the per-page views so the FIRST page load is instant — no
        # manual command, happens automatically at boot. Each fills its cache
        # once; requests then read the stored result (recompute only after an
        # invalidating mutation: dispose / retrain / rerun).
        _alerts_with_case_status()                                 # Cases
        get_dashboard()                                            # Dashboard
        geo_flows()                                                # Geo Map
        analytics_channels(); analytics_branches()                 # Channel/Branch
    except Exception as e:
        print(f"[backend] cache warm skipped: {e}")

    state["loaded"] = True
    aa_mode = "REAL" if state["aa_client"].is_real else "mock"
    kyc_mode = "REAL" if state["dilisense_client"].is_real else "mock"
    f1 = (state["ml_metrics"] or {}).get("f1", 0)
    print(f"[backend] ready: variant={ACTIVE_VARIANT}, {len(df)} txns, "
          f"{len(state['alerts'])} alerts, {len(state['incidents'])} incidents, "
          f"{state['graph'].number_of_nodes()} entities, ML F1={f1:.3f}, "
          f"AA={aa_mode}, KYC={kyc_mode}")


@app.on_event("startup")
def startup():
    load_or_generate()


@app.on_event("startup")
async def startup_streaming():
    """Start the Kafka ingestor (or fall back to in-process queue).

    Runs after `startup()` above because FastAPI executes hooks in registration
    order; the ingestor needs the graph + ml_bundle already loaded into state.
    Failing to start the stream isn't fatal — the rest of the API stays up.
    """
    ingestor = get_ingestor(
        score_fn=score_live_txn,
        graph_provider=lambda: state["graph"],
        bundle_provider=lambda: state["ml_bundle"],
    )
    state["ingestor"] = ingestor
    try:
        await ingestor.start()
        print(f"[backend] stream ingestor: {ingestor.status()['mode']} "
              f"(topic={ingestor.topic}, bootstrap={ingestor.bootstrap})")
    except Exception as e:
        print(f"[backend] stream ingestor failed to start: {e}")


@app.on_event("shutdown")
async def shutdown_streaming():
    ing = state.get("ingestor")
    if ing is not None:
        try:
            await ing.stop()
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_alerts_with_case_status() -> List[Dict]:
    """Decorate alerts with case status + ML score + incident id (UNCACHED).

    Everything except the cache wrapper should call _alerts_with_case_status().
    This is multi-second on ~8k alerts (a SQLite case lookup + tag_alert each),
    which is why the result is memoised.
    """
    cases = state["cases"]
    edge_scores = state["edge_scores"] or {}
    graph = state["graph"]
    a2i = alert_to_incident_map(state["incidents"] or [])
    out = []
    for a in state["alerts"]:
        entities = a.get("entities", [])
        case = cases.get(a.get("alert_id"))
        # ML-generated alerts already carry their ensemble score — keep it rather
        # than overwriting with the XGB edge-score max below.
        ml_score = a.get("ml_score")
        if ml_score is None and len(entities) >= 2 and graph is not None:
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
        # Funnel alerts persisted before total_flow was canonical carry only
        # total_inflow/total_outflow — derive total_flow so UI sums don't read 0.
        if decorated.get("total_flow") is None and (
            "total_inflow" in decorated or "total_outflow" in decorated
        ):
            decorated["total_flow"] = round(max(
                decorated.get("total_inflow", 0) or 0,
                decorated.get("total_outflow", 0) or 0,
            ), 2)
        decorated["case_status"] = case.get("status") if case else "OPEN"
        decorated["assigned_to"] = case.get("assigned_to") if case else None
        decorated["ml_score"] = ml_score
        decorated["incident_id"] = a2i.get(a.get("alert_id"))
        # FATF typology + Indian-regulatory refs + graded legal basis (additive).
        decorated = tag_alert(decorated)
        out.append(decorated)
    return out


def _alerts_with_case_status() -> List[Dict]:
    """Cached decorated-alert list — recomputed lazily after a cache invalidation.

    Read-only by contract: callers filter/select but must NOT mutate the returned
    dicts (they are shared cache entries). All current callers comply.
    """
    if state.get("_alerts_cache") is None:
        state["_alerts_cache"] = _compute_alerts_with_case_status()
    return state["_alerts_cache"]


def _invalidate_derived_caches() -> None:
    """Drop memoised views derived from alerts / cases / incidents / metrics.

    MUST be called after any mutation to those — a case dispose (changes
    case_status / assigned_to) or an ML retrain (replaces alerts + incidents +
    scores). Miss a call here and the UI shows stale case status. The next
    request recomputes lazily.
    """
    state["_alerts_cache"] = None
    state["_dashboard_cache"] = None
    state["_view_cache"] = {}   # geo + analytics group-bys (transaction-derived)


def _tracer_caches() -> Dict:
    """The per-entity structures the fund tracer needs, built ONCE at startup.

    Passing these in turns the journey endpoints from ~14s (four full passes over
    100k+ transactions per request) into a fast lookup. Falls back to on-demand
    building inside the tracer for any cache that wasn't pre-built.
    """
    out = {}
    for state_key, arg in (("_txn_index", "txn_index"), ("_baseline_stats", "baseline_stats"),
                           ("_burst_counts", "burst_counts"), ("_transit_ratios", "transit_ratios")):
        v = state.get(state_key)
        if v is not None:
            out[arg] = v
    return out


def _ensemble_edge_scores() -> Dict:
    """Lazily load + cache the per-edge ensemble scores ({xgb,sage,gat,ensemble})."""
    if state.get("ensemble_edge_scores") is None:
        try:
            from ensemble_model import load_ensemble_edge_scores
            state["ensemble_edge_scores"] = load_ensemble_edge_scores(DATA_DIR, variant=state["variant"]) or {}
        except Exception:
            state["ensemble_edge_scores"] = {}
    return state["ensemble_edge_scores"]


def _risk_scores_with_taint() -> List[Dict]:
    """Entity risk list with each score floored by its persisted taint.

    A clean-looking account that sits near confirmed fraud can't drop below its
    taint; genuinely high scores are preserved. Adds `taint` + `effective_risk`.
    """
    base = state["risk_scores"] or []
    taint = state.get("taint")
    if not taint:
        return base
    taint_map = taint.get_all()
    out = []
    for e in base:
        eid = e.get("entity_id")
        t = taint_map.get(eid, {}).get("taint", 0.0)
        eff = max(float(e.get("risk_score", 0.0)), float(t))
        row = dict(e)
        row["taint"] = round(float(t), 4)
        row["effective_risk"] = round(eff, 4)
        out.append(row)
    return out


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "RUDRA API",
        "version": "3.0",
        "variant": ACTIVE_VARIANT,
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
def get_dashboard():
    # Cached: the Pandas group-bys over ~100k rows are multi-second. case_status_counts
    # is the only live field, so this is invalidated on dispose + retrain via
    # _invalidate_derived_caches().
    if state.get("_dashboard_cache") is None:
        state["_dashboard_cache"] = _compute_dashboard()
    return state["_dashboard_cache"]


def _compute_dashboard():
    df = state["transactions"]
    alerts = state["alerts"]

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

    # New AML signals — pre-computed at startup, just look up the counts
    burst_counts = state.get("_burst_counts") or {}
    transit_ratios = state.get("_transit_ratios") or {}
    velocity_burst_entities = sum(1 for v in burst_counts.values() if v >= 1)
    transit_node_entities = sum(1 for v in transit_ratios.values() if v >= 0.5)

    return {
        "kpis": {
            "total_transactions": len(df),
            "total_volume": round(total_volume, 2),
            "fraud_transactions": len(fraud_txns),
            "fraud_volume": round(fraud_volume, 2),
            "fraud_rate": round(len(fraud_txns) / max(len(df), 1) * 100, 1),
            # Count the actual tiered alert set (what Cases/Incidents show), not
            # summary["total_alerts"] — that's the pre-fuse rule count (~701) and
            # under-reports the real ML+rule total (~8.7k).
            "total_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a.get("severity") == "CRITICAL"),
            "high_risk_entities": sum(1 for r in state["risk_scores"] if r["risk_score"] >= 0.5),
            "incidents": len(state["incidents"] or []),
            "model_f1": ml.get("f1"),
            "model_auc": ml.get("auc"),
            "velocity_burst_entities": velocity_burst_entities,
            "transit_node_entities": transit_node_entities,
        },
        "daily_data": daily.to_dict("records"),
        "pattern_breakdown": pattern_breakdown,
        "risk_distribution": risk_dist,
        "case_status_counts": case_status_counts,
        "amount_distribution": amount_distribution,
    }


# ── Graph ──────────────────────────────────────────────────────────────────

@app.get("/api/graph")
def get_graph(
    fraud_only: bool = False,
    min_amount: float = 0,
    high_risk_only: bool = False,
    limit: int = 200,
):
    """Return a meaningful slice of the fund-flow graph.

    For IBM AML (70k+ nodes / 87k+ edges) returning the whole graph is
    useless — the browser turns it into a hairball. The default behavior
    picks the top `limit` nodes by risk score, pulls in their 1-hop
    neighborhood, and caps the result. `limit=0` disables the cap.
    """
    graph = state["graph"]
    risk_map = {r["entity_id"]: r["risk_score"] for r in state["risk_scores"]}
    edge_scores = state["edge_scores"] or {}

    fraud_edges = {(u, v) for u, v, d in graph.edges(data=True) if d.get("fraud_count", 0) > 0}
    fraud_nodes = set()
    for u, v in fraud_edges:
        fraud_nodes.add(u); fraud_nodes.add(v)

    if fraud_only:
        seed = set(fraud_nodes)
    elif high_risk_only:
        seed = {r["entity_id"] for r in state["risk_scores"] if r["risk_score"] >= 0.3}
    else:
        # Default: rank by risk, take top `limit` seeds — keeps the response
        # bounded on large graphs like IBM AML.
        sorted_by_risk = sorted(
            state["risk_scores"], key=lambda r: r["risk_score"], reverse=True,
        )
        if limit and limit > 0:
            seed = {r["entity_id"] for r in sorted_by_risk[:limit]}
        else:
            seed = set(graph.nodes())

    # 1-hop expansion: pull in immediate predecessors/successors so the
    # selected nodes aren't dangling.
    expanded = set(seed)
    for n in seed:
        if n in graph:
            expanded.update(graph.predecessors(n))
            expanded.update(graph.successors(n))

    # Hard cap: still trim to `limit` total after expansion, keeping the
    # highest-risk nodes. limit=0 means "no cap" (use with care).
    if limit and limit > 0 and len(expanded) > limit:
        ranked = sorted(expanded, key=lambda n: risk_map.get(n, 0), reverse=True)
        expanded = set(ranked[:limit])

    sub = graph.subgraph(expanded).copy()
    edges_to_remove = [(u, v) for u, v, d in sub.edges(data=True) if d["total_amount"] < min_amount]
    sub.remove_edges_from(edges_to_remove)
    sub.remove_nodes_from(list(nx.isolates(sub)))

    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()

    if sub.number_of_nodes() == 0:
        return {
            "nodes": [], "links": [],
            "total_nodes": total_nodes, "total_edges": total_edges,
            "showing_nodes": 0, "showing_edges": 0,
            "applied_limit": limit, "default_filter": not (fraud_only or high_risk_only),
        }

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

    return {
        "nodes": nodes, "links": links,
        "total_nodes": total_nodes, "total_edges": total_edges,
        "showing_nodes": len(nodes), "showing_edges": len(links),
        "applied_limit": limit, "default_filter": not (fraud_only or high_risk_only),
    }


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


@app.get("/api/incidents/{incident_id}/rca")
def get_incident_rca(incident_id: str):
    inc = next((i for i in (state["incidents"] or []) if i.get("incident_id") == incident_id), None)
    if not inc:
        raise HTTPException(404, "Incident not found")
    alerts = _alerts_with_case_status()
    pid = inc.get("primary_alert_id")
    primary = next((a for a in alerts if a.get("alert_id") == pid), None)
    if primary is None:
        ids = set(inc.get("alert_ids", []))
        primary = next((a for a in alerts if a.get("alert_id") in ids), None)
    if primary is None:
        raise HTTPException(422, "Incident has no resolvable alert")
    return build_rca(
        inc, primary, state["graph"], state["transactions"], state["risk_scores"],
        edge_ml_scores=state["edge_scores"], **_tracer_caches(),
    )


# ── Patterns ───────────────────────────────────────────────────────────────

@app.get("/api/patterns/{pattern_type}")
def get_pattern(pattern_type: str):
    """Transactions + alerts for one detector pattern.

    Derived from the DETECTED ALERTS' entities — NOT from a per-pattern
    `fraud_pattern` tag. That tag only exists in the synthetic generator; real
    datasets (IBM AML) carry a single is_fraud label, so the old filter always
    returned nothing and every tab showed "No transaction data". Pulling the
    flows among each pattern's flagged entities works on any dataset.
    """
    df = state["transactions"]
    # substring matched against the alert's pattern_type label
    needle = {
        "circular": "circular", "layering": "layering", "smurfing": "smurfing",
        "funnel": "funnel", "dormant": "dormant", "profile": "profile",
        "recruiter": "recruiter",
    }.get(pattern_type, pattern_type).lower()

    matched = [a for a in _alerts_with_case_status()
               if needle in a.get("pattern_type", "").lower()]
    entities = set()
    for a in matched:
        entities.update(a.get("entities", []))

    if entities and {"sender_id", "receiver_id"}.issubset(df.columns):
        txns = df[df["sender_id"].isin(entities) & df["receiver_id"].isin(entities)]
        if txns.empty:
            # single-entity patterns (dormant / profile): show the entity's own activity
            txns = df[df["sender_id"].isin(entities) | df["receiver_id"].isin(entities)]
    else:
        txns = df.iloc[0:0]

    cols = [c for c in ["transaction_id", "timestamp", "sender_name", "receiver_name",
                        "amount", "transaction_type", "channel", "sender_type",
                        "receiver_type", "sender_branch", "is_fraud"]
            if c in txns.columns]
    sorted_txns = txns.sort_values("timestamp") if "timestamp" in txns.columns else txns
    if cols:
        out = sorted_txns[cols].head(300).copy()
        if "timestamp" in out.columns:
            out["timestamp"] = out["timestamp"].astype(str)
        txn_list = out.to_dict("records")
    else:
        txn_list = []
    return {
        "pattern": pattern_type,
        "alerts": matched,
        "transactions": txn_list,
        "total_volume": round(float(txns["amount"].sum()), 2) if len(txns) and "amount" in txns.columns else 0,
        "total_transactions": int(len(txns)),
    }


# ── Entities ───────────────────────────────────────────────────────────────

@app.get("/api/entities")
def get_entities(
    search: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = 500,
):
    """List entities, ranked by risk score (highest first).

    `limit` caps the response — on real datasets (IBM AML has 70k+ accounts)
    returning everything makes the frontend dropdown unusable. limit=0 means
    no cap.
    """
    entities = _risk_scores_with_taint()
    if search:
        entities = [e for e in entities if search.lower() in e["name"].lower()]
    if risk_level:
        entities = [e for e in entities if e["risk_level"] == risk_level]
    total = len(entities)
    # Rank by taint-floored effective risk so confirmed-adjacent accounts surface.
    entities_sorted = sorted(entities, key=lambda e: e.get("effective_risk", e["risk_score"]), reverse=True)
    if limit and limit > 0:
        entities_sorted = entities_sorted[:limit]
    return {"entities": entities_sorted, "total": total, "returned": len(entities_sorted)}


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

    # Compute the tracer flags so the UI can show them.
    from fund_tracer import (
        _annotate_node_flags, _build_txn_index,
        _build_baseline_stats, _build_burst_counts, _build_transit_ratios,
    )
    scc_set = state.get("ffg").get_sccs(min_size=3) if hasattr(state.get("ffg"), "get_sccs") else set()
    risk_map = {r["entity_id"]: r["risk_score"] for r in state["risk_scores"]}
    flags = _annotate_node_flags(
        graph, entity_id,
        state.get("_txn_index") or _build_txn_index(df),
        risk_map,
        in_scc=entity_id in scc_set,
        baseline_stats=state.get("_baseline_stats") or _build_baseline_stats(df),
        burst_counts=state.get("_burst_counts") or _build_burst_counts(df),
        transit_ratios=state.get("_transit_ratios") or _build_transit_ratios(df),
    )

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
        "flags": flags,
    }


# ── Copilot ────────────────────────────────────────────────────────────────

@app.post("/api/copilot/query")
async def copilot_query(body: dict):
    query = body.get("query", "")
    if not query:
        raise HTTPException(400, "Query is required")
    result = state["copilot"].query(query)
    # Surface mode + mode_label so the UI can render the right banner
    # (AI Copilot vs Quick Commands fallback).
    return {
        "response": result["response"],
        "source": result.get("source", "local"),
        "tool_calls": result.get("tool_calls", []),
        "mode": result.get("mode", "quick_commands"),
        "mode_label": result.get("mode_label", "Quick Commands (no LLM)"),
        "fallback_reason": result.get("fallback_reason"),
    }


@app.post("/api/copilot/stream")
async def copilot_stream(body: dict):
    """Streaming copilot endpoint — returns SSE chunks so the UI renders tokens as they arrive."""
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query required")
    return StreamingResponse(
        state["copilot"].stream_query(query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── SAR Reports ────────────────────────────────────────────────────────────

@app.post("/api/sar/generate/{alert_id}")
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
def get_ml_metrics(variant: str = None):
    if variant is None:
        variant = ACTIVE_VARIANT
    m = ml_load_metrics(DATA_DIR, variant=variant)
    if not m:
        return {"trained": False, "variant": variant,
                "message": f"Model variant '{variant}' not trained."}
    gnn_m = load_gnn_metrics(DATA_DIR, variant=variant)
    tgn_m = load_tgn_metrics(DATA_DIR, variant=variant)
    return {"trained": True, **m, "gnn": gnn_m or None, "tgn": tgn_m or None}


@app.get("/api/ml/ensemble")
def get_ensemble_metrics(variant: str = "ibm_aml"):
    """Stacked-ensemble metrics — XGBoost + GraphSAGE + GAT, LR meta-learner.

    Trained via 3-fold OOF stacking on real labelled data. Returns per-base
    model AUC/F1 and the ensemble's lift over the strongest base model.
    """
    from ensemble_model import load_ensemble_metrics
    m = load_ensemble_metrics(DATA_DIR, variant=variant)
    if not m:
        return {
            "trained": False, "variant": variant,
            "message": (
                f"Ensemble not trained for variant '{variant}'. "
                f"Run: python src/train_ibm_aml.py"
            ),
        }
    return {"trained": True, **m}


@app.get("/api/ml/ensemble/edge_scores")
def get_ensemble_edge_scores(variant: str = "ibm_aml", limit: int = 100):
    """Per-edge breakdown: xgb / sage / gat / ensemble scores.

    The UI uses this on the model-comparison page to show where the three
    base models disagree and how the meta-learner resolves it.
    """
    from ensemble_model import load_ensemble_edge_scores
    scores = load_ensemble_edge_scores(DATA_DIR, variant=variant)
    if not scores:
        return {"trained": False, "variant": variant}
    items = [{"edge": k, **v} for k, v in list(scores.items())[:limit]]
    return {"trained": True, "variant": variant, "edges": items, "total": len(scores)}


@app.get("/api/tgn/predictions")
def get_tgn_predictions(variant: str = None):
    """Return TGN predicted-fraud list (JSON only — never loads .pt weights)."""
    v = variant or ACTIVE_VARIANT
    data = load_tgn_predictions(DATA_DIR, variant=v)
    preds = data.get("predictions", [])
    return {"trained": bool(preds), "variant": v, "predictions": preds}


@app.get("/api/ml/ensemble/edge/{u}/{v}")
def get_ensemble_edge(u: str, v: str):
    """Per-model scores for ONE edge: how XGBoost, GraphSAGE and GAT each vote
    and how the meta-learner resolves them. Powers the 'layers agree' panel."""
    scores = _ensemble_edge_scores()
    row = scores.get(f"{u}->{v}")
    if not row:
        return {"found": False, "edge": f"{u}->{v}"}
    models = {k: row[k] for k in ("xgb", "sage", "gat") if k in row}
    ens = row.get("ensemble")
    vals = list(models.values())
    spread = (max(vals) - min(vals)) if vals else 0.0
    return {
        "found": True,
        "edge": f"{u}->{v}",
        "models": models,
        "ensemble": ens,
        "agreement": round(1.0 - spread, 3),     # 1.0 = all models agree
        "spread": round(spread, 3),
    }


# ── Health (deployment / docker healthcheck) ────────────────────────────────

@app.get("/api/health")
def health():
    """Probe every subsystem so a load-balancer / compose healthcheck can gate
    readiness on the brain being loaded, not just the process being up."""
    checks = {}
    g = state.get("graph")
    checks["graph_loaded"] = bool(g is not None and g.number_of_nodes() > 0)
    checks["alerts_loaded"] = bool(state.get("alerts"))
    checks["ml_bundle"] = bool(state.get("ml_bundle"))
    try:
        state["config"].get_all()
        checks["db"] = True
    except Exception:
        checks["db"] = False
    ing = state.get("ingestor")
    try:
        checks["stream"] = bool(ing and ing.status().get("running"))
    except Exception:
        checks["stream"] = False
    core_ok = checks["graph_loaded"] and checks["alerts_loaded"]
    return {"status": "ok" if core_ok else "degraded", "variant": state.get("variant"), "checks": checks}


# ── Geo: inter-city fund flows + fraud hotspots ─────────────────────────────

@app.get("/api/geo/flows")
def geo_flows():
    """Aggregate the (real) branch network into inter-city flows + per-city
    fraud hotspots for the India map view. Cached (group-bys over 100k rows)."""
    vc = state.setdefault("_view_cache", {})
    if "geo" not in vc:
        vc["geo"] = city_flows(state["transactions"])
    return vc["geo"]


# ── Persistent taint memory ─────────────────────────────────────────────────

@app.get("/api/taint")
def list_taint(limit: int = 100):
    """Entities carrying persistent taint (from confirmed-fraud cases)."""
    taint = state.get("taint")
    if not taint:
        return {"entities": [], "total": 0}
    allt = taint.get_all()
    g = state["graph"]
    rows = []
    for eid, info in list(allt.items())[:limit]:
        name = g.nodes[eid].get("name", eid) if g is not None and g.has_node(eid) else eid
        rows.append({"entity_id": eid, "name": name, **info})
    return {"entities": rows, "total": len(allt)}


@app.post("/api/taint/seed/{alert_id}")
def seed_taint(alert_id: str, role: str = Depends(get_role)):
    """Manually seed decaying taint from an alert's entities (investigator action).
    Auto-seeding also happens when a case is escalated / SAR-filed."""
    require("case.note", role)
    taint = state.get("taint")
    if not taint:
        raise HTTPException(503, "Taint store unavailable")
    alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
    if not alert or not alert.get("entities"):
        raise HTTPException(404, "Alert not found or has no entities")
    computed = taint.seed(state["graph"], alert["entities"], source=f"manual:{alert_id}")
    return {"seeded": len(computed), "alert_id": alert_id, "entities": alert["entities"]}


# ── Simulation Studio: score-your-own-transaction + scripted scenarios ──────

_SIM_SCENARIOS = {
    "layering_chain": {
        "label": "Rapid layering chain",
        "description": "₹50L relayed through 3 fresh intermediaries within minutes.",
        "txns": [
            {"sender": "SIM_SRC", "receiver": "SIM_M1", "amount": 5_000_000, "rail": "RTGS"},
            {"sender": "SIM_M1", "receiver": "SIM_M2", "amount": 4_800_000, "rail": "RTGS"},
            {"sender": "SIM_M2", "receiver": "SIM_DST", "amount": 4_600_000, "rail": "NEFT"},
        ],
    },
    "smurfing_fanout": {
        "label": "Smurfing fan-out",
        "description": "One sender sprays sub-threshold ₹1.9L transfers to 8 mules.",
        "txns": [
            {"sender": "SIM_SMURF", "receiver": f"SIM_MULE{i}", "amount": 190_000, "rail": "IMPS"}
            for i in range(8)
        ],
    },
    "round_trip": {
        "label": "Round-trip cycle",
        "description": "Funds cycle A→B→C→A — classic round-tripping.",
        "txns": [
            {"sender": "SIM_A", "receiver": "SIM_B", "amount": 1_000_000, "rail": "NEFT"},
            {"sender": "SIM_B", "receiver": "SIM_C", "amount": 980_000, "rail": "NEFT"},
            {"sender": "SIM_C", "receiver": "SIM_A", "amount": 960_000, "rail": "NEFT"},
        ],
    },
    "recruiter_fleet": {
        "label": "Recruiter / coordinator",
        "description": "A coordinator funds 6 accounts that each forward onward.",
        "txns": (
            [{"sender": "SIM_COORD", "receiver": f"SIM_R{i}", "amount": 1_000_000, "rail": "IMPS"} for i in range(6)]
            + [{"sender": f"SIM_R{i}", "receiver": f"SIM_SINK{i}", "amount": 950_000, "rail": "NEFT"} for i in range(6)]
        ),
    },
}


async def _score_and_publish(sender, receiver, amount, channel, rail, currency, ts, publish=True) -> Dict:
    """Score one transaction live (ml + severity + signals + ensemble + SHAP) and,
    best-effort, publish it onto the stream so the live feed reacts."""
    from streaming.ingestor import _severity_from_score, _signals_from_features
    bundle = state.get("ml_bundle")
    graph = state["graph"]
    out = {"sender": sender, "receiver": receiver, "amount": amount, "channel": channel, "rail": rail}
    if bundle is None:
        out["error"] = "ML bundle not loaded"
        return out

    res = score_live_txn(bundle, graph, sender, receiver, float(amount), channel, rail, ts, currency)
    ml_score = res["ml_score"]
    features = res["features"]
    threshold = bundle.get("threshold")
    out["ml_score"] = round(float(ml_score), 4)
    out["threshold"] = threshold
    out["severity"] = _severity_from_score(ml_score, float(amount), currency, threshold)
    out["signals"] = _signals_from_features(features)
    out["latency_ms"] = res["latency_ms"]
    out["edge_exists"] = bool(graph.has_edge(sender, receiver))

    edge_row = _ensemble_edge_scores().get(f"{sender}->{receiver}")
    if edge_row:
        out["ensemble"] = {k: edge_row[k] for k in ("xgb", "sage", "gat", "ensemble") if k in edge_row}
    if graph.has_edge(sender, receiver):
        try:
            expl = explain_edge(bundle, graph, state["transactions"], sender, receiver)
            if expl:
                out["shap"] = expl.get("top_features")
        except Exception:
            pass

    if publish and state.get("ingestor"):
        try:
            import uuid
            txn = StreamTxn(
                transaction_id=f"SIM_{uuid.uuid4().hex[:8]}",
                sender_id=sender, receiver_id=receiver, amount=float(amount),
                timestamp=ts.isoformat(), channel=channel, transaction_type=rail, currency=currency,
            )
            await state["ingestor"].publish(txn)
            out["published"] = True
        except Exception:
            out["published"] = False
    return out


def _sample_flagged_edge():
    """A real, high-ML-score edge from the dataset, for prefilling the Studio.

    Scoring fictitious accounts yields a contextless number with no ensemble/SHAP
    (those need a real graph edge). Defaulting the form to an actual flagged edge
    makes the first score meaningful — high probability, ensemble votes, and SHAP
    all populate. We pick the highest XGBoost-scored edge that exists in the graph.
    """
    graph = state["graph"]

    def _max_scored_edge(scores):
        b, bs = None, -1.0
        for k, s in scores.items():
            try:
                sv = float(s)
            except (TypeError, ValueError):
                continue
            if "->" not in k or sv <= bs:
                continue
            uu, vv = k.split("->", 1)
            if graph.has_edge(uu, vv):
                b, bs = (uu, vv), sv
        return b

    # 1. Highest XGBoost-scored edge in the graph.
    best = _max_scored_edge(state.get("edge_scores") or {})
    # 2. Fall back to ensemble scores if the XGB edge_scores failed to load.
    if best is None:
        ens = _ensemble_edge_scores() or {}
        best = _max_scored_edge({k: v.get("ensemble", 0) for k, v in ens.items()
                                 if isinstance(v, dict)})
    # 3. Last resort: any edge that carries a flagged transaction.
    if best is None:
        for uu, vv, d in graph.edges(data=True):
            if d.get("fraud_count", 0) > 0:
                best = (uu, vv)
                break
    if best is None:
        return None
    u, v = best
    ed = graph[u][v]
    rail_mix = ed.get("rail_mix") or {}
    ch_mix = ed.get("channel_mix") or {}
    rail = max(rail_mix, key=rail_mix.get) if rail_mix else "NEFT"
    channel = max(ch_mix, key=ch_mix.get) if ch_mix else "NetBanking"
    return {
        "sender": u, "receiver": v,
        "sender_name": graph.nodes[u].get("name", u),
        "receiver_name": graph.nodes[v].get("name", v),
        "amount": round(float(ed.get("total_amount", 0.0)), 2),
        "channel": channel, "rail": rail,
        "xgb_score": round(float((state.get("edge_scores") or {}).get(f"{u}->{v}", 0.0)), 4),
    }


@app.get("/api/simulate/scenarios")
def simulate_scenarios():
    """List the one-click fraud scenarios + a real flagged edge to prefill the form."""
    return {
        "scenarios": [
            {"name": k, "label": v["label"], "description": v["description"], "n_txns": len(v["txns"])}
            for k, v in _SIM_SCENARIOS.items()
        ],
        "sample_edge": _sample_flagged_edge(),
    }


@app.post("/api/simulate/score")
async def simulate_score(body: dict):
    """Score a user-built transaction and surface the full breakdown (ml score,
    severity, honest signals, ensemble votes, SHAP) — and push it to the live feed."""
    sender = str(body.get("sender") or "SIM_SENDER")
    receiver = str(body.get("receiver") or "SIM_RECEIVER")
    # Coerce hostile input gracefully — a non-numeric amount or bad timestamp
    # must yield a clean 400, never a 500.
    try:
        amount = float(body.get("amount") or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "amount must be a number")
    if amount < 0:
        raise HTTPException(400, "amount must be non-negative")
    channel = str(body.get("channel") or "NetBanking")
    rail = str(body.get("rail") or body.get("transaction_type") or "NEFT")
    currency = str(body.get("currency") or "INR")
    try:
        ts = pd.Timestamp(body["timestamp"]) if body.get("timestamp") else pd.Timestamp.now()
    except (ValueError, TypeError):
        raise HTTPException(400, "timestamp is not a valid date/time")
    return await _score_and_publish(sender, receiver, amount, channel, rail, currency, ts, publish=True)


@app.post("/api/simulate/scenario/{name}")
async def simulate_scenario(name: str):
    """Inject a scripted fraud scenario; each txn is scored live and streamed."""
    sc = _SIM_SCENARIOS.get(name)
    if not sc:
        raise HTTPException(404, f"Unknown scenario. Options: {list(_SIM_SCENARIOS)}")
    results = []
    for t in sc["txns"]:
        results.append(await _score_and_publish(
            t["sender"], t["receiver"], float(t["amount"]),
            t.get("channel", "NetBanking"), t.get("rail", "NEFT"), "INR",
            pd.Timestamp.now(), publish=True,
        ))
    return {"scenario": name, "label": sc["label"], "description": sc["description"],
            "injected": len(results), "results": results}


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
                          variant=ACTIVE_VARIANT, dataset_name=ACTIVE_VARIANT)
    state["ml_metrics"] = ml_load_metrics(DATA_DIR, variant=ACTIVE_VARIANT)
    state["edge_scores"] = ml_load_edge_scores(DATA_DIR, variant=ACTIVE_VARIANT)
    state["ml_bundle"] = ml_load_model(DATA_DIR, variant=ACTIVE_VARIANT)
    # New edge scores + metrics → invalidate memoised alert + dashboard views.
    _invalidate_derived_caches()
    return {"status": "ok", "variant": ACTIVE_VARIANT, "metrics": m}


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
        **_tracer_caches(),
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
        **_tracer_caches(),
    )


# ── Case Workbench ─────────────────────────────────────────────────────────

@app.get("/api/cases")
def list_cases(status: Optional[str] = None):
    cases = state["cases"].list(status=status)
    return {"cases": cases, "total": len(cases)}


@app.get("/api/cases/{alert_id}")
def get_case(alert_id: str):
    case = state["cases"].get(alert_id)
    if case:
        return case
    alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
    if not alert:
        raise HTTPException(404, "Case / alert not found")
    # Read-only: a GET must NOT persist a case (that would spawn junk cases +
    # audit-log entries on a mere refresh). Return a transient "unopened" view
    # in the same shape as a real case (status OPEN, as status_counts treats
    # caseless alerts). The case is created lazily by the first write action
    # (dispose / note), which already call open_case().
    return {
        "alert_id": alert_id,
        "pattern_type": alert.get("pattern_type", ""),
        "severity": alert.get("severity", ""),
        "total_flow": alert.get("total_flow", 0),
        "entities": alert.get("entities", []),
        "status": "OPEN",
        "assigned_to": None,
        "created_at": None,
        "updated_at": None,
        "incident_id": None,
        "audit_log": [],
    }


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
    # Confirming fraud seeds persistent, decaying taint from the alert's entities
    # so the suspicion survives future pipeline runs and floors neighbours' risk.
    if status in ("ESCALATED", "SAR_FILED") and state.get("taint"):
        alert = next((a for a in state["alerts"] if a.get("alert_id") == alert_id), None)
        if alert and alert.get("entities"):
            state["taint"].seed(state["graph"], alert["entities"], source=f"{status}:{alert_id}")
    # Case status / assignment changed → drop memoised alert + dashboard views.
    _invalidate_derived_caches()
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
    result = state["cases"].add_note(alert_id, note=note, author=body.get("author", role.lower()))
    # Defensive: a note (or its open_case fallback) touches the case store. The
    # cached alert/dashboard views don't depend on note text today, but drop them
    # anyway so this can't silently go stale if note semantics change later.
    _invalidate_derived_caches()
    return result


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
    # SARs are generated on-request (not pre-baked). export_sar_pdf makedirs the
    # target itself, so no directory needs to pre-exist.
    sar_dir = os.path.join(VARIANT_DIR, "sar_reports")
    sar = state["sar_gen"].generate_sar(alert)
    sar_pdf_path = state["sar_gen"].export_sar_pdf(sar, sar_dir)
    case = state["cases"].get(alert_id)
    # Embed the alert's SHAP attributions in the STR XML when the ML bundle is
    # available, so the model's reasoning travels with the regulatory filing.
    shap_features = None
    if state.get("ml_bundle"):
        try:
            expl = shap_explain_alert(state["ml_bundle"], state["graph"], state["transactions"], alert)
            if expl:
                shap_features = expl.get("top_features")
        except Exception:
            shap_features = None
    zip_bytes = build_fiu_package(
        state["graph"], state["transactions"], alert, sar_pdf_path, case=case,
        shap_features=shap_features,
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
    # Load the trained risk-weights LR bundle if present (T2.10). Falls back
    # cleanly to hand-tuned weights when the file isn't there.
    try:
        from risk_score_learner import load_risk_weights
        rw_bundle = load_risk_weights(DATA_DIR, variant=ACTIVE_VARIANT)
    except Exception:
        rw_bundle = None
    det = FraudDetector(
        state["graph"], transactions=state["transactions"],
        config=cfg, risk_weights_bundle=rw_bundle,
    )
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
    profile_alerts = ProfileMismatchDetector(
        state["graph"], state["transactions"], risk_scores_data,
        config=cfg, edge_scores=state.get("edge_scores"),
    ).detect()
    all_alerts = results["all_alerts"] + dormant_alerts + profile_alerts
    results["all_alerts"] = all_alerts
    results["dormant_activation"] = dormant_alerts
    results["profile_mismatch"] = profile_alerts
    results["summary"]["total_alerts"] = len(all_alerts)
    results["summary"]["critical_alerts"] = sum(1 for a in all_alerts if a["severity"] == "CRITICAL")
    results["summary"]["high_alerts"] = sum(1 for a in all_alerts if a["severity"] == "HIGH")
    results["summary"]["medium_alerts"] = sum(1 for a in all_alerts if a["severity"] == "MEDIUM")
    det.save_results(results, VARIANT_DIR)

    incidents = cluster_alerts(all_alerts, graph=state["graph"])
    with open(os.path.join(VARIANT_DIR, "incidents.json"), "w") as f:
        json.dump(incidents, f, indent=2, default=str)
    state["alerts"] = all_alerts
    state["incidents"] = incidents
    state["risk_scores"] = risk_scores_data
    # Alerts / incidents / scores all replaced → invalidate memoised views.
    _invalidate_derived_caches()
    return {
        "status": "ok",
        "alert_count": len(all_alerts),
        "incident_count": len(incidents),
        "summary": results["summary"],
    }


# ── Analytics ──────────────────────────────────────────────────────────────

@app.get("/api/analytics/channels")
def analytics_channels():
    vc = state.setdefault("_view_cache", {})
    if "channels" in vc:
        return vc["channels"]
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
    vc["channels"] = {"by_channel": by_channel, "by_rail": by_rail, "by_hour": by_hour}
    return vc["channels"]


@app.get("/api/analytics/branches")
def analytics_branches():
    vc = state.setdefault("_view_cache", {})
    if "branches" in vc:
        return vc["branches"]
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
    vc["branches"] = {"branches": merged.to_dict("records")}
    return vc["branches"]


# ── Account Aggregator + KYC ───────────────────────────────────────────────
# Both flows go through real adapters: src.integrations.AAClient and
# DilisenseClient. They call real Sahamati / DiliSense endpoints when their
# env-driven creds are present, and transparently fall back to the
# schema-accurate mock when they aren't. Every response carries `_real`
# so the UI can show the operator which mode is active.

@app.post("/api/aa/consent")
def aa_consent(body: dict):
    return state["aa_client"].create_consent(
        customer_id=body.get("customer_id", "CUST-000"),
        fip_ids=body.get("fip_ids", ["FIP-HDFC", "FIP-AXIS"]),
        purpose_code=body.get("purpose_code", "103"),
        duration_days=body.get("duration_days", 30),
    )


@app.get("/api/aa/consents")
def aa_consents():
    return {"consents": state["aa_client"].list_consents()}


@app.get("/api/aa/pull/{consent_handle}")
def aa_pull(consent_handle: str, days_back: int = 30):
    return state["aa_client"].pull_data(consent_handle, days_back=days_back)


@app.post("/api/aa/revoke/{consent_handle}")
def aa_revoke(consent_handle: str):
    return state["aa_client"].revoke(consent_handle)


@app.get("/api/kyc/screen")
def kyc_screen(name: str, entity_type: str = "individual"):
    return state["dilisense_client"].screen(name, entity_type)


@app.get("/api/integrations/status")
def integrations_status():
    """Which external APIs are live (real creds present) vs. mocked.

    The UI surfaces this so an operator can see at a glance whether they
    are hitting the real Sahamati sandbox / DiliSense API or the local mock.
    """
    return {
        "aa": state["aa_client"].mode(),
        "kyc": state["dilisense_client"].mode(),
    }


# ── Streaming (real Kafka, in-process fallback) ────────────────────────────

@app.get("/api/stream/status")
def stream_status():
    ing = state.get("ingestor")
    if ing is None:
        return {"running": False, "mode": "stopped"}
    return ing.status()


@app.post("/api/stream/start")
async def stream_start(role: str = Depends(get_role)):
    require("stream.control", role)
    ing = state.get("ingestor")
    if ing is None:
        raise HTTPException(503, "Ingestor not initialised")
    return await ing.start()


@app.post("/api/stream/stop")
async def stream_stop(role: str = Depends(get_role)):
    require("stream.control", role)
    ing = state.get("ingestor")
    if ing is None:
        raise HTTPException(503, "Ingestor not initialised")
    return await ing.stop()


@app.post("/api/stream/reset")
async def stream_reset(role: str = Depends(get_role)):
    """Hard reset — stop the consumer and wipe the ring buffer + counters so the
    feed restarts from empty at seq 0. Used by the Live page's Reset button."""
    require("stream.control", role)
    ing = state.get("ingestor")
    if ing is None:
        raise HTTPException(503, "Ingestor not initialised")
    return await ing.reset()


@app.get("/api/stream/recent")
def stream_recent(limit: int = 50):
    """Newest-first window over the in-memory ring buffer of scored stream events."""
    ing = state.get("ingestor")
    if ing is None:
        return {"events": [], "count": 0}
    events = ing.recent(limit=limit)
    return {"events": events, "count": len(events)}


@app.post("/api/stream/replay")
async def stream_replay(body: dict, role: str = Depends(get_role)):
    """Replay a slice of the loaded transactions onto the stream bus.

    Body: {"rate": 5, "total": 100, "shuffle": true}
    """
    require("stream.control", role)
    ing = state.get("ingestor")
    if ing is None or not ing.status().get("running"):
        raise HTTPException(503, "Stream ingestor is not running")
    rate = float(body.get("rate", 5.0))
    total = int(body.get("total", 100))
    shuffle = bool(body.get("shuffle", True))
    count = await replay_transactions(
        ing, state["transactions"], rate=rate, total=total, shuffle=shuffle,
    )
    return {"published": count, "rate": rate, "total": total}


@app.get("/api/stream/velocity_alerts")
def stream_velocity_alerts(limit: int = 50):
    """Read the Pathway velocity-alert log (if Pathway is running).

    Pathway is an optional companion process. When it's running it tails the
    same Kafka topic this backend subscribes to, computes 5-min sliding
    windows of per-entity volume, and writes alerts to a JSONL file we
    surface here. When Pathway isn't running, the list is empty.
    """
    try:
        from streaming.pathway_engine import read_recent_alerts
        return {"alerts": read_recent_alerts(limit=limit)}
    except Exception as e:
        return {"alerts": [], "error": str(e)}


# ── Pipeline trigger ───────────────────────────────────────────────────────

@app.post("/api/pipeline/run")
def trigger_pipeline(role: str = Depends(get_role)):
    require("pipeline.run", role)
    run_pipeline()
    state["loaded"] = False
    load_or_generate()
    return {"status": "ok", "message": "Pipeline completed"}
