"""
ML-driven alert generation + rule/ML confidence tiering.

Why this module exists
----------------------
On real, diverse laundering data the heuristic rule engine has a low recall
ceiling — most fraud is structureless single transfers (A→B) that no graph
pattern can catch (measured: rule entity-recall ~17%). The ML edge classifier
catches ~67%. Historically the ML score only *coloured edges* and never
generated an alert. This module promotes the ML model to a first-class alert
generator so its recall actually produces caught cases, while the rule engine
is kept for what it is genuinely good at: precise, explainable typologies.

The two lanes are combined as a CONFIDENCE TIER, not an averaged score
(averaging a 67%-recall probability with a 3%-precision rule-only signal helps
nothing). Measured on IBM AML:

  * Tier 1 — ML and a rule agree on an entity  → precision ~74%  (priority queue)
  * Tier 2 — ML only                            → precision ~27%  (recall workhorse)
  * Tier 3 — rule only (ML disagrees)           → precision ~3%   (mostly noise)

So rule-only alerts are SUPPRESSED unless they come from a high-precision
typology detector (cycle / layering / recruiter), which stay as Tier 3.

The emitted ML alerts match the exact schema the rest of the pipeline expects
(incident clustering, case manager, FIU package), so they flow through the
existing spine with no downstream changes. The per-alert SHAP "reason" is NOT
precomputed here — it would mean SHAP over thousands of edges — it is fetched
lazily by the existing /api/alerts/{id}/explain endpoint when a case is opened.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx


# Rule-only alerts (ML disagrees) survive ONLY for these high-precision
# typologies — they are surgical and produce clean STR narratives. Everything
# else firing rule-only is the false-positive factory and is dropped.
KEEP_RULE_ONLY_TYPES = {
    "Circular Transaction",
    "Rapid Layering",
    "Recruiter / Coordinator",
}


def _severity_for_score(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    if score >= 0.80:
        return "HIGH"
    return "MEDIUM"


def generate_ml_alerts(
    graph: nx.DiGraph,
    ensemble_edge_scores: Dict[str, Dict],
    threshold: float,
    max_alerts: Optional[int] = None,
) -> List[Dict]:
    """Emit one alert per edge whose ensemble score clears `threshold`.

    Args:
        graph: the fund-flow graph (for amounts / names).
        ensemble_edge_scores: {"u->v": {"xgb":.., "sage":.., "gat":.., "ensemble":..}}.
            Falls back to a flat {"u->v": score} mapping (XGB-only) if that's all
            that is available.
        threshold: ensemble score at/above which an edge becomes an alert
            (the recall-favouring F2 operating point by default).
        max_alerts: optional cap (highest scores first) to bound volume.

    Returns: list of alert dicts in the canonical schema.
    """
    scored = []
    for key, row in ensemble_edge_scores.items():
        score = row.get("ensemble") if isinstance(row, dict) else row
        if score is None:
            continue
        if score >= threshold:
            scored.append((key, float(score)))
    scored.sort(key=lambda x: -x[1])
    if max_alerts is not None:
        scored = scored[:max_alerts]

    alerts: List[Dict] = []
    seq = 0
    for key, score in scored:
        if "->" not in key:
            continue
        u, v = key.split("->", 1)
        if not graph.has_edge(u, v):
            continue
        ed = graph[u][v]
        total = float(ed.get("total_amount", 0) or 0)
        cnt = int(ed.get("transaction_count", 0) or 0)
        un = graph.nodes[u].get("name", u) if u in graph.nodes else u
        vn = graph.nodes[v].get("name", v) if v in graph.nodes else v
        seq += 1
        alerts.append({
            "alert_id": f"ALERT_ML_{seq:05d}",
            "pattern_type": "ML-Detected Anomaly",
            "severity": _severity_for_score(score),
            "confidence": round(score * 100, 1),
            "ml_score": round(score, 4),
            "entities": [u, v],
            "entity_names": [un, vn],
            "total_flow": round(total, 2),
            "algorithm": "stacked_ensemble",
            "source": "ml",
            "description": (
                f"ML ensemble flagged the counterparty relationship "
                f"{un} → {vn}: fraud score {score:.2f} on ₹{total:,.0f} across "
                f"{cnt} transaction(s). Open the case for SHAP feature attribution."
            ),
            "recommendation": (
                "Review the SHAP drivers and the transaction pattern; corroborate "
                "against typology rules before disposition."
            ),
        })
    return alerts


def apply_tiers(rule_alerts: List[Dict], ml_alerts: List[Dict]) -> List[Dict]:
    """Combine rule + ML alerts into a single tiered list.

    Tier 1: ML alert sharing an entity with a rule alert (or vice-versa) — both
            lanes agree, highest precision. Carries `corroborated_by`.
    Tier 2: ML alert with no rule agreement — recall workhorse.
    Tier 3: rule-only alert from a high-precision typology (kept).
    Suppressed: rule-only alert from a low-precision typology (dropped).

    Mutates the alerts in place (adds `tier`, `corroborated_by`) and returns the
    surviving list.
    """
    # entity -> set of rule pattern_types that flagged it
    rule_ent_patterns: Dict[str, set] = {}
    for a in rule_alerts:
        for e in a.get("entities", []):
            rule_ent_patterns.setdefault(e, set()).add(a.get("pattern_type", "Rule"))

    ml_entities = set()
    for a in ml_alerts:
        ml_entities.update(a.get("entities", []))

    out: List[Dict] = []

    # ML alerts → Tier 1 if corroborated by any rule on a shared entity, else Tier 2
    for a in ml_alerts:
        corrob = set()
        for e in a.get("entities", []):
            corrob |= rule_ent_patterns.get(e, set())
        if corrob:
            a["tier"] = 1
            a["corroborated_by"] = sorted(corrob)
        else:
            a["tier"] = 2
            a["corroborated_by"] = []
        out.append(a)

    # Rule alerts → Tier 1 if ML agrees on a shared entity; else keep only
    # high-precision typologies as Tier 3, suppress the rest.
    for a in rule_alerts:
        ml_hit = any(e in ml_entities for e in a.get("entities", []))
        if ml_hit:
            a["tier"] = 1
            a["corroborated_by"] = ["ML-Detected Anomaly"]
            out.append(a)
        elif a.get("pattern_type") in KEEP_RULE_ONLY_TYPES:
            a["tier"] = 3
            a["corroborated_by"] = []
            out.append(a)
        # else: low-precision rule-only → suppressed (not appended)

    return out


def tier_summary(alerts: List[Dict]) -> Dict[str, int]:
    """Count alerts per tier (for pipeline logging / metrics)."""
    out = {"tier1": 0, "tier2": 0, "tier3": 0, "total": len(alerts)}
    for a in alerts:
        t = a.get("tier")
        if t in (1, 2, 3):
            out[f"tier{t}"] += 1
    return out


def _load_scores_and_threshold(data_dir: str, variant: str):
    """Load the best available per-edge ML scores + decision threshold.

    Prefers the stacked ensemble; falls back to the XGBoost edge scores. Returns
    (scores, threshold) or (None, None) if no ML artefacts exist yet.
    """
    import json
    import os

    ens_scores = os.path.join(data_dir, "ml", variant, "ensemble", "edge_scores.json")
    ens_metrics = os.path.join(data_dir, "ml", variant, "ensemble", "metrics.json")
    xgb_scores = os.path.join(data_dir, "ml", variant, "edge_scores.json")
    xgb_metrics = os.path.join(data_dir, "ml", variant, "metrics.json")

    if os.path.exists(ens_scores):
        scores = json.load(open(ens_scores))
        thr = None
        if os.path.exists(ens_metrics):
            thr = json.load(open(ens_metrics)).get("ensemble", {}).get("threshold")
        return scores, thr
    if os.path.exists(xgb_scores):
        scores = json.load(open(xgb_scores))  # flat {edge: float} — handled by generate_ml_alerts
        thr = None
        if os.path.exists(xgb_metrics):
            thr = json.load(open(xgb_metrics)).get("threshold")
        return scores, thr
    return None, None


def fuse_ml_alerts(
    data_dir: str,
    variant: str,
    graph: nx.DiGraph,
    rule_alerts: List[Dict],
    threshold_override: Optional[float] = None,
):
    """End-to-end fuse step for the pipeline.

    Loads ML edge scores, generates ML alerts at the (recall-favouring)
    threshold, tiers them against the rule alerts, and returns
    (combined_alerts, summary). If no ML scores exist yet, returns the rule
    alerts unchanged so the pipeline degrades gracefully.
    """
    scores, thr = _load_scores_and_threshold(data_dir, variant)
    if threshold_override is not None:
        thr = threshold_override
    if scores is None or thr is None:
        return list(rule_alerts), {"note": "no ML scores available — rule-only", **tier_summary(rule_alerts)}

    ml_alerts = generate_ml_alerts(graph, scores, thr)
    combined = apply_tiers(rule_alerts, ml_alerts)
    summary = tier_summary(combined)
    summary["ml_threshold"] = round(float(thr), 4)
    summary["ml_alerts_generated"] = len(ml_alerts)
    summary["rule_alerts_in"] = len(rule_alerts)
    summary["rule_alerts_kept"] = sum(1 for a in combined if a.get("source") != "ml")
    return combined, summary
