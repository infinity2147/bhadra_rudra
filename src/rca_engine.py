"""rca_engine.py — assemble a Forensic RCA dossier for a clustered incident.

build_rca() is the public entrypoint; it orchestrates
reconstruct -> diagnose_root_cause -> recommend. All functions are pure
(no I/O) so they unit-test without booting the app.
"""
from __future__ import annotations

from typing import Dict, Optional

from fund_tracer import trace_for_alert
from fatf_typology import TYPOLOGY, _GENERIC

STRUCTURING_THRESHOLD = 200000  # ₹2L — mirrors fund_tracer.STRUCTURING_THRESHOLD


def _structuring_threshold(config: Optional[Dict]) -> float:
    if config is not None:
        return float(config.get("structuring_threshold", STRUCTURING_THRESHOLD))
    return STRUCTURING_THRESHOLD


def reconstruct(primary_alert, graph, transactions, risk_scores,
                *, edge_ml_scores=None, config=None, **tracer_caches) -> Dict:
    trace = trace_for_alert(
        graph, transactions, risk_scores, alert=primary_alert,
        edge_ml_scores=edge_ml_scores, config=config, **tracer_caches,
    )
    if "error" in trace:
        return {"error": str(trace["error"])}

    nodes = trace.get("nodes", [])
    timeline = trace.get("timeline", [])
    red_flags = trace.get("summary", {}).get("red_flags", [])
    thr = _structuring_threshold(config)

    inflow: Dict[str, float] = {}
    outflow: Dict[str, float] = {}
    fan_in: Dict[str, set] = {}
    for t in timeline:
        s = t.get("sender_id")
        r = t.get("receiver_id")
        if not s or not r:
            continue
        amt = float(t.get("amount", 0) or 0)
        outflow[s] = outflow.get(s, 0.0) + amt
        inflow[r] = inflow.get(r, 0.0) + amt
        fan_in.setdefault(r, set()).add(s)

    def _net(e):
        return inflow.get(e, 0.0) - outflow.get(e, 0.0)

    ents = [n["id"] for n in nodes]
    origin = sorted([e for e in ents if _net(e) < 0], key=_net)[:3]
    cashout = sorted([e for e in ents if _net(e) > 0], key=_net, reverse=True)[:3]

    signals = {
        "in_scc": any(str(f).startswith("Cycle") for f in red_flags),
        "shell_count": sum(1 for n in nodes if "shell_company" in n.get("flags", [])),
        "dormant_count": sum(1 for n in nodes if "dormant_then_active" in n.get("flags", [])),
        "subthreshold_deposits": sum(
            1 for t in timeline
            if 0.5 * thr <= float(t.get("amount", 0) or 0) < thr
        ),
        "max_fan_in": max((len(v) for v in fan_in.values()), default=0),
        "n_txns": len(timeline),
        "total_amount": round(sum(float(t.get("amount", 0) or 0) for t in timeline), 2),
    }

    entry = TYPOLOGY.get(primary_alert.get("pattern_type", ""), _GENERIC)
    method = {
        "pattern": primary_alert.get("pattern_type"),
        "fatf_code": entry["fatf_code"],
        "fatf_typology": entry["fatf_typology"],
    }
    return {
        "method": method,
        "origin": origin,
        "cashout": cashout,
        "signals": signals,
        "red_flags": red_flags,
        "trace": trace,
    }


def _infer_pattern(signals: Dict, thr_count: int = 3) -> str:
    if signals.get("in_scc"):
        return "Rapid Layering"
    if signals.get("shell_count", 0) > 0 and signals.get("max_fan_in", 0) >= 3:
        return "Shell Company Funnel"
    if signals.get("dormant_count", 0) > 0:
        return "Dormant Activation"
    if signals.get("subthreshold_deposits", 0) >= thr_count:
        return "Smurfing / Structuring"
    return ""


def _evidence_for(pattern: str, signals: Dict, thr: float) -> str:
    if pattern == "Smurfing / Structuring":
        return (f"{signals.get('subthreshold_deposits', 0)} transfers between "
                f"₹{0.5 * thr:,.0f} and ₹{thr:,.0f}, each under the ₹{thr:,.0f} "
                f"reporting threshold")
    if pattern == "Shell Company Funnel":
        return (f"{signals.get('shell_count', 0)} shell entity(ies) with fan-in up to "
                f"{signals.get('max_fan_in', 0)} counterparties")
    if pattern == "Dormant Activation":
        return f"{signals.get('dormant_count', 0)} dormant-then-active account(s) in the flow"
    if pattern in ("Rapid Layering", "Circular Transaction"):
        return (f"funds cycled through a closed loop spanning "
                f"{signals.get('n_txns', 0)} transactions")
    return (f"anomalous flow of ₹{signals.get('total_amount', 0):,.0f} across "
            f"{signals.get('n_txns', 0)} transactions inconsistent with entity profiles")


def diagnose_root_cause(incident, reconstruction, config=None) -> Dict:
    signals = reconstruction.get("signals", {})
    candidates = [incident.get("primary_pattern")] + list(incident.get("patterns", []))
    matched = next((p for p in candidates if p in TYPOLOGY), "")
    basis = "rule"
    if not matched:
        matched = _infer_pattern(signals)
        basis = "inferred" if matched else "generic"
    entry = TYPOLOGY.get(matched, _GENERIC)
    thr = _structuring_threshold(config)
    return {
        "pattern_resolved": matched or "Unclassified anomaly",
        "basis": basis,
        "control_gap": entry["control_gap"],
        "remediation": entry["remediation"],
        "evidence": _evidence_for(matched, signals, thr),
    }


def recommend(diagnosis, incident, *, max_accounts: int = 5) -> Dict:
    ents = list(incident.get("entities", []))[:max_accounts]
    names = list(incident.get("entity_names", []))
    account_level = [
        {
            "entity_id": e,
            "name": names[i] if i < len(names) else e,
            "action": "Enhanced Due Diligence (EDD) + transaction hold pending review",
        }
        for i, e in enumerate(ents)
    ]
    policy_level = [{
        "recommendation": diagnosis["remediation"],
        "closes_gap": diagnosis["control_gap"],
    }]
    return {"account_level": account_level, "policy_level": policy_level}


def rca_narrative(dossier: Dict) -> str:
    d = dossier["diagnosis"]
    r = dossier["reconstruction"]
    return (
        f"Method: {r['method']['fatf_typology']} ({r['method']['fatf_code']}). "
        f"Root cause: {d['control_gap']} Evidenced by {d['evidence']}. "
        f"Recommended fix: {d['remediation']}"
    )


def build_rca(incident, primary_alert, graph, transactions, risk_scores,
              *, edge_ml_scores=None, config=None, **tracer_caches) -> Dict:
    recon = reconstruct(primary_alert, graph, transactions, risk_scores,
                        edge_ml_scores=edge_ml_scores, config=config, **tracer_caches)
    if "error" in recon:
        return {"incident_id": incident.get("incident_id"), "error": recon["error"]}
    diag = diagnose_root_cause(incident, recon, config=config)
    recs = recommend(diag, incident)
    dossier = {
        "incident_id": incident.get("incident_id"),
        "reconstruction": recon,
        "diagnosis": diag,
        "recommendations": recs,
    }
    dossier["narrative"] = rca_narrative(dossier)
    return dossier
