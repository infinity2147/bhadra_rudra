"""rca_engine.py — assemble a Forensic RCA dossier for a clustered incident.

build_rca() is the public entrypoint; it orchestrates
reconstruct -> diagnose_root_cause -> recommend. All functions are pure
(no I/O) so they unit-test without booting the app.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fund_tracer import trace_for_alert
from fatf_typology import TYPOLOGY, _GENERIC

STRUCTURING_THRESHOLD = 200000  # ₹2L — mirrors fund_tracer.STRUCTURING_THRESHOLD


def _structuring_threshold(config: Optional[Dict]) -> float:
    if config:
        return float(config.get("structuring_threshold", STRUCTURING_THRESHOLD))
    return STRUCTURING_THRESHOLD


def reconstruct(primary_alert, graph, transactions, risk_scores,
                *, edge_ml_scores=None, config=None, **tracer_caches) -> Dict:
    trace = trace_for_alert(
        graph, transactions, risk_scores, alert=primary_alert,
        edge_ml_scores=edge_ml_scores, config=config, **tracer_caches,
    )
    if "error" in trace:
        return {"error": trace["error"]}

    nodes = trace.get("nodes", [])
    timeline = trace.get("timeline", [])
    red_flags = trace.get("summary", {}).get("red_flags", [])
    thr = _structuring_threshold(config)

    inflow: Dict[str, float] = {}
    outflow: Dict[str, float] = {}
    fan_in: Dict[str, set] = {}
    for t in timeline:
        s, r = t.get("sender_id"), t.get("receiver_id")
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
