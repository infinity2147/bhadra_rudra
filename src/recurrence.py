"""Recurrence-based severity escalation — a temporal triage axis.

Adapted from the IRB ("Intimation Rule Based") escalation in Ahmed et al.,
*A semantic rule based digital fraud detection* (PeerJ CS 2021): an entity
re-flagged across multiple time windows is escalated, so a serial re-offender
stands out from a one-shot flag.

This is a triage/legibility layer, **not** a detection change: nothing is
suppressed (recall is untouched) and no alert is added or removed (precision is
untouched). It only labels alerts with an escalation level and re-orders them.

Three axes, kept orthogonal:
  * tier      — ML+rule confidence agreement (existing)
  * severity  — transaction amount (existing)
  * escalation— temporal recurrence (this module)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

LEVEL_LABEL = {1: "Suspected", 2: "Investigate", 3: "Recurring Pattern"}

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_ts(s) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:19], _TS_FMT)
    except (ValueError, TypeError):
        return None


def derive_alert_times(alerts: List[Dict], graph) -> Dict[str, Optional[datetime]]:
    """Map each alert to a representative time = max `last_seen` over the graph
    edges that run *between the alert's own entities* (so a smurfing hub's alert
    is timed by its own activity). Alerts with no such edge get None → L1.
    """
    out: Dict[str, Optional[datetime]] = {}
    for a in alerts:
        ents = set(a.get("entities", []))
        best = None
        for u in ents:
            if u not in graph:
                continue
            for v in graph.successors(u):
                if v in ents:
                    t = _parse_ts(graph[u][v].get("last_seen"))
                    if t and (best is None or t > best):
                        best = t
        out[a.get("alert_id")] = best
    return out


def compute_recurrence(
    alerts: List[Dict],
    alert_times: Dict[str, Optional[datetime]],
    window_hours: float = 24.0,
    l2_windows: int = 2,
    l3_windows: int = 3,
) -> Dict[str, Dict]:
    """Per-entity recurrence over distinct time windows.

    Returns {entity: {"hit_count": n_distinct_windows, "windows": [...],
    "level": 1|2|3}}. Alerts whose time is None are ignored (they cannot place a
    window); an entity seen only in such alerts simply won't appear here and is
    treated as L1 downstream.
    """
    present = [t for t in alert_times.values() if t is not None]
    if not present:
        return {}
    t0 = min(present)
    span = window_hours * 3600.0

    ent_windows: Dict[str, set] = defaultdict(set)
    for a in alerts:
        t = alert_times.get(a.get("alert_id"))
        if t is None:
            continue
        w = int((t - t0).total_seconds() // span)
        for e in a.get("entities", []):
            ent_windows[e].add(w)

    out: Dict[str, Dict] = {}
    for e, ws in ent_windows.items():
        n = len(ws)
        level = 3 if n >= l3_windows else 2 if n >= l2_windows else 1
        out[e] = {"hit_count": n, "windows": sorted(ws), "level": level}
    return out


def apply_escalation(alerts: List[Dict], recurrence_map: Dict[str, Dict]) -> List[Dict]:
    """Attach an `escalation` block to every alert, in place.

    An alert's level is the MAX recurrence level over its entities (so a hub's
    alerts inherit the hub's recurrence; a one-shot leaf stays L1). The driving
    entity and its window list ride along for the UI's recurrence trail. Additive
    only — every input alert is returned (recall guard).
    """
    for a in alerts:
        best_ent, best_info = None, None
        for e in a.get("entities", []):
            info = recurrence_map.get(e)
            if info and (best_info is None or info["level"] > best_info["level"]
                         or (info["level"] == best_info["level"]
                             and info["hit_count"] > best_info["hit_count"])):
                best_ent, best_info = e, info
        if best_info is None:
            a["escalation"] = {"level": 1, "label": LEVEL_LABEL[1],
                               "hit_count": 1, "entity": None, "windows": []}
        else:
            a["escalation"] = {
                "level": best_info["level"],
                "label": LEVEL_LABEL[best_info["level"]],
                "hit_count": best_info["hit_count"],
                "entity": best_ent,
                "windows": best_info["windows"],
            }
    return alerts
