"""Recurrence-based severity escalation (temporal triage axis).

Guarantees under test:
  * recurrence counts DISTINCT time windows per entity and maps to L1/L2/L3;
  * an alert inherits the MAX level over its entities;
  * escalation is additive only — no alert is dropped (recall guard);
  * alerts with no derivable time default to L1.
"""

import os
import sys
from datetime import datetime, timedelta

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recurrence import compute_recurrence, apply_escalation, derive_alert_times  # noqa: E402

BASE = datetime(2025, 1, 1, 0, 0, 0)


# ── compute_recurrence: distinct-window counting + levels ────────────────────

def test_same_window_counts_once_l1():
    alerts = [{"alert_id": "a1", "entities": ["X", "Y"]},
              {"alert_id": "a2", "entities": ["X", "Z"]}]
    times = {"a1": BASE, "a2": BASE + timedelta(hours=3)}  # both in one 24h window
    rec = compute_recurrence(alerts, times, window_hours=24, l2_windows=2, l3_windows=3)
    assert rec["X"]["hit_count"] == 1
    assert rec["X"]["level"] == 1


def test_two_windows_is_l2():
    alerts = [{"alert_id": "a1", "entities": ["X"]},
              {"alert_id": "a2", "entities": ["X"]}]
    times = {"a1": BASE, "a2": BASE + timedelta(days=1)}
    rec = compute_recurrence(alerts, times, window_hours=24, l2_windows=2, l3_windows=3)
    assert rec["X"]["hit_count"] == 2 and rec["X"]["level"] == 2


def test_three_windows_is_l3():
    alerts = [{"alert_id": f"a{i}", "entities": ["X"]} for i in range(3)]
    times = {"a0": BASE, "a1": BASE + timedelta(days=1), "a2": BASE + timedelta(days=2)}
    rec = compute_recurrence(alerts, times, window_hours=24, l2_windows=2, l3_windows=3)
    assert rec["X"]["hit_count"] == 3 and rec["X"]["level"] == 3
    assert rec["X"]["windows"] == [0, 1, 2]


def test_alert_with_no_time_is_ignored_in_counting():
    alerts = [{"alert_id": "a1", "entities": ["X"]}, {"alert_id": "a2", "entities": ["X"]}]
    times = {"a1": BASE, "a2": None}     # a2 has no derivable time
    rec = compute_recurrence(alerts, times, window_hours=24, l2_windows=2, l3_windows=3)
    assert rec["X"]["hit_count"] == 1 and rec["X"]["level"] == 1


# ── apply_escalation: max over entities, additive, default L1 ────────────────

def test_alert_inherits_max_entity_level():
    rec = {"hub": {"hit_count": 4, "windows": [0, 1, 2, 3], "level": 3},
           "leaf": {"hit_count": 1, "windows": [3], "level": 1}}
    alerts = [{"alert_id": "a", "entities": ["hub", "leaf"]}]
    apply_escalation(alerts, rec)
    e = alerts[0]["escalation"]
    assert e["level"] == 3
    assert e["label"] == "Recurring Pattern"
    assert e["entity"] == "hub"
    assert e["hit_count"] == 4
    assert e["windows"] == [0, 1, 2, 3]


def test_unknown_entity_defaults_to_l1():
    alerts = [{"alert_id": "a", "entities": ["nobody"]}]
    apply_escalation(alerts, {})
    assert alerts[0]["escalation"]["level"] == 1
    assert alerts[0]["escalation"]["label"] == "Suspected"


def test_escalation_never_drops_alerts():
    alerts = [{"alert_id": f"a{i}", "entities": ["X"]} for i in range(5)]
    out = apply_escalation(alerts, {})
    assert len(out) == 5  # recall guard: additive only


# ── derive_alert_times: max last_seen over entity edges ──────────────────────

def test_derive_time_uses_max_last_seen_among_entities():
    g = nx.DiGraph()
    g.add_edge("X", "Y", last_seen="2025-03-01 12:00:00")
    g.add_edge("X", "Z", last_seen="2025-03-05 09:30:00")
    alerts = [{"alert_id": "a", "entities": ["X", "Y"]}]
    times = derive_alert_times(alerts, g)
    assert times["a"] == datetime(2025, 3, 1, 12, 0, 0)  # only X->Y is among entities


def test_derive_time_none_when_no_edge_among_entities():
    g = nx.DiGraph()
    g.add_node("X")
    times = derive_alert_times([{"alert_id": "a", "entities": ["X"]}], g)
    assert times["a"] is None
