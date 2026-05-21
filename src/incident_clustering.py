"""
Alert clustering into Incidents.

A real bank investigator looking at 300 raw fraud alerts is overwhelmed.
Almost half of them overlap — different detectors flag the same group of
entities, multiple cycles share members, layering chains chain into funnels.

We collapse the alert set into *incidents*: connected components of alerts
that share at least one entity. Each incident has:

  - one or more alerts
  - the union of all involved entities
  - a "primary pattern" (the most severe alert's pattern)
  - the worst severity in the cluster
  - the summed total flow (deduplicated where it would over-count)

This converts a 290-alert noise stream into ~30 actionable cases for
triage. The same investigator who would burn out reading every alert can
now work through real distinct incidents.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set


SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def cluster_alerts(alerts: List[Dict], graph=None) -> List[Dict]:
    """Group overlapping alerts into incidents.

    Two alerts are linked if they share at least one entity. We then take
    connected components — each component is one incident.
    """
    if not alerts:
        return []

    # Build an entity -> [alert_indices] map
    ent_to_alerts: Dict[str, List[int]] = defaultdict(list)
    for i, a in enumerate(alerts):
        for e in a.get("entities", []):
            ent_to_alerts[e].append(i)

    # Union-Find over alert indices
    n = len(alerts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for indices in ent_to_alerts.values():
        for j in range(1, len(indices)):
            union(indices[0], indices[j])

    # Collect components
    components: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    incidents = []
    for incident_idx, (root, members) in enumerate(sorted(components.items()), start=1):
        member_alerts = [alerts[i] for i in members]

        # Aggregate
        all_entities: Set[str] = set()
        for a in member_alerts:
            all_entities.update(a.get("entities", []))

        worst_severity = min((a.get("severity", "MEDIUM") for a in member_alerts),
                              key=lambda s: SEVERITY_ORDER.get(s, 99))
        # Primary pattern = pattern of the highest-severity alert with the largest flow
        sorted_alerts = sorted(
            member_alerts,
            key=lambda a: (SEVERITY_ORDER.get(a.get("severity", "MEDIUM"), 99),
                            -float(a.get("total_flow", 0))),
        )
        primary = sorted_alerts[0]

        # Total flow: take MAX of each alert's flow (not sum — sum would double-count overlapping txns)
        total_flow = max((float(a.get("total_flow", 0)) for a in member_alerts), default=0)

        # Pattern diversity
        patterns = sorted({a.get("pattern_type", "") for a in member_alerts if a.get("pattern_type")})

        entity_names = []
        if graph is not None:
            for eid in list(all_entities)[:20]:
                if graph.has_node(eid):
                    entity_names.append(graph.nodes[eid].get("name", eid))

        incident_id = f"INC-{incident_idx:04d}"
        incidents.append({
            "incident_id": incident_id,
            "alert_ids": [a.get("alert_id") for a in member_alerts],
            "alert_count": len(member_alerts),
            "primary_pattern": primary.get("pattern_type", ""),
            "patterns": patterns,
            "severity": worst_severity,
            "entities": sorted(all_entities),
            "entity_names": entity_names,
            "n_entities": len(all_entities),
            "total_flow": round(total_flow, 2),
            "primary_alert_id": primary.get("alert_id"),
            "primary_alert_description": primary.get("description", ""),
        })

    # Sort by severity then by alert count then by flow
    incidents.sort(
        key=lambda i: (SEVERITY_ORDER.get(i["severity"], 99),
                       -i["alert_count"], -i["total_flow"]),
    )
    return incidents


def alert_to_incident_map(incidents: List[Dict]) -> Dict[str, str]:
    """Return alert_id -> incident_id mapping for quick lookup."""
    out = {}
    for inc in incidents:
        for aid in inc["alert_ids"]:
            out[aid] = inc["incident_id"]
    return out
