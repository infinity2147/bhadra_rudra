"""
Fund Journey Tracer — compute the end-to-end movement of funds around a given
entity, with timeline and red-flag annotations.

The output is built for the Journey page in the UI: a Sankey-ready node/link
list, a chronologically-sorted timeline of underlying transactions, and a
summary that names the specific suspicious patterns found in the trace.

This is the feature the problem statement is asking for:
> "enable investigators to trace the complete journey of funds"
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import pandas as pd
import networkx as nx


# Tunable thresholds used during annotation
NIGHT_HOURS = (22, 6)  # 10 PM – 6 AM inclusive
STRUCTURING_THRESHOLD = 200000  # ₹2L
HIGH_VALUE_THRESHOLD = 1000000  # ₹10L


def _node_meta(graph: nx.DiGraph, node_id: str, risk_map: Dict[str, float]) -> Dict:
    nd = dict(graph.nodes[node_id])
    return {
        "id": node_id,
        "name": nd.get("name", node_id),
        "type": nd.get("type", "individual"),
        "branch": nd.get("branch", ""),
        "product": nd.get("product", ""),
        "risk_score": round(float(risk_map.get(node_id, 0.0)), 3),
    }


def _bfs_in_direction(
    graph: nx.DiGraph,
    source: str,
    direction: str,
    max_hops: int,
    min_amount: float,
) -> Tuple[Dict[str, int], Set[Tuple[str, str]]]:
    """Walk the graph from `source` and return (node_depth, edges_visited).

    direction ∈ {"forward", "backward"}.
    Edges below `min_amount` are skipped — same convention as the rest of the app.
    """
    visited_depth = {source: 0}
    edges: Set[Tuple[str, str]] = set()
    frontier = [source]

    for depth in range(max_hops):
        next_frontier: List[str] = []
        for n in frontier:
            neighbors = (
                graph.successors(n) if direction == "forward" else graph.predecessors(n)
            )
            for nb in neighbors:
                edge = (n, nb) if direction == "forward" else (nb, n)
                u, v = edge
                if not graph.has_edge(u, v):
                    continue
                if graph[u][v]["total_amount"] < min_amount:
                    continue
                if edge in edges:
                    continue
                edges.add(edge)
                if nb not in visited_depth:
                    visited_depth[nb] = depth + 1
                    next_frontier.append(nb)
        if not next_frontier:
            break
        frontier = next_frontier

    return visited_depth, edges


def _annotate_node_flags(
    graph: nx.DiGraph,
    node_id: str,
    transactions: pd.DataFrame,
    risk_map: Dict[str, float],
    in_scc: bool,
) -> List[str]:
    flags: List[str] = []
    nd = graph.nodes[node_id]
    if nd.get("type") == "shell_company":
        flags.append("shell_company")
    if risk_map.get(node_id, 0) >= 0.5:
        flags.append("high_risk")
    if in_scc:
        flags.append("part_of_cycle")
    # Cross-branch activity check (only if we have node txns)
    node_txns = transactions[
        (transactions["sender_id"] == node_id) | (transactions["receiver_id"] == node_id)
    ]
    if len(node_txns) > 0:
        branches = set(node_txns["sender_branch"].tolist() + node_txns["receiver_branch"].tolist())
        branches.discard("")
        if len(branches) >= 5:
            flags.append("multi_branch_activity")
        # Dormant signature: gap > 30 days then activity
        ts = pd.to_datetime(node_txns["timestamp"]).sort_values()
        if len(ts) >= 2:
            diffs = ts.diff().dt.total_seconds() / 86400.0
            if diffs.max() >= 30:
                flags.append("dormant_then_active")
    return flags


def _annotate_edge_flags(graph: nx.DiGraph, u: str, v: str) -> List[str]:
    flags: List[str] = []
    ed = graph[u][v]
    avg = float(ed.get("avg_amount", 0))
    fraud_count = int(ed.get("fraud_count", 0))
    if fraud_count > 0:
        flags.append("contains_fraud_txn")
    # Below-threshold structuring hint
    if avg < STRUCTURING_THRESHOLD and avg > 0.7 * STRUCTURING_THRESHOLD:
        flags.append("near_reporting_threshold")
    # High-value rail share
    rail_mix = ed.get("rail_mix") or {}
    total = sum(rail_mix.values()) or 1
    high_share = (rail_mix.get("RTGS", 0) + rail_mix.get("Wire Transfer", 0)) / total
    if high_share > 0.5 and ed.get("total_amount", 0) > HIGH_VALUE_THRESHOLD:
        flags.append("high_value_rail")
    return flags


def trace_journey(
    graph: nx.DiGraph,
    transactions: pd.DataFrame,
    risk_scores: List[Dict],
    entity_id: str,
    direction: str = "both",
    max_hops: int = 3,
    min_amount: float = 0,
    edge_ml_scores: Optional[Dict[str, float]] = None,
) -> Dict:
    """Compute the end-to-end fund journey around a focal entity.

    Returns a dict ready for serialisation to the frontend: source entity,
    nodes (with side + flags), links (with flags + ML score), chronological
    timeline of underlying transactions, and a top-level summary.
    """
    if not graph.has_node(entity_id):
        return {"error": f"Entity {entity_id} not found"}

    direction = direction.lower()
    if direction not in {"forward", "backward", "both"}:
        direction = "both"

    risk_map = {r["entity_id"]: r["risk_score"] for r in risk_scores}

    # 1. Walk graph in requested direction(s)
    forward_depth: Dict[str, int] = {}
    backward_depth: Dict[str, int] = {}
    forward_edges: Set[Tuple[str, str]] = set()
    backward_edges: Set[Tuple[str, str]] = set()

    if direction in ("forward", "both"):
        forward_depth, forward_edges = _bfs_in_direction(
            graph, entity_id, "forward", max_hops, min_amount,
        )
    if direction in ("backward", "both"):
        backward_depth, backward_edges = _bfs_in_direction(
            graph, entity_id, "backward", max_hops, min_amount,
        )

    all_node_ids = set(forward_depth.keys()) | set(backward_depth.keys())
    all_edges = forward_edges | backward_edges
    if not all_node_ids:
        all_node_ids = {entity_id}

    # 2. SCC membership across the trace subgraph
    trace_subgraph = graph.subgraph(all_node_ids).copy()
    scc_set: Set[str] = set()
    for comp in nx.strongly_connected_components(trace_subgraph):
        if len(comp) >= 3:
            scc_set.update(comp)

    # 3. Build node list
    nodes_out = []
    for nid in all_node_ids:
        meta = _node_meta(graph, nid, risk_map)
        # Side classification
        if nid == entity_id:
            side = "focus"
            depth = 0
        elif nid in forward_depth and nid not in backward_depth:
            side = "downstream"
            depth = forward_depth[nid]
        elif nid in backward_depth and nid not in forward_depth:
            side = "upstream"
            depth = backward_depth[nid]
        else:
            side = "loop"
            depth = max(forward_depth.get(nid, 0), backward_depth.get(nid, 0))
        meta["side"] = side
        meta["depth"] = depth
        meta["flags"] = _annotate_node_flags(
            graph, nid, transactions, risk_map, in_scc=nid in scc_set,
        )
        nodes_out.append(meta)

    # 4. Build link list, with ML score + flags
    edge_ml_scores = edge_ml_scores or {}
    links_out = []
    for u, v in all_edges:
        ed = graph[u][v]
        ml_score = edge_ml_scores.get(f"{u}->{v}")
        rail_mix = ed.get("rail_mix") or {}
        channel_mix = ed.get("channel_mix") or {}
        links_out.append({
            "source": u,
            "target": v,
            "amount": round(float(ed["total_amount"]), 2),
            "avg_amount": round(float(ed["avg_amount"]), 2),
            "txn_count": int(ed["transaction_count"]),
            "fraud_count": int(ed.get("fraud_count", 0)),
            "first_seen": str(ed.get("first_seen", "")),
            "last_seen": str(ed.get("last_seen", "")),
            "rails": rail_mix,
            "channels": channel_mix,
            "ml_score": round(float(ml_score), 3) if ml_score is not None else None,
            "flags": _annotate_edge_flags(graph, u, v),
        })
    links_out.sort(key=lambda l: l["amount"], reverse=True)

    # 5. Timeline — pull every txn whose endpoints are both in our trace
    txn_mask = (
        transactions["sender_id"].isin(all_node_ids)
        & transactions["receiver_id"].isin(all_node_ids)
    )
    timeline_df = transactions.loc[txn_mask].copy()
    timeline_df = timeline_df.sort_values("timestamp")
    timeline_keep_cols = [
        "transaction_id", "timestamp", "sender_id", "sender_name", "receiver_id",
        "receiver_name", "amount", "transaction_type", "channel", "purpose_code",
        "is_fraud", "fraud_pattern",
    ]
    timeline_keep_cols = [c for c in timeline_keep_cols if c in timeline_df.columns]
    timeline = timeline_df[timeline_keep_cols].astype({c: str for c in timeline_keep_cols if c == "timestamp"}).to_dict("records")
    # Cap timeline to keep payload small
    timeline = timeline[-200:] if len(timeline) > 200 else timeline

    # 6. Aggregate summary
    in_flow = sum(
        graph[u][entity_id]["total_amount"]
        for u in graph.predecessors(entity_id) if (u, entity_id) in all_edges
    )
    out_flow = sum(
        graph[entity_id][v]["total_amount"]
        for v in graph.successors(entity_id) if (entity_id, v) in all_edges
    )

    red_flags: List[str] = []
    if scc_set:
        red_flags.append(f"Cycle detected: {len(scc_set)} entities form a closed loop")
    shell_count = sum(1 for n in nodes_out if "shell_company" in n["flags"])
    if shell_count:
        red_flags.append(f"{shell_count} shell company(ies) in the journey")
    dormant_count = sum(1 for n in nodes_out if "dormant_then_active" in n["flags"])
    if dormant_count:
        red_flags.append(f"{dormant_count} account(s) reactivated after dormancy")
    threshold_edges = sum(1 for l in links_out if "near_reporting_threshold" in l["flags"])
    if threshold_edges:
        red_flags.append(f"{threshold_edges} relationship(s) clustered near the ₹2L reporting threshold")
    fraud_txn_count = sum(1 for t in timeline if t.get("is_fraud"))
    if fraud_txn_count:
        red_flags.append(f"{fraud_txn_count} flagged transactions in this journey")

    return {
        "entity": _node_meta(graph, entity_id, risk_map),
        "direction": direction,
        "max_hops": max_hops,
        "nodes": nodes_out,
        "links": links_out,
        "timeline": timeline,
        "summary": {
            "n_nodes": len(nodes_out),
            "n_links": len(links_out),
            "total_inflow": round(float(in_flow), 2),
            "total_outflow": round(float(out_flow), 2),
            "net_flow": round(float(in_flow - out_flow), 2),
            "forward_hops_reached": max(forward_depth.values()) if forward_depth else 0,
            "backward_hops_reached": max(backward_depth.values()) if backward_depth else 0,
            "n_fraud_txns": fraud_txn_count,
            "red_flags": red_flags,
        },
    }


def trace_for_alert(
    graph: nx.DiGraph,
    transactions: pd.DataFrame,
    risk_scores: List[Dict],
    alert: Dict,
    edge_ml_scores: Optional[Dict[str, float]] = None,
    include_neighbors: bool = False,
) -> Dict:
    """Trace a journey scoped to the specific entities named in an alert.

    By default the trace contains only the entities named in the alert and
    the edges between them — exactly the chain the detector flagged. Pass
    include_neighbors=True to expand by one hop on either side.
    """
    entities = list(alert.get("entities", []))
    if not entities:
        return {"error": "Alert has no entities"}

    focus = entities[0]
    if not graph.has_node(focus):
        return {"error": "Focus entity not found"}

    # Default: only the alert entities themselves
    node_set = set(entities)
    if include_neighbors:
        for e in entities:
            if graph.has_node(e):
                node_set.update(graph.successors(e))
                node_set.update(graph.predecessors(e))

    risk_map = {r["entity_id"]: r["risk_score"] for r in risk_scores}
    edge_ml_scores = edge_ml_scores or {}

    # Collect edges among these nodes
    all_edges = [
        (u, v) for u, v in graph.edges()
        if u in node_set and v in node_set
    ]

    # Build response identical in shape to trace_journey
    sub = graph.subgraph(node_set).copy()
    scc_set: Set[str] = set()
    for comp in nx.strongly_connected_components(sub):
        if len(comp) >= 3:
            scc_set.update(comp)

    entity_set = set(entities)
    nodes_out = []
    for nid in node_set:
        meta = _node_meta(graph, nid, risk_map)
        meta["side"] = "alert" if nid in entity_set else "neighbor"
        meta["depth"] = 0 if nid in entity_set else 1
        meta["flags"] = _annotate_node_flags(graph, nid, transactions, risk_map, in_scc=nid in scc_set)
        nodes_out.append(meta)

    links_out = []
    for u, v in all_edges:
        ed = graph[u][v]
        ml_score = edge_ml_scores.get(f"{u}->{v}")
        links_out.append({
            "source": u,
            "target": v,
            "amount": round(float(ed["total_amount"]), 2),
            "avg_amount": round(float(ed["avg_amount"]), 2),
            "txn_count": int(ed["transaction_count"]),
            "fraud_count": int(ed.get("fraud_count", 0)),
            "first_seen": str(ed.get("first_seen", "")),
            "last_seen": str(ed.get("last_seen", "")),
            "rails": ed.get("rail_mix") or {},
            "channels": ed.get("channel_mix") or {},
            "ml_score": round(float(ml_score), 3) if ml_score is not None else None,
            "flags": _annotate_edge_flags(graph, u, v),
        })
    links_out.sort(key=lambda l: l["amount"], reverse=True)

    txn_mask = (
        transactions["sender_id"].isin(node_set)
        & transactions["receiver_id"].isin(node_set)
    )
    timeline_df = transactions.loc[txn_mask].sort_values("timestamp")
    timeline_keep_cols = [
        c for c in ["transaction_id", "timestamp", "sender_id", "sender_name",
                    "receiver_id", "receiver_name", "amount", "transaction_type",
                    "channel", "purpose_code", "is_fraud", "fraud_pattern"]
        if c in timeline_df.columns
    ]
    timeline = timeline_df[timeline_keep_cols].astype({"timestamp": str}).to_dict("records")
    timeline = timeline[-300:] if len(timeline) > 300 else timeline

    return {
        "entity": _node_meta(graph, focus, risk_map),
        "alert_id": alert.get("alert_id"),
        "pattern_type": alert.get("pattern_type"),
        "direction": "alert_scope",
        "nodes": nodes_out,
        "links": links_out,
        "timeline": timeline,
        "summary": {
            "n_nodes": len(nodes_out),
            "n_links": len(links_out),
            "n_alert_entities": len(entities),
            "n_fraud_txns": sum(1 for t in timeline if t.get("is_fraud")),
            "red_flags": [f for f in [
                f"Cycle of {len(scc_set)} entities inside the alert scope" if scc_set else None,
                f"{sum(1 for n in nodes_out if 'shell_company' in n['flags'])} shell company(ies) involved"
                if any("shell_company" in n["flags"] for n in nodes_out) else None,
                f"{sum(1 for n in nodes_out if 'dormant_then_active' in n['flags'])} dormant-then-active account(s)"
                if any("dormant_then_active" in n["flags"] for n in nodes_out) else None,
            ] if f],
        },
    }
