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


def _build_txn_index(transactions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pre-group transactions by entity once — O(N_txns).

    Returns a dict mapping entity_id → sub-DataFrame of rows where that entity
    appears as sender OR receiver.  Every subsequent per-node lookup is a single
    dict access instead of an O(N_txns) boolean-mask scan.

    Uses raw numpy arrays for the grouping loop to avoid pandas row overhead.
    """
    senders   = transactions["sender_id"].to_numpy()
    receivers = transactions["receiver_id"].to_numpy()
    indices   = transactions.index.to_numpy()

    groups: Dict[str, list] = defaultdict(list)
    for i in range(len(indices)):
        groups[senders[i]].append(indices[i])
        if receivers[i] != senders[i]:          # avoid double-counting same-entity rows
            groups[receivers[i]].append(indices[i])
    return {eid: transactions.loc[idxs] for eid, idxs in groups.items()}


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
    txn_index: Dict[str, pd.DataFrame],
    risk_map: Dict[str, float],
    in_scc: bool,
) -> List[str]:
    """Flag a node for suspicious characteristics.

    `txn_index` is the pre-built entity→DataFrame map from _build_txn_index().
    Using the index avoids an O(N_txns) boolean scan per node.
    """
    flags: List[str] = []
    nd = graph.nodes[node_id]
    if nd.get("type") == "shell_company":
        flags.append("shell_company")
    if risk_map.get(node_id, 0) >= 0.5:
        flags.append("high_risk")
    if in_scc:
        flags.append("part_of_cycle")
    # Cross-branch activity check — O(1) lookup via pre-built index
    node_txns = txn_index.get(node_id)
    if node_txns is not None and len(node_txns) > 0:
        branches = set(node_txns["sender_branch"].tolist() + node_txns["receiver_branch"].tolist())
        branches.discard("")
        if len(branches) >= 5:
            flags.append("multi_branch_activity")
        # Dormant signature: gap > 30 days then activity
        ts = pd.to_datetime(node_txns["timestamp"], format="mixed").sort_values()
        if len(ts) >= 2:
            diffs = ts.diff().dt.total_seconds() / 86400.0
            if diffs.max() >= 30:
                flags.append("dormant_then_active")
    return flags


def _build_link(
    graph: nx.DiGraph,
    u: str,
    v: str,
    edge_ml_scores: Dict[str, float],
) -> Dict:
    """Construct the serialisable link dict for one graph edge.

    Extracted to avoid copy-paste between trace_journey and trace_for_alert.
    Includes velocity fields (time_span_hours, txn_velocity) so the UI and
    downstream detectors can spot rapid-fire layering without extra queries.
    """
    ed = graph[u][v]
    ml_score = edge_ml_scores.get(f"{u}->{v}")

    # Velocity: how many transactions per hour on this edge?
    first_seen = ed.get("first_seen")
    last_seen  = ed.get("last_seen")
    try:
        span_hours = (
            pd.to_datetime(last_seen) - pd.to_datetime(first_seen)
        ).total_seconds() / 3600.0
    except Exception:
        span_hours = 0.0
    txn_count = int(ed["transaction_count"])
    txn_velocity = round(txn_count / max(span_hours, 0.01), 4)  # txns/hour

    return {
        "source": u,
        "target": v,
        "amount": round(float(ed["total_amount"]), 2),
        "avg_amount": round(float(ed["avg_amount"]), 2),
        "txn_count": txn_count,
        "fraud_count": int(ed.get("fraud_count", 0)),
        "first_seen": str(first_seen or ""),
        "last_seen": str(last_seen or ""),
        "time_span_hours": round(span_hours, 4),
        "txn_velocity": txn_velocity,
        "rails": ed.get("rail_mix") or {},
        "channels": ed.get("channel_mix") or {},
        "ml_score": round(float(ml_score), 3) if ml_score is not None else None,
        "flags": _annotate_edge_flags(graph, u, v),
    }


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

    # Pre-build entity→transactions index ONCE (O(N_txns))
    # so _annotate_node_flags can do a dict lookup instead of a full scan.
    txn_index = _build_txn_index(transactions)

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

    # 2. SCC membership — use the FULL graph so cycles that extend beyond
    # max_hops are still detected.  A node in a 6-hop cycle with max_hops=3
    # would be invisible if we only ran SCC on the BFS-clipped subgraph.
    scc_set: Set[str] = set()
    for comp in nx.strongly_connected_components(graph):
        if len(comp) >= 3:
            scc_set.update(comp)
    # Keep only the SCC members that are actually in our trace
    scc_set &= all_node_ids

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
            graph, nid, txn_index, risk_map, in_scc=nid in scc_set,
        )
        nodes_out.append(meta)

    # 4. Build link list, with ML score + flags
    edge_ml_scores = edge_ml_scores or {}
    links_out = [_build_link(graph, u, v, edge_ml_scores) for u, v in all_edges]
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
    # Cap timeline to keep payload small, but preserve BOTH the earliest
    # and most-recent transactions.  Dropping the earliest silently hides
    # the origin of the funds, which is usually the most important evidence.
    _CAP = 200
    if len(timeline) > _CAP:
        half = _CAP // 2
        timeline = timeline[:half] + timeline[-half:]

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

    # 7. Dominant-flow paths — the top-3 highest-throughput routes FROM the
    # focus entity.  Uses Dijkstra with negated total_amount as the cost so
    # the "shortest" path is the one carrying the most money.  This answers
    # "where did the bulk of the funds actually go?" without the investigator
    # having to manually trace the force-graph.
    dominant_paths: List[Dict] = []
    if direction in ("forward", "both"):
        # Candidate sink nodes: downstream leaves (out-degree 0 within trace)
        trace_sub = graph.subgraph(all_node_ids)
        sinks = [n for n in all_node_ids if n != entity_id and trace_sub.out_degree(n) == 0]
        # Also include the farthest-depth downstream nodes if no pure sinks
        if not sinks and forward_depth:
            max_d = max(forward_depth.values())
            sinks = [n for n, d in forward_depth.items() if d == max_d and n != entity_id]
        for sink in sinks[:5]:  # limit candidates
            try:
                # Weight = 1/(amount+1): higher-flow edges get lower cost,
                # so Dijkstra finds the path of maximum throughput.
                path = nx.dijkstra_path(
                    graph, entity_id, sink,
                    weight=lambda u, v, d: 1.0 / (d.get("total_amount", 0) + 1.0),
                )
                path_amount = min(
                    graph[path[i]][path[i + 1]].get("total_amount", 0)
                    for i in range(len(path) - 1)
                ) if len(path) > 1 else 0
                dominant_paths.append({
                    "path": path,
                    "sink": sink,
                    "bottleneck_amount": round(float(path_amount), 2),
                    "hops": len(path) - 1,
                })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        # Sort by bottleneck descending, keep top 3
        dominant_paths.sort(key=lambda p: p["bottleneck_amount"], reverse=True)
        dominant_paths = dominant_paths[:3]

    return {
        "entity": _node_meta(graph, entity_id, risk_map),
        "direction": direction,
        "max_hops": max_hops,
        "nodes": nodes_out,
        "links": links_out,
        "timeline": timeline,
        "dominant_paths": dominant_paths,
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
    max_hops: int = 1,
) -> Dict:
    """Trace a journey scoped to the specific entities named in an alert.

    `include_neighbors=False` (default): only the entities named in the alert
    and the edges between them — exactly the chain the detector flagged.

    `include_neighbors=True` or `max_hops > 1`: BFS-expand by `max_hops` hops
    on both sides, giving the investigator more context around the alert.
    `include_neighbors=True` is a convenience alias for `max_hops=1`.
    """
    entities = list(alert.get("entities", []))
    if not entities:
        return {"error": "Alert has no entities"}

    focus = entities[0]
    if not graph.has_node(focus):
        return {"error": "Focus entity not found"}

    # Resolve include_neighbors → max_hops for backwards compatibility
    effective_hops = max(max_hops, 1 if include_neighbors else 0)

    node_set = set(entities)
    if effective_hops > 0:
        frontier = set(entities)
        for _ in range(effective_hops):
            next_frontier: set = set()
            for e in frontier:
                if graph.has_node(e):
                    next_frontier.update(graph.successors(e))
                    next_frontier.update(graph.predecessors(e))
            new_nodes = next_frontier - node_set
            node_set.update(new_nodes)
            frontier = new_nodes
            if not frontier:
                break

    risk_map = {r["entity_id"]: r["risk_score"] for r in risk_scores}
    edge_ml_scores = edge_ml_scores or {}

    # Pre-build entity→transactions index ONCE for flag annotation
    txn_index = _build_txn_index(transactions)

    # Collect edges among these nodes
    all_edges = [
        (u, v) for u, v in graph.edges()
        if u in node_set and v in node_set
    ]

    # SCC from the FULL graph (same reasoning as trace_journey — cycles can
    # extend beyond the alert entity set).
    scc_set: Set[str] = set()
    for comp in nx.strongly_connected_components(graph):
        if len(comp) >= 3:
            scc_set.update(comp)
    scc_set &= node_set

    entity_set = set(entities)
    nodes_out = []
    for nid in node_set:
        meta = _node_meta(graph, nid, risk_map)
        meta["side"] = "alert" if nid in entity_set else "neighbor"
        meta["depth"] = 0 if nid in entity_set else 1
        meta["flags"] = _annotate_node_flags(graph, nid, txn_index, risk_map, in_scc=nid in scc_set)
        nodes_out.append(meta)

    links_out = [_build_link(graph, u, v, edge_ml_scores) for u, v in all_edges]
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
    _CAP = 300
    if len(timeline) > _CAP:
        half = _CAP // 2
        timeline = timeline[:half] + timeline[-half:]

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
