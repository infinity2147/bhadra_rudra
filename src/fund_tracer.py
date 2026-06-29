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


# Default tunable thresholds — used when no `config` dict is passed.  When a
# ConfigStore-backed dict IS passed, the tracer reads from that instead so
# admin threshold changes propagate to both detection AND tracer annotation.
NIGHT_HOURS = (22, 6)  # 10 PM – 6 AM inclusive
STRUCTURING_THRESHOLD = 200000  # ₹2L
HIGH_VALUE_THRESHOLD = 1000000  # ₹10L


def _cfg(config: Optional[Dict], key: str, default):
    """Read a tracer threshold from a config dict if present, else default."""
    if config is None:
        return default
    return config.get(key, default)


def _scc3_members(graph: nx.DiGraph) -> Set[str]:
    """Union of all nodes in strongly-connected components of size >= 3.

    Memoised on the graph object (`graph.graph`) — Tarjan over a 100k+ node
    graph is far too slow to repeat on every trace request, and the graph is
    stable between rebuilds (a rebuild creates a fresh object, dropping the cache).
    """
    cached = graph.graph.get("_scc3_members")
    if cached is not None:
        return cached
    members: Set[str] = set()
    for comp in nx.strongly_connected_components(graph):
        if len(comp) >= 3:
            members.update(comp)
    graph.graph["_scc3_members"] = members
    return members


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


def _classify_terminals_in_trace(
    graph: nx.DiGraph,
    all_node_ids: set,
    focus: str,
    forward_depth: Dict[str, int],
) -> Dict[str, List[str]]:
    """For the downstream leaves of the trace, return {category: [node_id, ...]}."""
    trace_sub = graph.subgraph(all_node_ids)
    leaves = [
        n for n in all_node_ids
        if n != focus and trace_sub.out_degree(n) == 0 and n in forward_depth
    ]
    by_category: Dict[str, List[str]] = defaultdict(list)
    for leaf in leaves:
        cat = _classify_terminal(graph, leaf)
        by_category[cat].append(leaf)
    return dict(by_category)


def _classify_terminal(graph: nx.DiGraph, node_id: str) -> str:
    """Classify a leaf node by what kind of "exit" the funds took.

    Categories:
      cash_out      — counterparty type or channel suggests cash withdrawal
      cross_border  — RTGS/SWIFT-heavy rail mix on incoming edge
      conversion    — counterparty is an exchange/conversion entity
      layered       — none of the above, just another business/individual sink
                       (likely truncated by max_hops)
    """
    nd = graph.nodes[node_id]
    entity_type = (nd.get("type") or "").lower()
    name = (nd.get("name") or "").lower()
    if any(k in name for k in ("exchange", "wallet", "crypto", "bitcoin")):
        return "conversion"
    if "atm" in name or "cash" in name:
        return "cash_out"
    # Inspect incoming edges' rail mix
    rail_totals: Dict[str, int] = {}
    channel_totals: Dict[str, int] = {}
    for u in graph.predecessors(node_id):
        ed = graph[u][node_id]
        for k, v in (ed.get("rail_mix") or {}).items():
            rail_totals[k] = rail_totals.get(k, 0) + v
        for k, v in (ed.get("channel_mix") or {}).items():
            channel_totals[k] = channel_totals.get(k, 0) + v
    total_rail = sum(rail_totals.values()) or 1
    cross_border_share = (
        rail_totals.get("SWIFT", 0) + rail_totals.get("Wire Transfer", 0)
        + rail_totals.get("RTGS", 0)
    ) / total_rail
    if cross_border_share >= 0.5:
        return "cross_border"
    if channel_totals.get("ATM", 0) / max(sum(channel_totals.values()), 1) >= 0.3:
        return "cash_out"
    return "layered"


def _build_transit_ratios(
    transactions: pd.DataFrame,
    window_hours: float = 1.0,
    min_inflow_threshold: float = 10_000.0,
) -> Dict[str, float]:
    """Per-entity ratio of inflow that exits the account within `window_hours`.

    A pure pass-through mule sees money in and almost immediately routes it
    out — transit_ratio ≈ 1.0.  A genuine business consumer holds funds for
    days/weeks before spending.  This is THE textbook mule fingerprint.

    Algorithm:
      For each entity, walk all transactions ordered by time.  For each
      inflow at time T_in carrying amount A_in, find outflows in
      (T_in, T_in + window_hours] and accumulate the matched amount up to A_in.
      transit_ratio = sum(matched_outflow) / sum(inflow).
    """
    if "timestamp" not in transactions.columns:
        return {}
    df = transactions[["sender_id", "receiver_id", "amount", "timestamp"]].copy()
    df["ts"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("ts")

    inflows_by_eid: Dict[str, list] = defaultdict(list)
    outflows_by_eid: Dict[str, list] = defaultdict(list)
    for row in df.itertuples():
        inflows_by_eid[row.receiver_id].append((row.ts, float(row.amount)))
        outflows_by_eid[row.sender_id].append((row.ts, float(row.amount)))

    window = pd.Timedelta(hours=window_hours)
    transit: Dict[str, float] = {}
    for eid, ins in inflows_by_eid.items():
        total_in = sum(a for _, a in ins)
        if total_in < min_inflow_threshold:
            continue
        outs = outflows_by_eid.get(eid, [])
        if not outs:
            continue
        matched = 0.0
        out_idx = 0
        # Sort once (already sorted, but be defensive)
        outs_sorted = sorted(outs)
        for t_in, a_in in ins:
            need = a_in
            # Find outflows in (t_in, t_in + window]
            while out_idx < len(outs_sorted) and outs_sorted[out_idx][0] <= t_in:
                out_idx += 1
            scan_idx = out_idx
            while scan_idx < len(outs_sorted) and outs_sorted[scan_idx][0] <= t_in + window:
                take = min(need, outs_sorted[scan_idx][1])
                matched += take
                need -= take
                if need <= 0:
                    break
                scan_idx += 1
        ratio = matched / total_in if total_in > 0 else 0.0
        if ratio > 0:
            transit[eid] = round(ratio, 3)
    return transit


def _build_burst_counts(
    transactions: pd.DataFrame,
    window_minutes: int = 60,
    min_cluster: int = 3,
) -> Dict[str, int]:
    """Per-entity count of velocity bursts.

    A "burst" is ≥ `min_cluster` transactions involving that entity within
    `window_minutes`.  This is a much sharper smurfing signal than threshold
    proximity — real mule accounts have rapid-fire bursts during off-hours.

    Returns: entity_id → number of distinct bursts detected.
    """
    if "timestamp" not in transactions.columns:
        return {}
    df = transactions[["sender_id", "receiver_id", "timestamp"]].copy()
    df["ts"] = pd.to_datetime(df["timestamp"], format="mixed")
    # Long-form: one row per (entity, ts) so we can group by entity
    sender_rows   = df[["sender_id", "ts"]].rename(columns={"sender_id": "eid"})
    receiver_rows = df[["receiver_id", "ts"]].rename(columns={"receiver_id": "eid"})
    long = pd.concat([sender_rows, receiver_rows], ignore_index=True)
    long = long.sort_values(["eid", "ts"])

    burst_counts: Dict[str, int] = {}
    window = pd.Timedelta(minutes=window_minutes)
    for eid, grp in long.groupby("eid"):
        ts = grp["ts"].to_numpy()
        if len(ts) < min_cluster:
            continue
        bursts = 0
        i = 0
        while i + min_cluster - 1 < len(ts):
            # If min_cluster consecutive entries fit in the window, count one burst
            if ts[i + min_cluster - 1] - ts[i] <= window:
                bursts += 1
                # Skip past this cluster
                j = i + min_cluster
                while j < len(ts) and ts[j] - ts[i] <= window:
                    j += 1
                i = j
            else:
                i += 1
        if bursts > 0:
            burst_counts[eid] = bursts
    return burst_counts


def _build_baseline_stats(transactions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per-entity baseline of *its own* historical monthly outflow.

    Returns: entity_id → {mean, std, current, zscore} for monthly send-side amount.

    The detection signal:  an entity normally moving ₹50K/month suddenly moving
    ₹50L is a critical AML red flag that fixed-threshold detectors miss because
    ₹50L is below the structuring threshold for many corporate accounts.

    Implementation: group transactions by sender_id + month, take the most-recent
    month as "current" and the rest as the baseline.  Need ≥ 2 historical months
    before we can declare an anomaly.
    """
    if "timestamp" not in transactions.columns:
        return {}
    df = transactions[["sender_id", "timestamp", "amount"]].copy()
    df["ym"] = pd.to_datetime(df["timestamp"], format="mixed").dt.to_period("M")
    monthly = df.groupby(["sender_id", "ym"])["amount"].sum().reset_index()

    out: Dict[str, Dict[str, float]] = {}
    for sender_id, grp in monthly.groupby("sender_id"):
        if len(grp) < 3:                       # need ≥ 2 baseline months
            continue
        grp_sorted = grp.sort_values("ym")
        history = grp_sorted["amount"].iloc[:-1].to_numpy()
        current = float(grp_sorted["amount"].iloc[-1])
        mean = float(history.mean())
        std  = float(history.std(ddof=0))
        z    = (current - mean) / std if std > 0 else 0.0
        out[sender_id] = {
            "baseline_mean": round(mean, 2),
            "baseline_std":  round(std, 2),
            "current_month": round(current, 2),
            "zscore":        round(float(z), 3),
        }
    return out


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
    max_nodes: int = 400,
) -> Tuple[Dict[str, int], Set[Tuple[str, str]]]:
    """Walk the graph from `source` and return (node_depth, edges_visited).

    direction ∈ {"forward", "backward"}.
    Edges below `min_amount` are skipped — same convention as the rest of the app.
    `max_nodes` caps the frontier: a high-degree hub at 3 hops over a 100k-node
    graph would otherwise explode the walk (measured ~9s) plus the per-node
    annotation that follows. A journey of a few hundred nodes is already past
    what's readable, so we stop pulling in new nodes once the cap is hit. Every
    emitted edge still connects two visited nodes (no dangling endpoints).
    """
    visited_depth = {source: 0}
    edges: Set[Tuple[str, str]] = set()
    frontier: Set[str] = {source}

    for depth in range(max_hops):
        if len(visited_depth) >= max_nodes:
            break
        # Use a set to dedupe within a single BFS round — multiple frontier
        # nodes can point to the same neighbour, and the old list version would
        # process that neighbour once per pointer.
        next_frontier: Set[str] = set()
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
                if nb not in visited_depth:
                    if len(visited_depth) >= max_nodes:
                        continue   # at cap: don't pull in new nodes (or their edges)
                    visited_depth[nb] = depth + 1
                    next_frontier.add(nb)
                edges.add(edge)   # nb is now visited (or already was) → no dangling edge
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
    baseline_stats: Optional[Dict[str, Dict[str, float]]] = None,
    burst_counts: Optional[Dict[str, int]] = None,
    transit_ratios: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Flag a node for suspicious characteristics.

    `txn_index` is the pre-built entity→DataFrame map from _build_txn_index().
    Using the index avoids an O(N_txns) boolean scan per node.

    `baseline_stats` (optional) is the per-entity historical baseline from
    _build_baseline_stats; if the current month's outflow is >3σ above this
    entity's own historical mean, raise the outflow_zscore_anomaly flag.
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
    # Behavioural baseline anomaly — entity's own monthly outflow is >3σ above
    # its historical norm.  This catches the "₹50K-norm account suddenly moves
    # ₹50L" pattern that fixed thresholds miss.
    if baseline_stats:
        bs = baseline_stats.get(node_id)
        if bs and bs["zscore"] >= 3.0:
            flags.append("outflow_zscore_anomaly")
    # Velocity bursts — rapid-fire transaction clusters are the smurfing signature
    if burst_counts and burst_counts.get(node_id, 0) >= 1:
        flags.append("velocity_burst")
    # Transit-node signature — money in then immediately out is a mule
    if transit_ratios and transit_ratios.get(node_id, 0) >= 0.5:
        flags.append("transit_node")
    return flags


def _build_link(
    graph: nx.DiGraph,
    u: str,
    v: str,
    edge_ml_scores: Dict[str, float],
    config: Optional[Dict] = None,
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
        "flags": _annotate_edge_flags(graph, u, v, config=config),
    }


def _annotate_edge_flags(
    graph: nx.DiGraph,
    u: str,
    v: str,
    config: Optional[Dict] = None,
) -> List[str]:
    flags: List[str] = []
    ed = graph[u][v]
    avg = float(ed.get("avg_amount", 0))
    fraud_count = int(ed.get("fraud_count", 0))
    if fraud_count > 0:
        flags.append("contains_fraud_txn")
    # Below-threshold structuring hint — threshold from ConfigStore if provided
    structuring = _cfg(config, "tracer_structuring_threshold", STRUCTURING_THRESHOLD)
    if avg < structuring and avg > 0.7 * structuring:
        flags.append("near_reporting_threshold")
    # High-value rail share
    rail_mix = ed.get("rail_mix") or {}
    total = sum(rail_mix.values()) or 1
    high_share = (rail_mix.get("RTGS", 0) + rail_mix.get("Wire Transfer", 0)) / total
    high_value = _cfg(config, "tracer_high_value_threshold", HIGH_VALUE_THRESHOLD)
    if high_share > 0.5 and ed.get("total_amount", 0) > high_value:
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
    config: Optional[Dict] = None,
    txn_index: Optional[Dict] = None,
    baseline_stats: Optional[Dict] = None,
    burst_counts: Optional[Dict] = None,
    transit_ratios: Optional[Dict] = None,
) -> Dict:
    """Compute the end-to-end fund journey around a focal entity.

    Returns a dict ready for serialisation to the frontend: source entity,
    nodes (with side + flags), links (with flags + ML score), chronological
    timeline of underlying transactions, and a top-level summary.

    The four per-entity structures (txn_index/baseline_stats/burst_counts/
    transit_ratios) each cost a full O(N_txns) pass to build. Callers that
    serve many requests (the API) build them ONCE at startup and pass them in;
    when omitted they're built on demand so standalone use still works.
    """
    if not graph.has_node(entity_id):
        return {"error": f"Entity {entity_id} not found"}

    direction = direction.lower()
    if direction not in {"forward", "backward", "both"}:
        direction = "both"

    risk_map = {r["entity_id"]: r["risk_score"] for r in risk_scores}

    # Use caller-supplied caches when available; otherwise build on demand.
    if txn_index is None:
        txn_index = _build_txn_index(transactions)
    if baseline_stats is None:
        baseline_stats = _build_baseline_stats(transactions)
    if burst_counts is None:
        burst_counts = _build_burst_counts(transactions)
    if transit_ratios is None:
        transit_ratios = _build_transit_ratios(transactions)

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
    # Memoised on the graph so we don't re-run Tarjan over 100k+ nodes per call.
    scc_set = _scc3_members(graph) & all_node_ids

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
            baseline_stats=baseline_stats, burst_counts=burst_counts,
            transit_ratios=transit_ratios,
        )
        nodes_out.append(meta)

    # 4. Build link list, with ML score + flags
    edge_ml_scores = edge_ml_scores or {}
    links_out = [_build_link(graph, u, v, edge_ml_scores, config=config) for u, v in all_edges]
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
    flow_distribution: List[Dict] = []
    if direction in ("forward", "both"):
        # Candidate sink nodes: downstream leaves (out-degree 0 within trace).
        # Materialise (.copy()) so dijkstra/max-flow run on a real small graph,
        # not a view over the 100k-node parent.
        trace_sub = graph.subgraph(all_node_ids).copy()
        sinks = [n for n in all_node_ids if n != entity_id and trace_sub.out_degree(n) == 0]
        # Also include the farthest-depth downstream nodes if no pure sinks
        if not sinks and forward_depth:
            max_d = max(forward_depth.values())
            sinks = [n for n, d in forward_depth.items() if d == max_d and n != entity_id]
        for sink in sinks[:5]:  # limit candidates
            try:
                # Weight = 1 / (amount * (risk + 0.1)): higher-flow AND higher-risk
                # edges get lower cost, so Dijkstra surfaces the path that's both
                # high-throughput AND passes through suspicious entities.  This is
                # what an investigator actually wants prioritised — not just the
                # biggest pipe but the most suspicious biggest pipe.
                def risk_weight(u, v, d):
                    amt = d.get("total_amount", 0)
                    target_risk = risk_map.get(v, 0.0)
                    return 1.0 / ((amt + 1.0) * (target_risk + 0.1))
                # Run on the traced SUBGRAPH, not the full 100k-node graph:
                # Dijkstra (Python weight callable) + max-flow over the whole
                # bank per sink was the ~9s journey cost — and wrong, since a
                # dominant path must stay within the journey being shown.
                path = nx.dijkstra_path(trace_sub, entity_id, sink, weight=risk_weight)
                path_amount = min(
                    graph[path[i]][path[i + 1]].get("total_amount", 0)
                    for i in range(len(path) - 1)
                ) if len(path) > 1 else 0
                path_risk = sum(risk_map.get(n, 0.0) for n in path) / max(len(path), 1)
                dominant_paths.append({
                    "path": path,
                    "sink": sink,
                    "bottleneck_amount": round(float(path_amount), 2),
                    "avg_node_risk": round(float(path_risk), 3),
                    "hops": len(path) - 1,
                })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            # Max-flow distribution: shows how funds spread across parallel chains.
            # A laundering operation that splits ₹10L into 5 chains of ₹2L is
            # visible here but invisible in the single Dijkstra path above.
            try:
                flow_value, flow_dict = nx.maximum_flow(
                    trace_sub, entity_id, sink, capacity="total_amount",
                )
                edge_contribs = [
                    {"source": u, "target": v, "flow": round(float(f), 2)}
                    for u, nbrs in flow_dict.items()
                    for v, f in nbrs.items() if f > 0
                ]
                edge_contribs.sort(key=lambda e: e["flow"], reverse=True)
                # Heuristic: count edges carrying >10% of max flow as "parallel paths"
                threshold = max(flow_value * 0.1, 1.0)
                n_parallel = sum(1 for e in edge_contribs if e["flow"] >= threshold)
                flow_distribution.append({
                    "sink": sink,
                    "max_flow_amount": round(float(flow_value), 2),
                    "top_edges": edge_contribs[:8],
                    "n_parallel_paths_estimate": n_parallel,
                })
            except (nx.NetworkXError, nx.NodeNotFound, KeyError):
                pass
        # Sort by bottleneck descending, keep top 3
        dominant_paths.sort(key=lambda p: p["bottleneck_amount"], reverse=True)
        dominant_paths = dominant_paths[:3]
        flow_distribution.sort(key=lambda f: f["max_flow_amount"], reverse=True)
        flow_distribution = flow_distribution[:3]

        # Path-level red-flag aggregation: walk each dominant path, union the
        # per-node and per-edge flags, derive a composite 0-1 risk score so the
        # investigator sees one number per path rather than scrolling through
        # individual node tooltips.
        for dp in dominant_paths:
            path_node_flags: Set[str] = set()
            path_edge_flags: Set[str] = set()
            risk_sum = 0.0
            fraud_edges = 0
            for nid in dp["path"]:
                path_node_flags.update(
                    _annotate_node_flags(graph, nid, txn_index, risk_map,
                                          in_scc=nid in scc_set,
                                          baseline_stats=baseline_stats,
                                          burst_counts=burst_counts,
                                          transit_ratios=transit_ratios)
                )
                risk_sum += risk_map.get(nid, 0.0)
            for i in range(len(dp["path"]) - 1):
                u, v = dp["path"][i], dp["path"][i + 1]
                if graph.has_edge(u, v):
                    path_edge_flags.update(_annotate_edge_flags(graph, u, v, config=config))
                    if int(graph[u][v].get("fraud_count", 0)) > 0:
                        fraud_edges += 1
            n_hops = max(dp["hops"], 1)
            composite = (
                (risk_sum / n_hops)
                + 0.2 * fraud_edges
                + 0.1 * len(path_node_flags)
                + 0.05 * len(path_edge_flags)
            )
            dp["path_flags"] = sorted(path_node_flags | path_edge_flags)
            dp["path_risk_score"] = round(min(composite, 1.0), 3)

    return {
        "entity": _node_meta(graph, entity_id, risk_map),
        "direction": direction,
        "max_hops": max_hops,
        "nodes": nodes_out,
        "links": links_out,
        "timeline": timeline,
        "dominant_paths": dominant_paths,
        "flow_distribution": flow_distribution,
        "terminal_classification": _classify_terminals_in_trace(
            graph, all_node_ids, entity_id, forward_depth,
        ),
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
    config: Optional[Dict] = None,
    txn_index: Optional[Dict] = None,
    baseline_stats: Optional[Dict] = None,
    burst_counts: Optional[Dict] = None,
    transit_ratios: Optional[Dict] = None,
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

    # Use caller-supplied caches when available; otherwise build on demand.
    if txn_index is None:
        txn_index = _build_txn_index(transactions)
    if baseline_stats is None:
        baseline_stats = _build_baseline_stats(transactions)
    if burst_counts is None:
        burst_counts = _build_burst_counts(transactions)
    if transit_ratios is None:
        transit_ratios = _build_transit_ratios(transactions)

    # Collect edges among these nodes
    all_edges = [
        (u, v) for u, v in graph.edges()
        if u in node_set and v in node_set
    ]

    # SCC from the FULL graph (same reasoning as trace_journey — cycles can
    # extend beyond the alert entity set). Memoised on the graph.
    scc_set = _scc3_members(graph) & node_set

    entity_set = set(entities)
    nodes_out = []
    for nid in node_set:
        meta = _node_meta(graph, nid, risk_map)
        meta["side"] = "alert" if nid in entity_set else "neighbor"
        meta["depth"] = 0 if nid in entity_set else 1
        meta["flags"] = _annotate_node_flags(
            graph, nid, txn_index, risk_map,
            in_scc=nid in scc_set, baseline_stats=baseline_stats,
            burst_counts=burst_counts, transit_ratios=transit_ratios,
        )
        nodes_out.append(meta)

    links_out = [_build_link(graph, u, v, edge_ml_scores, config=config) for u, v in all_edges]
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
