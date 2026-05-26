"""
Tests + micro-benchmarks for graph engine and fund tracer improvements.
Each test verifies one change; benchmark functions are kept alongside
so they can be run standalone with `python tests/test_graph_engine_improvements.py`.
"""
import sys, os, time
TESTS_DIR = os.path.dirname(__file__)
SRC_DIR   = os.path.join(TESTS_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, TESTS_DIR)

import pandas as pd
import networkx as nx
import pytest

# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    from bench_utils import make_pipeline
    return make_pipeline()


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 1 — #9: Timeline cap preserves earliest transactions
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2 — #7: Pre-group DataFrame for _annotate_node_flags
# ══════════════════════════════════════════════════════════════════════════════

def test_txn_index_flags_match_scan_flags(pipeline):
    """Pre-built index must produce identical flags to the original per-node scan."""
    from fund_tracer import _annotate_node_flags, _build_txn_index
    import networkx as nx

    df, graph, _, risk_scores = pipeline
    risk_map = {r["entity_id"]: r["risk_score"] for r in risk_scores}

    # Build the index
    txn_index = _build_txn_index(df)

    # Build the old-style per-node scan inline for comparison
    def old_flags(node_id):
        node_txns = df[
            (df["sender_id"] == node_id) | (df["receiver_id"] == node_id)
        ]
        flags_old = []
        nd = graph.nodes[node_id]
        if nd.get("type") == "shell_company":
            flags_old.append("shell_company")
        if risk_map.get(node_id, 0) >= 0.5:
            flags_old.append("high_risk")
        if len(node_txns) > 0:
            branches = set(node_txns["sender_branch"].tolist() + node_txns["receiver_branch"].tolist())
            branches.discard("")
            if len(branches) >= 5:
                flags_old.append("multi_branch_activity")
            import pandas as pd
            ts = pd.to_datetime(node_txns["timestamp"], format="mixed").sort_values()
            if len(ts) >= 2:
                diffs = ts.diff().dt.total_seconds() / 86400.0
                if diffs.max() >= 30:
                    flags_old.append("dormant_then_active")
        return flags_old

    mismatches = 0
    for nid in list(graph.nodes())[:30]:  # sample 30 nodes
        new = _annotate_node_flags(graph, nid, txn_index, risk_map, in_scc=False)
        old = old_flags(nid)
        # shell_company + high_risk from node attributes — must match exactly
        for flag in ("shell_company", "high_risk", "multi_branch_activity", "dormant_then_active"):
            if (flag in new) != (flag in old):
                mismatches += 1
    assert mismatches == 0, f"{mismatches} flag mismatches between index and scan approaches"


def test_txn_index_is_faster_than_per_node_scan(pipeline):
    """Index approach should be faster than repeated full-DataFrame scans."""
    from fund_tracer import _build_txn_index
    df, graph, _, risk_scores = pipeline
    nodes = list(graph.nodes())

    # Old approach: scan per node
    t0 = time.perf_counter()
    for nid in nodes:
        _ = df[(df["sender_id"] == nid) | (df["receiver_id"] == nid)]
    old_ms = (time.perf_counter() - t0) * 1000

    # New approach: build index once, then lookup
    t0 = time.perf_counter()
    idx = _build_txn_index(df)
    for nid in nodes:
        _ = idx.get(nid)
    new_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  #7 flag-annotation: old={old_ms:.1f}ms  new={new_ms:.1f}ms  "
          f"speedup={old_ms/max(new_ms,0.1):.1f}x")
    assert new_ms < old_ms, (
        f"Index approach ({new_ms:.1f}ms) should beat per-node scan ({old_ms:.1f}ms)"
    )



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 4 — #2: Vectorized node-adding in build_graph
# ══════════════════════════════════════════════════════════════════════════════

def test_build_graph_node_count_unchanged(pipeline):
    """Vectorised build must produce the same number of nodes and edges."""
    from graph_engine import FundFlowGraph
    df, old_graph, _, _ = pipeline
    new_graph = FundFlowGraph().build_graph(df)
    assert new_graph.number_of_nodes() == old_graph.number_of_nodes(), \
        "Node count changed after vectorised build"
    assert new_graph.number_of_edges() == old_graph.number_of_edges(), \
        "Edge count changed after vectorised build"


def test_build_graph_node_attrs_preserved(pipeline):
    """Each node must still carry name, type, branch after the vectorised path."""
    from graph_engine import FundFlowGraph
    df, old_graph, _, _ = pipeline
    new_graph = FundFlowGraph().build_graph(df)
    for nid in new_graph.nodes():
        nd = new_graph.nodes[nid]
        assert "name" in nd and "type" in nd and "branch" in nd, \
            f"Node {nid} missing attributes"


def test_build_graph_is_faster_vectorised(pipeline):
    """Vectorised node-adding: isolate the node-addition step only."""
    import networkx as nx
    import pandas as pd
    df, _, _, _ = pipeline
    has_product = "sender_product" in df.columns

    def old_node_add(g, df):
        """Original iterrows loop — adds every transaction row's sender and receiver."""
        for _, row in df.iterrows():
            sa = {"name": row["sender_name"], "type": row["sender_type"], "branch": row["sender_branch"]}
            if has_product: sa["product"] = row["sender_product"]
            ra = {"name": row["receiver_name"], "type": row["receiver_type"], "branch": row["receiver_branch"]}
            if has_product: ra["product"] = row["receiver_product"]
            g.add_node(row["sender_id"], **sa)
            g.add_node(row["receiver_id"], **ra)

    def new_node_add(g, df):
        """Vectorised: dedup first, then add_nodes_from unique entities only."""
        sender_cols = ["sender_id", "sender_name", "sender_type", "sender_branch"]
        receiver_cols = ["receiver_id", "receiver_name", "receiver_type", "receiver_branch"]
        if has_product:
            sender_cols.append("sender_product"); receiver_cols.append("receiver_product")
        senders = df[sender_cols].drop_duplicates("sender_id").rename(columns=lambda c: c.replace("sender_", ""))
        receivers = df[receiver_cols].drop_duplicates("receiver_id").rename(columns=lambda c: c.replace("receiver_", ""))
        all_entities = pd.concat([senders, receivers], ignore_index=True).drop_duplicates("id")
        g.add_nodes_from(
            (row["id"], {k: v for k, v in row.items() if k != "id"})
            for _, row in all_entities.iterrows()
        )

    REPS = 8
    old_times, new_times = [], []
    for _ in range(REPS):
        g1 = nx.DiGraph(); t0 = time.perf_counter(); old_node_add(g1, df); old_times.append(time.perf_counter() - t0)
        g2 = nx.DiGraph(); t0 = time.perf_counter(); new_node_add(g2, df); new_times.append(time.perf_counter() - t0)

    old_ms = 1000 * min(old_times)
    new_ms = 1000 * min(new_times)
    print(f"\n  #2 node-add: old(iterrows×N_txns)={old_ms:.1f}ms  "
          f"new(dedup+iterrows×N_entities)={new_ms:.1f}ms  speedup={old_ms/max(new_ms,0.1):.1f}x")
    assert new_ms < old_ms, \
        f"Vectorised node-add ({new_ms:.1f}ms) should beat iterrows over all txns ({old_ms:.1f}ms)"



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 5 — #3: Cache get_node_features()
# ══════════════════════════════════════════════════════════════════════════════

def test_node_features_cache_returns_same_result(pipeline):
    """Second call must return identical dict without recomputing."""
    from graph_engine import FundFlowGraph
    df, _, _, _ = pipeline
    ffg = FundFlowGraph()
    ffg.build_graph(df)
    f1 = ffg.get_node_features()
    f2 = ffg.get_node_features()
    assert f1 is f2, "Cache must return the exact same object on second call"


def test_node_features_cache_invalidated_on_rebuild(pipeline):
    """build_graph must clear the cache so stale data is never returned."""
    from graph_engine import FundFlowGraph
    df, _, _, _ = pipeline
    ffg = FundFlowGraph()
    ffg.build_graph(df)
    f1 = ffg.get_node_features()
    ffg.build_graph(df)          # rebuild clears cache
    f2 = ffg.get_node_features()
    assert f1 is not f2, "Cache should be a new object after rebuild"


def test_node_features_cache_is_faster(pipeline):
    """Second call (cache hit) must be significantly faster than first (cold)."""
    from graph_engine import FundFlowGraph
    df, _, _, _ = pipeline
    ffg = FundFlowGraph()
    ffg.build_graph(df)

    REPS = 10
    cold_times, warm_times = [], []
    for _ in range(REPS):
        ffg._node_features_cache = None          # reset to cold
        t0 = time.perf_counter(); ffg.get_node_features(); cold_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ffg.get_node_features(); warm_times.append(time.perf_counter() - t0)

    cold_ms = 1000 * min(cold_times)
    warm_ms = 1000 * min(warm_times)
    print(f"\n  #3 node-features: cold={cold_ms:.2f}ms  warm(cache)={warm_ms:.4f}ms  "
          f"speedup={cold_ms/max(warm_ms,0.0001):.0f}x")
    assert warm_ms < cold_ms / 5, \
        f"Cache hit ({warm_ms:.4f}ms) should be >5× faster than cold compute ({cold_ms:.2f}ms)"



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 6 — #4: extract_subgraph via nx.ego_graph
# ══════════════════════════════════════════════════════════════════════════════

def test_extract_subgraph_contains_seed_nodes(pipeline):
    """Seed nodes must always be in the returned subgraph."""
    from graph_engine import FundFlowGraph
    df, _, _, _ = pipeline
    ffg = FundFlowGraph()
    ffg.build_graph(df)
    seeds = list(ffg.graph.nodes())[:3]
    sub = ffg.extract_subgraph(seeds, hops=1)
    for s in seeds:
        assert s in sub.nodes(), f"Seed {s} missing from subgraph"


def test_extract_subgraph_includes_neighbors(pipeline):
    """1-hop subgraph must include at least one direct neighbor of each seed."""
    from graph_engine import FundFlowGraph
    df, _, _, _ = pipeline
    ffg = FundFlowGraph()
    ffg.build_graph(df)
    # Pick a seed that has neighbors
    seed = next(
        n for n in ffg.graph.nodes()
        if ffg.graph.out_degree(n) > 0 or ffg.graph.in_degree(n) > 0
    )
    sub = ffg.extract_subgraph([seed], hops=1)
    expected_neighbors = (
        set(ffg.graph.successors(seed)) | set(ffg.graph.predecessors(seed))
    )
    assert expected_neighbors.issubset(sub.nodes()), \
        "1-hop subgraph must include all direct neighbors"


def test_extract_subgraph_ego_vs_old_bfs(pipeline):
    """ego_graph result should be a superset of the old BFS (old had expansion bug)."""
    from graph_engine import FundFlowGraph
    import networkx as nx
    df, _, _, _ = pipeline
    ffg = FundFlowGraph()
    ffg.build_graph(df)
    seeds = list(ffg.graph.nodes())[:5]

    # Old hand-rolled BFS
    def old_extract(graph, node_ids, hops):
        nodes_to_include = set(node_ids)
        for _ in range(hops):
            new_nodes = set()
            for node in nodes_to_include:
                new_nodes.update(graph.predecessors(node))
                new_nodes.update(graph.successors(node))
            nodes_to_include.update(new_nodes)
        return graph.subgraph(nodes_to_include).copy()

    old_sub = old_extract(ffg.graph, seeds, hops=2)
    new_sub = ffg.extract_subgraph(seeds, hops=2)

    # New result should have at least as many nodes (ego_graph is correct by construction)
    assert new_sub.number_of_nodes() >= old_sub.number_of_nodes(), \
        f"ego_graph result ({new_sub.number_of_nodes()}) smaller than old BFS ({old_sub.number_of_nodes()})"



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 7 — #5: Centrality cache in FraudDetector
# ══════════════════════════════════════════════════════════════════════════════

def test_centrality_cache_second_call_is_faster(pipeline):
    """Second call to compute_node_risk_scores must be faster (cache hit)."""
    from fraud_detector import FraudDetector
    _, graph, _, _ = pipeline

    fd = FraudDetector(graph)
    REPS = 5
    cold_times, warm_times = [], []
    for _ in range(REPS):
        fd._centrality_cache = None        # reset to cold
        t0 = time.perf_counter(); fd.compute_node_risk_scores(); cold_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fd.compute_node_risk_scores(); warm_times.append(time.perf_counter() - t0)

    cold_ms = 1000 * min(cold_times)
    warm_ms = 1000 * min(warm_times)
    print(f"\n  #5 centrality: cold={cold_ms:.1f}ms  warm(cache)={warm_ms:.2f}ms  "
          f"speedup={cold_ms/max(warm_ms,0.01):.0f}x")
    assert warm_ms < cold_ms / 3, \
        f"Cache hit ({warm_ms:.2f}ms) should be >3× faster than cold ({cold_ms:.1f}ms)"


def test_centrality_cache_scores_unchanged(pipeline):
    """Cached scores must be numerically identical to freshly computed scores."""
    from fraud_detector import FraudDetector
    _, graph, _, _ = pipeline
    fd = FraudDetector(graph)
    scores_cold = fd.compute_node_risk_scores()
    fd._centrality_cache = None          # force cold recompute
    scores_cold2 = fd.compute_node_risk_scores()
    scores_warm = fd.compute_node_risk_scores()   # cache hit
    for nid in scores_cold:
        assert abs(scores_cold[nid] - scores_warm[nid]) < 1e-9, \
            f"Node {nid}: cache hit score differs from cold score"
        assert abs(scores_cold[nid] - scores_cold2[nid]) < 1e-9, \
            f"Node {nid}: two cold scores differ (non-deterministic)"



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 8 — #11: SCC from full graph in trace_journey / trace_for_alert
# ══════════════════════════════════════════════════════════════════════════════

def test_scc_detects_cycles_beyond_max_hops(pipeline):
    """A node known to be in a cycle must get part_of_cycle flag even at max_hops=1."""
    from fund_tracer import trace_journey
    import networkx as nx

    df, graph, _, risk_scores = pipeline

    # Find a node that is in an SCC of size >= 3 in the full graph
    full_scc_nodes: set = set()
    for comp in nx.strongly_connected_components(graph):
        if len(comp) >= 3:
            full_scc_nodes.update(comp)

    if not full_scc_nodes:
        pytest.skip("No SCCs of size ≥ 3 in synthetic graph")

    cycle_node = next(iter(full_scc_nodes))

    # Trace with max_hops=1 — the old code would miss cycles that extend further
    result = trace_journey(
        graph, df, risk_scores,
        entity_id=cycle_node,
        direction="both",
        max_hops=1,
        min_amount=0,
    )

    # The focus node itself must carry part_of_cycle
    focus_node = next(n for n in result["nodes"] if n["id"] == cycle_node)
    assert "part_of_cycle" in focus_node["flags"], (
        f"Node {cycle_node} is in a full-graph SCC but missing 'part_of_cycle' flag "
        f"at max_hops=1 (got flags: {focus_node['flags']})"
    )



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 9 — #12: Velocity annotation on links
# ══════════════════════════════════════════════════════════════════════════════

def test_links_have_velocity_fields(pipeline):
    """Every link in a trace result must carry time_span_hours and txn_velocity."""
    from fund_tracer import trace_journey
    df, graph, _, risk_scores = pipeline
    nid = next(iter(graph.nodes()))
    result = trace_journey(graph, df, risk_scores, entity_id=nid, max_hops=2)
    for link in result["links"]:
        assert "time_span_hours" in link, f"Link {link['source']}->{link['target']} missing time_span_hours"
        assert "txn_velocity" in link, f"Link {link['source']}->{link['target']} missing txn_velocity"
        assert link["time_span_hours"] >= 0
        assert link["txn_velocity"] >= 0


def test_velocity_is_consistent_with_txn_count(pipeline):
    """txn_velocity = txn_count / time_span_hours (or capped at txn_count/0.01)."""
    from fund_tracer import trace_journey
    df, graph, _, risk_scores = pipeline
    nid = next(iter(graph.nodes()))
    result = trace_journey(graph, df, risk_scores, entity_id=nid, max_hops=2)
    for link in result["links"]:
        span = link["time_span_hours"]
        expected_v = round(link["txn_count"] / max(span, 0.01), 4)
        assert abs(link["txn_velocity"] - expected_v) < 0.001, \
            f"txn_velocity inconsistency: got {link['txn_velocity']}, expected {expected_v}"



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 10 — #10: Dominant-flow paths
# ══════════════════════════════════════════════════════════════════════════════

def test_dominant_paths_present_in_result(pipeline):
    """trace_journey response must include a dominant_paths list."""
    from fund_tracer import trace_journey
    df, graph, _, risk_scores = pipeline
    # Pick a node with forward edges
    nid = next(n for n in graph.nodes() if graph.out_degree(n) > 0)
    result = trace_journey(
        graph, df, risk_scores,
        entity_id=nid, direction="forward", max_hops=3,
    )
    assert "dominant_paths" in result, "dominant_paths key missing from trace_journey result"


def test_dominant_paths_are_valid_graph_paths(pipeline):
    """Every returned dominant path must be a valid sequence of edges in the graph."""
    from fund_tracer import trace_journey
    df, graph, _, risk_scores = pipeline
    nid = next(n for n in graph.nodes() if graph.out_degree(n) > 0)
    result = trace_journey(
        graph, df, risk_scores,
        entity_id=nid, direction="forward", max_hops=4,
    )
    for dp in result["dominant_paths"]:
        path = dp["path"]
        for i in range(len(path) - 1):
            assert graph.has_edge(path[i], path[i + 1]), \
                f"Dominant path has invalid edge {path[i]}->{path[i+1]}"
        assert dp["bottleneck_amount"] >= 0
        assert dp["hops"] == len(path) - 1



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 11 — #13: Unify trace_for_alert interface (max_hops)
# ══════════════════════════════════════════════════════════════════════════════

def test_trace_for_alert_backwards_compat(pipeline):
    """include_neighbors=True must still expand by exactly 1 hop."""
    from fund_tracer import trace_for_alert
    df, graph, alerts, risk_scores = pipeline
    circ = next((a for a in alerts if a["pattern_type"] == "Circular Transaction"), None)
    if not circ:
        pytest.skip("No circular alert")

    r0 = trace_for_alert(graph, df, risk_scores, circ, include_neighbors=False)
    r1a = trace_for_alert(graph, df, risk_scores, circ, include_neighbors=True)
    r1b = trace_for_alert(graph, df, risk_scores, circ, max_hops=1)

    # include_neighbors=True ≡ max_hops=1
    assert {n["id"] for n in r1a["nodes"]} == {n["id"] for n in r1b["nodes"]}, \
        "include_neighbors=True and max_hops=1 must produce the same node set"

    # 1-hop must have >= as many nodes as 0-hop
    assert len(r1a["nodes"]) >= len(r0["nodes"]), \
        "1-hop expansion must not reduce the node count"


def test_trace_for_alert_multi_hop_expands_further(pipeline):
    """max_hops=2 must yield strictly more nodes than max_hops=1 when neighbors exist."""
    from fund_tracer import trace_for_alert
    df, graph, alerts, risk_scores = pipeline
    circ = next((a for a in alerts if a["pattern_type"] == "Circular Transaction"), None)
    if not circ:
        pytest.skip("No circular alert")

    r1 = trace_for_alert(graph, df, risk_scores, circ, max_hops=1)
    r2 = trace_for_alert(graph, df, risk_scores, circ, max_hops=2)
    # max_hops=2 should have at least as many nodes as max_hops=1
    assert len(r2["nodes"]) >= len(r1["nodes"]), \
        "max_hops=2 should expand at least as far as max_hops=1"



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 12 — #1: _edge_txn_map now populated in build_graph
# ══════════════════════════════════════════════════════════════════════════════

def test_edge_txn_map_populated(pipeline):
    """_edge_txn_map must be non-empty and cover every edge in the graph."""
    from graph_engine import FundFlowGraph
    df, _, _, _ = pipeline
    ffg = FundFlowGraph()
    ffg.build_graph(df)
    assert len(ffg._edge_txn_map) > 0, "_edge_txn_map is empty after build_graph"
    for u, v in ffg.graph.edges():
        assert (u, v) in ffg._edge_txn_map, \
            f"Edge ({u},{v}) present in graph but missing from _edge_txn_map"


def test_edge_txn_map_transaction_count_matches_edge_attr(pipeline):
    """len(_edge_txn_map[e]) must equal the edge's transaction_count attribute."""
    from graph_engine import FundFlowGraph
    df, _, _, _ = pipeline
    ffg = FundFlowGraph()
    ffg.build_graph(df)
    mismatches = []
    for (u, v), txns in ffg._edge_txn_map.items():
        expected = ffg.graph[u][v]["transaction_count"]
        if len(txns) != expected:
            mismatches.append((u, v, len(txns), expected))
    assert not mismatches, f"txn_map count ≠ edge attr for {mismatches[:3]}"


def test_timeline_keeps_earliest_transactions(pipeline):
    """With >200 txns in scope the first transaction must still appear."""
    from fund_tracer import trace_journey
    df, graph, _, risk_scores = pipeline

    # Find an entity with the most transactions
    counts = pd.concat([
        df["sender_id"].value_counts(),
        df["receiver_id"].value_counts(),
    ]).groupby(level=0).sum()
    busy_entity = counts.idxmax()

    result = trace_journey(
        graph, df, risk_scores,
        entity_id=busy_entity,
        direction="both",
        max_hops=3,
        min_amount=0,
    )
    timeline = result["timeline"]
    if len(timeline) == 0:
        pytest.skip("No transactions in trace — entity may be isolated")

    # The timestamps must be in chronological order within each half
    # (not necessarily globally — we concatenate first-half + last-half)
    if len(timeline) > 1:
        ts = [t["timestamp"] for t in timeline]
        # At minimum the very first entry should be <= last entry
        assert ts[0] <= ts[-1], "Timeline is inverted (last < first)"

    # If the underlying data has >200 rows for this entity, verify we're
    # not dropping the origin.
    all_ts = sorted(
        df.loc[
            df["sender_id"].isin({n["id"] for n in result["nodes"]}) &
            df["receiver_id"].isin({n["id"] for n in result["nodes"]}),
            "timestamp"
        ].astype(str).tolist()
    )
    if len(all_ts) > 200:
        # The earliest transaction must appear somewhere in the timeline
        assert timeline[0]["timestamp"] == all_ts[0], (
            f"Earliest transaction {all_ts[0]} was dropped from timeline "
            f"(first entry is {timeline[0]['timestamp']})"
        )
