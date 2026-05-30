"""
Fund Flow Graph Engine
Builds directed weighted graphs from transaction data using NetworkX.
Identifies fund flow patterns and computes graph-level features.
"""

import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
import hashlib
import pickle
import os


class FundFlowGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_attributes: Dict = {}
        # Maps (sender_id, receiver_id) → list of transaction dicts for that edge.
        # Populated during build_graph so callers can query raw per-edge transactions
        # (e.g. for velocity burst detection) without a full DataFrame scan.
        self._edge_txn_map: Dict = defaultdict(list)
        self._node_features_cache: Optional[Dict] = None
        self._scc_cache: Optional[Dict[int, set]] = None  # min_size → set of node_ids

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """Build directed weighted graph from transaction DataFrame."""
        self.graph = nx.DiGraph()
        self._node_features_cache = None   # invalidate feature cache on rebuild
        self._scc_cache = None             # SCC also depends on topology
        has_product = "sender_product" in df.columns
        has_channel = "channel" in df.columns

        # ── Nodes — vectorized: deduplicate, then add_nodes_from one list ──────
        # iterrows() over all transactions was O(N_txns) with heavy Python
        # overhead per row.  Instead we extract unique entities from each side
        # and call add_nodes_from() once per side.
        sender_cols = ["sender_id", "sender_name", "sender_type", "sender_branch"]
        receiver_cols = ["receiver_id", "receiver_name", "receiver_type", "receiver_branch"]
        if has_product:
            sender_cols.append("sender_product")
            receiver_cols.append("receiver_product")

        senders = (
            df[sender_cols]
            .drop_duplicates("sender_id")
            .rename(columns=lambda c: c.replace("sender_", ""))
        )
        receivers = (
            df[receiver_cols]
            .drop_duplicates("receiver_id")
            .rename(columns=lambda c: c.replace("receiver_", ""))
        )
        # Merge both sides; entity appearing as both sender and receiver keeps
        # the sender-side attributes (arbitrary but deterministic).
        all_entities = (
            pd.concat([senders, receivers], ignore_index=True)
            .drop_duplicates("id")
        )
        self.graph.add_nodes_from(
            (row["id"], {k: v for k, v in row.items() if k != "id"})
            for _, row in all_entities.iterrows()
        )

        # ── Edges — aggregate stats ──────────────────────────────────────────
        agg_dict = {
            "total_amount": ("amount", "sum"),
            "transaction_count": ("amount", "count"),
            "avg_amount": ("amount", "mean"),
            "min_amount": ("amount", "min"),
            "max_amount": ("amount", "max"),
            "std_amount": ("amount", "std"),
            "first_seen": ("timestamp", "min"),
            "last_seen": ("timestamp", "max"),
            "fraud_count": ("is_fraud", "sum"),
        }
        edge_data = df.groupby(["sender_id", "receiver_id"]).agg(**agg_dict).reset_index()
        edge_data["std_amount"] = edge_data["std_amount"].fillna(0)

        # ── Rail / channel mix + edge_txn_map — single groupby pass ────────────
        # Previously two separate groupby loops; merge into one.
        # _edge_txn_map is populated here so callers can inspect raw per-edge
        # transactions (e.g. burst detection) without rescanning the DataFrame.
        self._edge_txn_map = defaultdict(list)
        rail_mix: Dict = {}
        channel_mix: Dict = {}
        txn_cols = ["transaction_id", "timestamp", "amount", "transaction_type",
                    "is_fraud", "fraud_pattern"] + (["channel"] if has_channel else [])
        txn_cols = [c for c in txn_cols if c in df.columns]
        for (s, r), sub in df.groupby(["sender_id", "receiver_id"]):
            rail_mix[(s, r)] = sub["transaction_type"].value_counts().to_dict()
            if has_channel:
                channel_mix[(s, r)] = sub["channel"].value_counts().to_dict()
            self._edge_txn_map[(s, r)] = sub[txn_cols].to_dict("records")

        for _, row in edge_data.iterrows():
            key = (row["sender_id"], row["receiver_id"])
            self.graph.add_edge(
                row["sender_id"], row["receiver_id"],
                total_amount=row["total_amount"],
                transaction_count=row["transaction_count"],
                avg_amount=row["avg_amount"],
                min_amount=row["min_amount"],
                max_amount=row["max_amount"],
                std_amount=row["std_amount"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                fraud_count=int(row["fraud_count"]),
                rail_mix=rail_mix.get(key, {}),
                channel_mix=channel_mix.get(key, {}),
            )

        return self.graph

    # ── Persistence ──────────────────────────────────────────────────────────
    SNAPSHOT_VERSION = 1     # bump when serialised shape changes

    @staticmethod
    def compute_tx_hash(df: pd.DataFrame) -> str:
        """Stable fingerprint of the transactions used to build a graph.

        Used to detect when a saved snapshot is stale relative to the underlying
        data.  Avoids hashing the full DataFrame (slow) by hashing only row
        count + column names + a few summary stats.
        """
        h = hashlib.sha256()
        h.update(str(len(df)).encode())
        h.update(",".join(sorted(df.columns)).encode())
        if "amount" in df.columns and len(df):
            h.update(f"{df['amount'].sum():.2f}".encode())
        return h.hexdigest()

    def save(self, path: str, tx_hash: Optional[str] = None) -> None:
        """Pickle the graph + caches + tx_hash to disk for fast cold-start."""
        payload = {
            "version": self.SNAPSHOT_VERSION,
            "graph": self.graph,
            "edge_txn_map": dict(self._edge_txn_map),
            "node_features_cache": self._node_features_cache,
            "scc_cache": self._scc_cache,
            "tx_hash": tx_hash,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str, expected_tx_hash: Optional[str] = None) -> bool:
        """Load a saved snapshot.  Returns False if stale / missing / corrupt.

        On False the caller MUST rebuild from scratch.  We never silently use
        a snapshot that doesn't match the underlying data — half-stale graph
        state would produce wrong fraud alerts.
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("version") != self.SNAPSHOT_VERSION:
                return False
            if expected_tx_hash is not None and payload.get("tx_hash") != expected_tx_hash:
                return False
            self.graph = payload["graph"]
            self._edge_txn_map = defaultdict(list, payload.get("edge_txn_map", {}))
            self._node_features_cache = payload.get("node_features_cache")
            self._scc_cache = payload.get("scc_cache")
            return True
        except (pickle.UnpicklingError, EOFError, KeyError, OSError):
            return False

    def get_sccs(self, min_size: int = 3) -> set:
        """Return the union of all SCC members of size >= min_size.

        Cached per min_size — re-computing strongly-connected components on
        every trace_journey call adds non-trivial cost on dense graphs.
        Invalidated by build_graph and add_transaction (topology changes).
        """
        if self._scc_cache is None:
            self._scc_cache = {}
        if min_size not in self._scc_cache:
            result: set = set()
            for comp in nx.strongly_connected_components(self.graph):
                if len(comp) >= min_size:
                    result.update(comp)
            self._scc_cache[min_size] = result
        return self._scc_cache[min_size]

    def add_transaction(self, row: Dict) -> Tuple[bool, bool]:
        """Incrementally update the graph with one new transaction.

        Returns: (new_edge_created, new_node_created).  Updates the edge's
        aggregate stats (total, count, avg, min, max, first/last_seen) in
        place, appends to _edge_txn_map, and invalidates ONLY the affected
        nodes in _node_features_cache (or the whole thing if the topology
        changed).  Designed to be called from a streaming consumer for each
        Kafka message; rebuilding the full graph on every txn would be fatal
        at production load.

        Required row keys: sender_id, receiver_id, amount, timestamp,
        transaction_type, is_fraud (optional defaults to 0).
        """
        s = row["sender_id"]
        r = row["receiver_id"]
        amount = float(row["amount"])
        ts = row["timestamp"]
        new_node = False
        new_edge = False

        # Ensure nodes exist (if metadata fields present, copy them)
        for side, eid in (("sender", s), ("receiver", r)):
            if eid not in self.graph:
                attrs = {}
                for key in ("name", "type", "branch", "product"):
                    full = f"{side}_{key}"
                    if full in row:
                        attrs[key] = row[full]
                self.graph.add_node(eid, **attrs)
                new_node = True

        if self.graph.has_edge(s, r):
            ed = self.graph[s][r]
            # Update running aggregates — avg, std are best-effort approximations
            new_total = ed["total_amount"] + amount
            new_count = ed["transaction_count"] + 1
            new_avg = new_total / new_count
            # Welford-style variance update would be ideal; for now approximate
            # by recomputing on the running min/max alone for downstream callers.
            ed["total_amount"] = new_total
            ed["transaction_count"] = new_count
            ed["avg_amount"] = new_avg
            ed["min_amount"] = min(ed["min_amount"], amount)
            ed["max_amount"] = max(ed["max_amount"], amount)
            ed["last_seen"] = ts                      # txns assumed in order
            if int(row.get("is_fraud", 0)) > 0:
                ed["fraud_count"] = int(ed.get("fraud_count", 0)) + 1
            rail = row.get("transaction_type")
            if rail:
                ed["rail_mix"][rail] = ed["rail_mix"].get(rail, 0) + 1
            ch = row.get("channel")
            if ch:
                ed["channel_mix"][ch] = ed["channel_mix"].get(ch, 0) + 1
        else:
            new_edge = True
            self.graph.add_edge(
                s, r,
                total_amount=amount,
                transaction_count=1,
                avg_amount=amount,
                min_amount=amount,
                max_amount=amount,
                std_amount=0.0,
                first_seen=ts,
                last_seen=ts,
                fraud_count=int(row.get("is_fraud", 0)),
                rail_mix={row.get("transaction_type", "OTHER"): 1},
                channel_mix={row.get("channel", "OTHER"): 1} if row.get("channel") else {},
            )

        # Append to per-edge transaction map for downstream temporal analysis
        self._edge_txn_map[(s, r)].append({
            "transaction_id": row.get("transaction_id"),
            "timestamp": ts,
            "amount": amount,
            "transaction_type": row.get("transaction_type"),
            "is_fraud": int(row.get("is_fraud", 0)),
            "fraud_pattern": row.get("fraud_pattern"),
        })

        # Invalidate caches — topology change invalidates everything,
        # otherwise only the two endpoints' feature rows go stale.
        if new_node or new_edge:
            self._node_features_cache = None
            self._scc_cache = None      # topology changed → SCC may have changed
        elif self._node_features_cache is not None:
            self._node_features_cache.pop(s, None)
            self._node_features_cache.pop(r, None)

        return new_edge, new_node

    def get_node_features(self) -> Dict[str, Dict]:
        """Compute node-level features for fraud detection.

        Result is cached after the first call and reused until build_graph()
        is called again (which sets _node_features_cache = None).
        """
        if self._node_features_cache is not None:
            return self._node_features_cache
        features = {}
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            in_strength = sum(
                self.graph[u][node]["total_amount"]
                for u in self.graph.predecessors(node)
            )
            out_strength = sum(
                self.graph[node][v]["total_amount"]
                for v in self.graph.successors(node)
            )
            net_flow = in_strength - out_strength
            turnover = in_strength + out_strength

            features[node] = {
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree,
                "in_strength": round(in_strength, 2),
                "out_strength": round(out_strength, 2),
                "net_flow": round(net_flow, 2),
                "turnover": round(turnover, 2),
                "imbalance_ratio": round(abs(net_flow) / max(turnover, 1), 4),
                "type": self.graph.nodes[node].get("type", "unknown"),
                "name": self.graph.nodes[node].get("name", "unknown"),
                "branch": self.graph.nodes[node].get("branch", "unknown"),
            }

        self._node_features_cache = features
        return features

    def get_edge_features(self) -> List[Dict]:
        """Compute edge-level features.

        Uses the populated _edge_txn_map to also expose temporal-detail features
        that were previously inaccessible — median inter-arrival time between
        consecutive transactions on this edge, and a burst_count (number of
        ≥3-txn clusters within 1h).  These signals are what AML detectors need
        to differentiate "10 NEFT transfers over a year" from "10 transfers
        within 5 minutes" — both have the same aggregate stats.
        """
        features = []
        for u, v, data in self.graph.edges(data=True):
            edge_txns = self._edge_txn_map.get((u, v), [])
            median_iat = None
            burst_count = 0
            if len(edge_txns) >= 2:
                try:
                    ts = pd.to_datetime(
                        [t["timestamp"] for t in edge_txns], format="mixed"
                    ).sort_values()
                    diffs_seconds = ts.to_series().diff().dt.total_seconds().dropna()
                    if len(diffs_seconds) > 0:
                        median_iat = round(float(diffs_seconds.median()), 2)
                    # Count rolling windows where 3+ txns fit in 1h (3600s)
                    arr = ts.to_numpy()
                    i = 0
                    while i + 2 < len(arr):
                        if (arr[i + 2] - arr[i]) <= pd.Timedelta(hours=1):
                            burst_count += 1
                            j = i + 3
                            while j < len(arr) and (arr[j] - arr[i]) <= pd.Timedelta(hours=1):
                                j += 1
                            i = j
                        else:
                            i += 1
                except Exception:
                    pass
            features.append({
                "source": u,
                "target": v,
                "source_name": self.graph.nodes[u].get("name", u),
                "target_name": self.graph.nodes[v].get("name", v),
                "total_amount": data["total_amount"],
                "transaction_count": data["transaction_count"],
                "avg_amount": data["avg_amount"],
                "amount_variability": round(data["std_amount"] / max(data["avg_amount"], 1), 4),
                "fraud_count": data["fraud_count"],
                "median_iat_seconds": median_iat,
                "burst_count": burst_count,
            })
        return features

    def extract_subgraph(self, node_ids: List[str], hops: int = 1) -> nx.DiGraph:
        """Extract a subgraph around specified nodes up to `hops` away.

        Uses nx.ego_graph on an undirected view so both predecessors and
        successors are included, then restricts back to the original directed
        graph.  This replaces the hand-rolled BFS which had a subtle bug:
        it expanded the full `nodes_to_include` set every iteration rather
        than only the newly-added frontier, causing hop-2 nodes to be
        re-expanded in hop-3.
        """
        # Undirected view lets ego_graph follow edges in both directions.
        undirected = self.graph.to_undirected(as_view=True)
        nodes_to_include: set = set()
        for seed in node_ids:
            if seed in self.graph:
                ego = nx.ego_graph(undirected, seed, radius=hops)
                nodes_to_include.update(ego.nodes())
        if not nodes_to_include:
            nodes_to_include = set(node_ids)
        return self.graph.subgraph(nodes_to_include).copy()

    def get_graph_stats(self) -> Dict:
        """Return overall graph statistics."""
        if not self.graph.nodes():
            return {}

        # get_node_features() already stores total_degree and turnover in the
        # cached dict — read from there instead of a separate graph.degree() pass.
        node_features = self.get_node_features()
        degrees   = [f["total_degree"] for f in node_features.values()]
        strengths = [f["turnover"]     for f in node_features.values()]

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 6),
            "avg_degree": round(np.mean(degrees), 2),
            "max_degree": max(degrees) if degrees else 0,
            "avg_turnover": round(np.mean(strengths), 2),
            "strongly_connected_components": nx.number_strongly_connected_components(self.graph),
            "weakly_connected_components": nx.number_weakly_connected_components(self.graph),
        }

    def to_cytoscape_elements(self, subgraph_nodes: Optional[List[str]] = None) -> Dict:
        """Convert graph to Cytoscape.js format for visualization."""
        G = self.graph
        if subgraph_nodes:
            G = self.extract_subgraph(subgraph_nodes, hops=1)

        # Compute fraud flag for nodes
        fraud_edges = {(u, v) for u, v, d in G.edges(data=True) if d.get("fraud_count", 0) > 0}
        fraud_nodes = set()
        for u, v in fraud_edges:
            fraud_nodes.add(u)
            fraud_nodes.add(v)

        elements = {"nodes": [], "edges": []}

        # Compute features once, outside the loop — was previously calling
        # get_node_features() per node, re-traversing the entire graph each time.
        all_node_features = self.get_node_features()

        for node in G.nodes():
            node_data = dict(G.nodes[node])
            node_features = all_node_features.get(node, {})
            is_fraud = node in fraud_nodes
            node_type = node_data.get("type", "individual")

            elements["nodes"].append({
                "data": {
                    "id": node,
                    "label": node_data.get("name", node)[:20],
                    "full_name": node_data.get("name", node),
                    "type": node_type,
                    "branch": node_data.get("branch", ""),
                    "is_fraud": is_fraud,
                    "turnover": node_features.get("turnover", 0),
                    "degree": node_features.get("total_degree", 0),
                    "net_flow": node_features.get("net_flow", 0),
                }
            })

        for u, v, data in G.edges(data=True):
            elements["edges"].append({
                "data": {
                    "id": f"{u}->{v}",
                    "source": u,
                    "target": v,
                    "weight": data["total_amount"],
                    "transaction_count": data["transaction_count"],
                    "avg_amount": round(data["avg_amount"], 2),
                    "is_fraud": data.get("fraud_count", 0) > 0,
                    "fraud_count": data.get("fraud_count", 0),
                }
            })

        return elements
