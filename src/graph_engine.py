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


class FundFlowGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_attributes: Dict = {}
        # Maps (sender_id, receiver_id) → list of transaction dicts for that edge.
        # Populated during build_graph so callers can query raw per-edge transactions
        # (e.g. for velocity burst detection) without a full DataFrame scan.
        self._edge_txn_map: Dict = defaultdict(list)
        self._node_features_cache: Optional[Dict] = None

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """Build directed weighted graph from transaction DataFrame."""
        self.graph = nx.DiGraph()
        self._node_features_cache = None   # invalidate feature cache on rebuild
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
        """Compute edge-level features."""
        features = []
        for u, v, data in self.graph.edges(data=True):
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
