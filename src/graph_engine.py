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
        self._edge_txn_map: Dict = defaultdict(list)

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """Build directed weighted graph from transaction DataFrame."""
        self.graph = nx.DiGraph()
        has_product = "sender_product" in df.columns
        has_channel = "channel" in df.columns

        # Add nodes with attributes (use most common product per entity if present)
        for _, row in df.iterrows():
            sender_attrs = {
                "name": row["sender_name"],
                "type": row["sender_type"],
                "branch": row["sender_branch"],
            }
            if has_product:
                sender_attrs["product"] = row["sender_product"]
            receiver_attrs = {
                "name": row["receiver_name"],
                "type": row["receiver_type"],
                "branch": row["receiver_branch"],
            }
            if has_product:
                receiver_attrs["product"] = row["receiver_product"]
            self.graph.add_node(row["sender_id"], **sender_attrs)
            self.graph.add_node(row["receiver_id"], **receiver_attrs)

        # Add edges with aggregated weights
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

        # Channel/rail/product mix per edge (computed separately to keep groupby simple)
        rail_mix = {}
        channel_mix = {}
        if has_channel:
            for (s, r), sub in df.groupby(["sender_id", "receiver_id"]):
                rail_mix[(s, r)] = sub["transaction_type"].value_counts().to_dict()
                channel_mix[(s, r)] = sub["channel"].value_counts().to_dict()
        else:
            for (s, r), sub in df.groupby(["sender_id", "receiver_id"]):
                rail_mix[(s, r)] = sub["transaction_type"].value_counts().to_dict()

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
        """Compute node-level features for fraud detection."""
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
        """Extract a subgraph around specified nodes."""
        nodes_to_include = set(node_ids)
        for _ in range(hops):
            new_nodes = set()
            for node in nodes_to_include:
                new_nodes.update(self.graph.predecessors(node))
                new_nodes.update(self.graph.successors(node))
            nodes_to_include.update(new_nodes)

        return self.graph.subgraph(nodes_to_include).copy()

    def get_graph_stats(self) -> Dict:
        """Return overall graph statistics."""
        if not self.graph.nodes():
            return {}

        node_features = self.get_node_features()
        degrees = [d for _, d in self.graph.degree()]
        strengths = [f["turnover"] for f in node_features.values()]

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

        for node in G.nodes():
            node_data = dict(G.nodes[node])
            node_features = self.get_node_features().get(node, {})
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
