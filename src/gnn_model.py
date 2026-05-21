"""
GraphSAGE GNN baseline for fund-flow fraud detection.

Why a GNN at all: the XGBoost baseline scores each edge from per-edge features
plus its endpoint stats — it can't see beyond two hops. A GNN learns
representations by aggregating information from each node's neighbourhood,
which is the right inductive bias for AML where fraud patterns (rings,
funnels, layering chains) are inherently multi-hop topology.

We train at the *edge* level for a fair head-to-head against XGBoost:

  1. Build a per-node feature vector (degree, log-strength, type one-hots,
     branch one-hot bucketing) from the fund-flow graph.
  2. Run two GraphSAGE convolutions to produce hop-aggregated node embeddings.
  3. For each edge, concatenate (sender_embed, receiver_embed, edge_features)
     and pass through a small MLP head that outputs a fraud probability.
  4. Train on a stratified 80/20 edge split with BCE loss + class weighting.

The model is intentionally small (hidden dim 64, two layers) so it trains in
seconds on CPU. We persist it alongside XGBoost under data/ml/{variant}/gnn/.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

# torch / torch_geometric imports are deferred so the rest of the pipeline
# still works if these packages aren't installed.


# Categorical buckets — kept small so node features stay tractable.
NODE_TYPES = ["individual", "business", "shell_company"]


def _torch_imports():
    """Lazy import + nice error if torch_geometric isn't installed."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
        return torch, nn, F, SAGEConv
    except ImportError as e:
        raise ImportError(
            "PyTorch Geometric is not installed.\n"
            "Install with:\n"
            "    pip install torch torch-geometric\n\n"
            f"Original error: {e}"
        )


# ── Feature builders ──────────────────────────────────────────────────────────

def _node_features(graph: nx.DiGraph) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build a [num_nodes, F] node-feature matrix and an id -> index map."""
    node_ids = list(graph.nodes())
    idx = {nid: i for i, nid in enumerate(node_ids)}

    # Branch bucketing — we hash to 8 columns. Small enough to keep the model
    # tiny while still letting the GNN learn that some branches are riskier.
    branch_buckets = 8

    rows = []
    for nid in node_ids:
        nd = graph.nodes[nid]
        t = nd.get("type", "individual")
        branch = nd.get("branch", "")
        in_deg = graph.in_degree(nid)
        out_deg = graph.out_degree(nid)
        in_str = sum(graph[u][nid]["total_amount"] for u in graph.predecessors(nid))
        out_str = sum(graph[nid][v]["total_amount"] for v in graph.successors(nid))

        # Type one-hot
        type_oh = [1.0 if t == kind else 0.0 for kind in NODE_TYPES]
        # Branch bucket one-hot (hashed)
        branch_oh = [0.0] * branch_buckets
        if branch:
            branch_oh[hash(branch) % branch_buckets] = 1.0

        row = type_oh + branch_oh + [
            float(in_deg),
            float(out_deg),
            float(np.log1p(in_str)),
            float(np.log1p(out_str)),
        ]
        rows.append(row)

    return np.array(rows, dtype=np.float32), idx


def _edge_index_and_label(graph: nx.DiGraph, idx: Dict[str, int]):
    """Build edge_index [2, E] (for message passing) and edge labels [E]."""
    src, dst = [], []
    labels = []
    edge_ids: List[Tuple[str, str]] = []
    for u, v, ed in graph.edges(data=True):
        if u not in idx or v not in idx:
            continue
        src.append(idx[u])
        dst.append(idx[v])
        labels.append(1 if ed.get("fraud_count", 0) > 0 else 0)
        edge_ids.append((u, v))
    edge_index = np.array([src, dst], dtype=np.int64)
    return edge_index, np.array(labels, dtype=np.int64), edge_ids


# ── Model definition ──────────────────────────────────────────────────────────

def _build_model(in_dim: int, hidden: int = 64):
    torch, nn, F, SAGEConv = _torch_imports()

    class EdgeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, 1),
            )

        def forward(self, x, edge_index, edge_pairs):
            # Two-layer GraphSAGE
            h = F.relu(self.conv1(x, edge_index))
            h = F.dropout(h, p=0.2, training=self.training)
            h = self.conv2(h, edge_index)
            # Edge classification: concat sender+receiver embeddings
            src_emb = h[edge_pairs[0]]
            dst_emb = h[edge_pairs[1]]
            pair = torch.cat([src_emb, dst_emb], dim=-1)
            return self.edge_mlp(pair).squeeze(-1)

    return EdgeClassifier()


# ── Public API ────────────────────────────────────────────────────────────────

def train_gnn(
    graph: nx.DiGraph,
    data_dir: str,
    variant: str = "synthetic",
    epochs: int = 60,
    hidden: int = 64,
    lr: float = 0.005,
) -> Dict:
    """Train a GraphSAGE edge classifier and persist alongside XGBoost.

    Returns the metrics dict. Skips gracefully (raising ImportError) if
    PyTorch Geometric isn't installed — the caller can catch and continue.
    """
    torch, nn, F, _ = _torch_imports()
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, average_precision_score,
    )

    out_dir = os.path.join(data_dir, "ml", variant, "gnn")
    os.makedirs(out_dir, exist_ok=True)

    # Build tensors
    X_np, idx = _node_features(graph)
    edge_index_np, y_np, edge_ids = _edge_index_and_label(graph, idx)
    if y_np.sum() == 0 or y_np.sum() == len(y_np):
        raise ValueError("Need both fraud and non-fraud edges to train GNN.")

    # Stratified edge split (we still pass the FULL edge_index for message passing —
    # that's the whole graph; we only differ in which edges we *score* during train/eval)
    edge_idx_np = np.arange(len(y_np))
    train_e, test_e = train_test_split(
        edge_idx_np, test_size=0.2, stratify=y_np, random_state=42,
    )

    X = torch.tensor(X_np, dtype=torch.float32)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    y = torch.tensor(y_np, dtype=torch.float32)

    src_all = edge_index[0]
    dst_all = edge_index[1]

    train_pairs = torch.stack([src_all[train_e], dst_all[train_e]], dim=0)
    test_pairs = torch.stack([src_all[test_e], dst_all[test_e]], dim=0)
    y_train = y[train_e]
    y_test = y[test_e]

    # Class-balanced positive weight
    pos = int(y_train.sum().item())
    neg = int((1 - y_train).sum().item())
    pos_weight = torch.tensor([max(neg / max(pos, 1), 1.0)], dtype=torch.float32)

    model = _build_model(in_dim=X.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X, edge_index, train_pairs)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_logits = model(X, edge_index, test_pairs)
        test_prob = torch.sigmoid(test_logits).numpy()
        test_pred = (test_prob >= 0.5).astype(int)
        y_test_np = y_test.numpy().astype(int)

        all_logits = model(X, edge_index, edge_index)  # score every edge
        all_prob = torch.sigmoid(all_logits).numpy()

    cm = confusion_matrix(y_test_np, test_pred)
    metrics = {
        "variant": variant,
        "model_kind": "graphsage_gnn",
        "n_nodes": int(X.shape[0]),
        "n_edges": int(len(y_np)),
        "n_features": int(X.shape[1]),
        "n_train": int(len(train_e)),
        "n_test": int(len(test_e)),
        "n_fraud_train": int(y_train.sum().item()),
        "n_fraud_test": int(y_test.sum().item()),
        "precision": float(precision_score(y_test_np, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test_np, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test_np, test_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test_np, test_prob)),
        "average_precision": float(average_precision_score(y_test_np, test_prob)),
        "confusion_matrix": cm.tolist(),
        "fraud_rate": float(y_np.mean()),
        "epochs": epochs,
        "hidden_dim": hidden,
        "final_loss": train_losses[-1],
        "loss_curve": train_losses[::max(1, epochs // 30)],  # downsample for UI
    }

    # Persist
    edge_scores = {
        f"{u}->{v}": float(round(p, 4))
        for (u, v), p in zip(edge_ids, all_prob)
    }
    torch.save(model.state_dict(), os.path.join(out_dir, "gnn_weights.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(out_dir, "edge_scores.json"), "w") as f:
        json.dump(edge_scores, f, indent=2)

    return metrics


def load_gnn_metrics(data_dir: str, variant: str = "synthetic") -> Dict:
    path = os.path.join(data_dir, "ml", variant, "gnn", "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_gnn_edge_scores(data_dir: str, variant: str = "synthetic") -> Dict[str, float]:
    path = os.path.join(data_dir, "ml", variant, "gnn", "edge_scores.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
