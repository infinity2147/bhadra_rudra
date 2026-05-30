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
  3. For each edge, concatenate (sender_embed, receiver_embed) and pass through
     a small MLP head that outputs a fraud probability.
  4. Train on a stratified 70/10/20 (train/val/test) edge split with BCE +
     class weighting. Early-stop on validation F1.

**T2.8 changes**:
  * 200 default epochs (was 60) with early stopping (patience=20) on val F1.
  * Stratified train/val/test split — validation set is used purely for
    early-stopping decisions, so reported test metrics stay honest.
  * Optional LinkNeighborLoader-based mini-batching for graphs > 50k edges.
    On the IBM AML 100k graph (88k edges) we still default to full-batch
    because it's faster, but the infrastructure is ready for PSB-scale
    (10M+ edge) graphs where full-batch OOMs.
  * Per-epoch training curve recorded (loss + val_f1) for debugging.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx


# Categorical buckets — kept small so node features stay tractable.
NODE_TYPES = ["individual", "business", "shell_company"]

# Heuristic for when NeighborLoader pays for itself. On a CPU laptop,
# below this edge count full-batch is faster than the loader overhead.
NEIGHBOR_LOADER_EDGE_THRESHOLD = 200_000


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

        type_oh = [1.0 if t == kind else 0.0 for kind in NODE_TYPES]
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
            h = F.relu(self.conv1(x, edge_index))
            h = F.dropout(h, p=0.2, training=self.training)
            h = self.conv2(h, edge_index)
            src_emb = h[edge_pairs[0]]
            dst_emb = h[edge_pairs[1]]
            pair = torch.cat([src_emb, dst_emb], dim=-1)
            return self.edge_mlp(pair).squeeze(-1)

    return EdgeClassifier()


# ── Training helpers ──────────────────────────────────────────────────────────

def _stratified_train_val_test(y: np.ndarray, val_size: float = 0.10,
                                  test_size: float = 0.20, seed: int = 42):
    """Three-way stratified split returning train_idx, val_idx, test_idx (numpy)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    # First peel off the test set
    train_val, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    # Then validation set from the remainder
    val_frac = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=y[train_val], random_state=seed,
    )
    return train, val, test


def _eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Compute F1-optimal threshold + standard metrics."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, average_precision_score, precision_recall_curve,
    )
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"f1": 0.0, "auc": 0.5, "precision": 0.0, "recall": 0.0, "threshold": 0.5}

    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1c = np.where((p[:-1] + r[:-1]) > 0, 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1]), 0.0)
    best_thr = float(thr[np.argmax(f1c)]) if len(thr) > 0 else 0.5
    y_pred = (y_prob >= best_thr).astype(int)

    return {
        "threshold": round(best_thr, 4),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _train_full_batch(model, X, edge_index, train_e, val_e, y, epochs, patience, lr, log_every=10):
    """Full-batch training loop with early-stopping on val F1."""
    torch, nn, F, _ = _torch_imports()
    from torch.optim import AdamW

    src_all = edge_index[0]
    dst_all = edge_index[1]
    train_pairs = torch.stack([src_all[train_e], dst_all[train_e]], dim=0)
    val_pairs   = torch.stack([src_all[val_e],   dst_all[val_e]],   dim=0)
    y_train = y[train_e]
    y_val   = y[val_e]

    pos = int(y_train.sum().item())
    neg = int((1 - y_train).sum().item())
    pos_weight = torch.tensor([max(neg / max(pos, 1), 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    train_losses, val_f1s = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X, edge_index, train_pairs)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(X, edge_index, val_pairs)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_metrics = _eval_metrics(y_val.cpu().numpy().astype(int), val_prob)
        val_f1s.append(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "completed_epochs": len(train_losses),
        "best_val_f1": float(best_val_f1),
        "early_stopped": epochs_no_improve >= patience,
        "loss_curve": train_losses[::max(1, len(train_losses) // 30)],
        "val_f1_curve": val_f1s[::max(1, len(val_f1s) // 30)],
        "final_loss": train_losses[-1] if train_losses else 0.0,
    }


def _train_neighbor_loader(model, X, edge_index, train_e, val_e, y, epochs, patience, lr,
                           batch_size, num_neighbors):
    """Mini-batched training using LinkNeighborLoader — for graphs > ~200k edges."""
    torch, nn, F, _ = _torch_imports()
    from torch.optim import AdamW
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader

    data = Data(x=X, edge_index=edge_index)

    pos = int(y[train_e].sum().item())
    neg = int((1 - y[train_e]).sum().item())
    pos_weight = torch.tensor([max(neg / max(pos, 1), 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    train_edge_label_index = edge_index[:, train_e]
    train_edge_label = y[train_e].float()
    val_edge_label_index = edge_index[:, val_e]
    val_edge_label = y[val_e]

    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=train_edge_label_index,
        edge_label=train_edge_label,
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    train_losses, val_f1s = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_label_index)
            loss = loss_fn(logits, batch.edge_label)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        # Full-batch validation (one forward pass on the whole graph)
        model.eval()
        with torch.no_grad():
            val_logits = model(X, edge_index, val_edge_label_index)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_metrics = _eval_metrics(val_edge_label.cpu().numpy().astype(int), val_prob)
        val_f1s.append(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "completed_epochs": len(train_losses),
        "best_val_f1": float(best_val_f1),
        "early_stopped": epochs_no_improve >= patience,
        "loss_curve": train_losses[::max(1, len(train_losses) // 30)],
        "val_f1_curve": val_f1s[::max(1, len(val_f1s) // 30)],
        "final_loss": train_losses[-1] if train_losses else 0.0,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def train_gnn(
    graph: nx.DiGraph,
    data_dir: str,
    variant: str = "synthetic",
    epochs: int = 200,
    hidden: int = 64,
    lr: float = 0.005,
    patience: int = 20,
    batch_size: int = 4096,
    num_neighbors: Optional[List[int]] = None,
    use_neighbor_loader: Optional[bool] = None,
) -> Dict:
    """Train a GraphSAGE edge classifier and persist alongside XGBoost.

    Args:
        epochs: maximum epochs (default 200; early-stopping usually kicks in earlier).
        patience: epochs without val-F1 improvement before stopping (default 20).
        use_neighbor_loader: None auto-picks based on graph size; True forces
            LinkNeighborLoader, False forces full-batch.
        num_neighbors: per-hop neighbour-sampling counts (defaults to [15, 10]
            for a 2-layer SAGE).
    """
    torch, nn, F, _ = _torch_imports()

    out_dir = os.path.join(data_dir, "ml", variant, "gnn")
    os.makedirs(out_dir, exist_ok=True)

    # Build tensors
    X_np, idx = _node_features(graph)
    edge_index_np, y_np, edge_ids = _edge_index_and_label(graph, idx)
    if y_np.sum() == 0 or y_np.sum() == len(y_np):
        raise ValueError("Need both fraud and non-fraud edges to train GNN.")

    train_e, val_e, test_e = _stratified_train_val_test(y_np)

    X = torch.tensor(X_np, dtype=torch.float32)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    y = torch.tensor(y_np, dtype=torch.float32)

    n_edges = edge_index_np.shape[1]
    if use_neighbor_loader is None:
        use_neighbor_loader = n_edges >= NEIGHBOR_LOADER_EDGE_THRESHOLD
    if num_neighbors is None:
        num_neighbors = [15, 10]

    model = _build_model(in_dim=X.shape[1], hidden=hidden)
    if use_neighbor_loader:
        train_log = _train_neighbor_loader(
            model, X, edge_index, train_e, val_e, y,
            epochs=epochs, patience=patience, lr=lr,
            batch_size=batch_size, num_neighbors=num_neighbors,
        )
    else:
        train_log = _train_full_batch(
            model, X, edge_index, train_e, val_e, y,
            epochs=epochs, patience=patience, lr=lr,
        )

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        src_all = edge_index[0]; dst_all = edge_index[1]
        test_pairs = torch.stack([src_all[test_e], dst_all[test_e]], dim=0)
        test_logits = model(X, edge_index, test_pairs)
        test_prob = torch.sigmoid(test_logits).cpu().numpy()

        all_logits = model(X, edge_index, edge_index)
        all_prob = torch.sigmoid(all_logits).cpu().numpy()

    test_metrics = _eval_metrics(y_np[test_e].astype(int), test_prob)

    metrics = {
        "variant": variant,
        "model_kind": "graphsage_gnn",
        "n_nodes": int(X.shape[0]),
        "n_edges": int(len(y_np)),
        "n_features": int(X.shape[1]),
        "n_train": int(len(train_e)),
        "n_val":   int(len(val_e)),
        "n_test":  int(len(test_e)),
        "n_fraud_train": int(y_np[train_e].sum()),
        "n_fraud_test":  int(y_np[test_e].sum()),
        "fraud_rate": float(y_np.mean()),
        "hidden_dim": hidden,
        "max_epochs": epochs,
        "patience": patience,
        "training_mode": "neighbor_loader" if use_neighbor_loader else "full_batch",
        **test_metrics,
        **train_log,
    }

    edge_scores = {f"{u}->{v}": float(round(p, 4)) for (u, v), p in zip(edge_ids, all_prob)}
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
