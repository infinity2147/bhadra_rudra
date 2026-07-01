"""Train the TGN fraud classifier: chronological, predict-then-update, memory
detached each batch (no BPTT / VRAM blow-up), selected on val AUPRC, with an F2
operating point. Persists metrics + weights + a ranked predicted-fraud list.
Adapted from the canonical PyG TGN example (self-supervised link prediction) to
SUPERVISED fraud classification over real edges.

Correctness invariants preserved:
  1. Predict-then-update: embeddings + logits computed from *current* memory;
     memory.update_state + nbr.insert run AFTER backward/step.
  2. memory.detach() every batch — cuts BPTT, prevents VRAM growth.
  3. No temporal leakage: memory+nbr reset at epoch start; val/test scored with
     memory warmed on train only; model selected on val AUPRC.
  4. F2 operating threshold (recall-favouring, beta=2).
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np


def _metrics(y_true, y_prob) -> Dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from ml_model import fbeta_optimal_threshold

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"auprc": 0.0, "auc": 0.5, "f1": 0.0, "f2": 0.0,
                "precision": 0.0, "recall": 0.0, "threshold": 0.5}
    thr, _ = fbeta_optimal_threshold(y_true, y_prob, beta=2.0)
    pred = (y_prob >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    return {
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4),
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "threshold": round(float(thr), 4),
    }


def train_tgn(
    data_dir: str,
    variant: str = "ibm_aml",
    epochs: int = 30,
    seed: int = 42,
    batch_size: int = 200,
    neighbors: int = 10,
    patience: int = 5,
    top_n: int = 50,
) -> Dict:
    """Train TGN, persist artifacts, return test-set metrics dict.

    Args:
        data_dir:   Root data directory; CSV expected at
                    ``{data_dir}/{variant}/transactions.csv``.
        variant:    Sub-directory name, mirrors other ML pipeline variants.
        epochs:     Maximum training epochs.
        seed:       RNG seed for reproducibility.
        batch_size: Events per temporal batch.
        neighbors:  LastNeighborLoader size (recent-neighbor cap).
        patience:   Early-stopping patience on val AUPRC.
        top_n:      Maximum entries in predictions.json (ranked by prob desc).

    Returns:
        Test-set metrics dict (auprc, auc, f1, f2, precision, recall, threshold,
        plus training metadata).
    """
    import torch
    from torch_geometric.loader import TemporalDataLoader
    from torch_geometric.nn.models.tgn import LastNeighborLoader
    from temporal_data_loader import load_temporal_data
    from tgn_model import build_tgn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.join(data_dir, variant, "transactions.csv")
    bundle = load_temporal_data(csv_path)
    data = bundle["data"].to(device)
    train_data = bundle["train"].to(device)
    val_data = bundle["val"].to(device)
    test_data = bundle["test"].to(device)
    num_nodes = bundle["num_nodes"]
    msg_dim = bundle["msg_dim"]

    memory, gnn, decoder = build_tgn(
        num_nodes, msg_dim, memory_dim=100, time_dim=100, embedding_dim=100
    )
    memory = memory.to(device)
    gnn = gnn.to(device)
    decoder = decoder.to(device)

    # LastNeighborLoader tracks the most-recent `neighbors` interactions per node.
    nbr = LastNeighborLoader(num_nodes, size=neighbors, device=device)

    # assoc maps global node id → local index within the current batch's n_id.
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    opt = torch.optim.Adam(
        list(memory.parameters()) + list(gnn.parameters()) + list(decoder.parameters()),
        lr=1e-4,
    )

    # Class-imbalance: weight positive (fraud) examples by neg/pos ratio on train.
    n_pos = float((train_data.y == 1).sum())
    n_neg = float((train_data.y == 0).sum())
    pos_weight = torch.tensor([max(n_neg, 1.0) / max(n_pos, 1.0)], device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def process(split, train: bool):
        """Stream `split` through the model batch-by-batch.

        Correctness invariant (predict-then-update):
          1. Query memory + neighbor_loader with CURRENT state → embeddings.
          2. Compute logits / loss / backward (training only).
          3. THEN call memory.update_state + nbr.insert.
          4. memory.detach() to cut BPTT.

        This ensures the model never conditions a score on the memory update
        caused by the edge it is scoring — i.e. no information leakage within
        a batch.
        """
        if train:
            memory.train(); gnn.train(); decoder.train()
        else:
            memory.eval(); gnn.eval(); decoder.eval()

        loader = TemporalDataLoader(split, batch_size=batch_size)
        all_y, all_p = [], []

        for batch in loader:
            src = batch.src
            dst = batch.dst
            t = batch.t
            msg = batch.msg
            y = batch.y

            # All unique nodes involved in this batch + their recent neighbors.
            n_id_batch = torch.cat([src, dst]).unique()

            # nbr(n_id) returns (n_id_expanded, edge_index, e_id):
            #   n_id_expanded — node ids including those reached via recent edges
            #   edge_index    — edge index over n_id_expanded (local indices)
            #   e_id          — indices into the GLOBAL data object for edge features
            n_id, edge_index, e_id = nbr(n_id_batch)

            # Map global node ids → local positions in n_id for this batch.
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # ── PREDICT (using CURRENT memory — before this batch's update) ──
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
            logit = decoder(z[assoc[src]], z[assoc[dst]]).view(-1)

            if train:
                loss = crit(logit, y.float())
                opt.zero_grad()
                loss.backward()
                opt.step()

            # ── UPDATE (after prediction + gradient step) ──
            # memory.update_state incorporates this batch's events into memory.
            # nbr.insert records these edges for future neighbor queries.
            # Both happen AFTER scoring — predict-then-update invariant.
            memory.update_state(src, dst, t, msg)
            nbr.insert(src, dst)

            # Cut BPTT: detach memory gradients so next batch's backward
            # doesn't traverse the entire history (would cause VRAM growth).
            memory.detach()

            all_y.append(y.detach().cpu())
            all_p.append(torch.sigmoid(logit).detach().cpu())

        return torch.cat(all_y).numpy(), torch.cat(all_p).numpy()

    # ── Training loop with early stopping on val AUPRC ──
    best_auprc = -1.0
    best_state = None
    bad = 0

    for epoch in range(epochs):
        # Reset at epoch start: no leakage of memory from a previous epoch's
        # val/test pass into the next epoch's training.
        memory.reset_state()
        nbr.reset_state()

        # Train pass: learns weights + warms memory over the train window.
        process(train_data, train=True)

        # Val pass: memory continues from end of train (no reset) — models the
        # real-world condition where the bank keeps its memory state.
        vy, vp = process(val_data, train=False)
        v_auprc = _metrics(vy, vp)["auprc"]

        if v_auprc > best_auprc:
            best_auprc = v_auprc
            bad = 0
            # Clone state dicts to CPU — prefixed by module initial to avoid
            # key collisions between memory/gnn/decoder.
            best_state = {
                **{f"m.{k}": v.detach().cpu().clone()
                   for k, v in memory.state_dict().items()},
                **{f"g.{k}": v.detach().cpu().clone()
                   for k, v in gnn.state_dict().items()},
                **{f"d.{k}": v.detach().cpu().clone()
                   for k, v in decoder.state_dict().items()},
            }
        else:
            bad += 1
            if bad >= patience:
                break

    # ── Final evaluation with best weights ──
    # Restore best-val checkpoint, then re-warm memory on train+val before
    # scoring test — same chronological order as real deployment.
    if best_state is not None:
        memory.load_state_dict(
            {k[2:]: v for k, v in best_state.items() if k.startswith("m.")}
        )
        gnn.load_state_dict(
            {k[2:]: v for k, v in best_state.items() if k.startswith("g.")}
        )
        decoder.load_state_dict(
            {k[2:]: v for k, v in best_state.items() if k.startswith("d.")}
        )

    memory.reset_state()
    nbr.reset_state()
    process(train_data, train=False)   # warm memory on train
    process(val_data, train=False)     # continue through val
    ty, tp = process(test_data, train=False)   # score test

    metrics = _metrics(ty, tp)
    metrics.update({
        "model_kind": "tgn",
        "variant": variant,
        "epochs_run": epoch + 1,
        "n_train": int(train_data.num_events),
        "n_val": int(val_data.num_events),
        "n_test": int(test_data.num_events),
        "val_auprc": round(best_auprc, 4),
    })

    # ── Persist ──
    out_dir = os.path.join(data_dir, "ml", variant, "tgn")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(
        {
            "memory": memory.state_dict(),
            "gnn": gnn.state_dict(),
            "decoder": decoder.state_dict(),
        },
        os.path.join(out_dir, "model.pt"),
    )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Ranked predicted-fraud list over the test window (desc by prob).
    id_to_name = bundle["id_to_name"]
    src_np = test_data.src.cpu().numpy()
    dst_np = test_data.dst.cpu().numpy()
    order = np.argsort(-tp)[:top_n]
    preds = [
        {
            "src_id": int(src_np[i]),
            "dst_id": int(dst_np[i]),
            "src_name": id_to_name.get(int(src_np[i]), str(int(src_np[i]))),
            "dst_name": id_to_name.get(int(dst_np[i]), str(int(dst_np[i]))),
            "prob": round(float(tp[i]), 4),
            "is_fraud": int(ty[i]),
        }
        for i in order
    ]
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump({"variant": variant, "predictions": preds}, f, indent=2)

    return metrics


def load_tgn_metrics(data_dir: str, variant: str = "ibm_aml") -> Dict:
    """Load persisted TGN metrics, or return empty dict if not found."""
    path = os.path.join(data_dir, "ml", variant, "tgn", "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_tgn_predictions(data_dir: str, variant: str = "ibm_aml") -> Dict:
    """Load persisted TGN ranked predictions, or return empty predictions."""
    path = os.path.join(data_dir, "ml", variant, "tgn", "predictions.json")
    if not os.path.exists(path):
        return {"predictions": []}
    with open(path) as f:
        return json.load(f)
