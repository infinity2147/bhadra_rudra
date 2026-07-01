# Temporal GNN (TGN) Fraud Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Temporal Graph Network (TGN, Rossi et al. 2020) that classifies transaction edges as fraud using a continuous-time per-node memory + temporal attention, trained/evaluated on a strict chronological split, and surfaced alongside the existing static models.

**Architecture:** `temporal_data_loader` builds a PyG `TemporalData` (chronological 70/15/15) → `tgn_model` defines the TGN memory/attention/decoder modules → `train_tgn` runs the predict-then-update loop with `memory.detach()`, selects on val AUPRC, applies an F2 threshold, and persists `metrics.json`/`model.pt`/`predictions.json` → `run_pipeline` trains it after XGBoost → the backend serves the persisted JSON → ModelMetrics shows a TGN column + a predicted-fraud list.

**Tech Stack:** PyTorch 2.11 + PyTorch Geometric 2.7 (`TGNMemory`, `LastNeighborLoader`, `TransformerConv`, `TimeEncoder`, `TemporalDataLoader`); pandas/scikit-learn; FastAPI; React + Vite.

## Global Constraints

- **Objective is fraud classification, NOT link existence.** Every real edge is an example, label = `is_fraud`; imbalance handled by `BCEWithLogitsLoss(pos_weight)`. No random-pair negative sampling.
- **Strict chronological split, no leakage:** `train.t.max() <= val.t.min() <= test.t.min()`. Memory is only ever updated by past events (reset per epoch; warm on train before scoring val/test).
- **Selection on val AUPRC**; operating point via `ml_model.fbeta_optimal_threshold(y_true, y_prob, beta=2.0)` (shared with XGB/SAGE/ensemble).
- **XGB-before-torch:** TGN trains in `run_pipeline` only after XGBoost; **every torch-importing test runs in an isolated subprocess** (mirror `tests/test_gnn_model.py`).
- **Optional dependency:** wrap torch/PyG imports so absence → graceful skip.
- **Serve from persisted JSON; never load `.pt` at runtime.** No live interactive prediction this round.
- Artifacts live in `data/ml/{variant}/tgn/` (mirrors `.../gnn/`). Seed everything (`seed=42`).
- Commit after each task. Co-author trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Local commits only — do NOT push.

## File Structure

- `src/temporal_data_loader.py` — **create**: CSV → sorted `TemporalData` + chronological split + id map.
- `src/tgn_model.py` — **create**: `GraphAttentionEmbedding`, `FraudDecoder`, `build_tgn(...)` factory.
- `src/train_tgn.py` — **create**: training/eval loop + persistence + `load_tgn_metrics`/`load_tgn_predictions`.
- `src/run_pipeline.py` — **modify**: train TGN after XGBoost (optional, try/except).
- `backend/main.py` — **modify**: add `tgn` to `/api/ml/metrics`; add `GET /api/tgn/predictions`.
- `frontend/src/api.js` + `frontend/src/pages/ModelMetrics.jsx` — **modify**: TGN column + predicted-fraud list.
- `tests/test_temporal_data_loader.py`, `tests/test_tgn_model.py` — **create** (subprocess-isolated).
- `tests/test_api_integration.py` — **modify**: TGN endpoint test.

---

### Task 1: Temporal data loader

**Files:**
- Create: `src/temporal_data_loader.py`
- Test: `tests/test_temporal_data_loader.py`

**Interfaces:**
- Produces: `load_temporal_data(csv_path) -> dict` with keys `data` (`TemporalData`), `train`/`val`/`test` (`TemporalData` splits), `num_nodes` (int), `msg_dim` (int), `id_to_name` (dict[int,str]).

- [ ] **Step 1: Write the failing test** (subprocess-isolated — torch must stay out of the main pytest process)

```python
# tests/test_temporal_data_loader.py
"""Chronological-split correctness for the TGN temporal loader.

Runs in an ISOLATED SUBPROCESS: torch/PyG cannot share a process with xgboost
on this platform (xgboost must import first or it segfaults), so — like
tests/test_gnn_model.py — we shell out to keep torch out of the main pytest run.
"""
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def _run(body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r})
        try:
            import torch  # noqa
            from torch_geometric.data import TemporalData  # noqa
        except Exception:
            print("SKIP"); sys.exit(0)
        import pandas as pd, tempfile, os
        from temporal_data_loader import load_temporal_data
        # tiny deterministic CSV, timestamps intentionally OUT of order
        rows = []
        ts = ["2022-01-03 00:00:00", "2022-01-01 00:00:00", "2022-01-02 00:00:00"]
        for i in range(30):
            rows.append(dict(
                transaction_id=f"T{{i}}", timestamp=ts[i % 3],
                sender_id=f"A{{i%5}}", receiver_id=f"B{{i%7}}",
                amount=100.0 + i, transaction_type="NEFT",
                is_fraud=int(i % 4 == 0),
            ))
        df = pd.DataFrame(rows)
        p = os.path.join(tempfile.mkdtemp(), "transactions.csv")
        df.to_csv(p, index=False)
        b = load_temporal_data(p)
    """) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def _ok(r):
    if "SKIP" in r.stdout:
        pytest.skip("torch/PyG not installed")
    assert r.returncode == 0, r.stderr
    return r


def test_split_is_chronological_and_shaped():
    r = _ok(_run("""
        d, tr, va, te = b["data"], b["train"], b["val"], b["test"]
        # strictly sorted event times
        assert bool((d.t[1:] >= d.t[:-1]).all()), "events not time-sorted"
        # chronological split, no leakage
        assert int(tr.t.max()) <= int(va.t.min()) <= int(te.t.min())
        # labels are 0/1 floats; msg present
        assert set(d.y.unique().tolist()) <= {0.0, 1.0}
        assert b["msg_dim"] == d.msg.size(-1) and b["msg_dim"] >= 2
        assert b["num_nodes"] >= 1
        print("OK")
    """))
    assert "OK" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_temporal_data_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'temporal_data_loader'` (surfaced as a non-zero subprocess return).

- [ ] **Step 3: Implement `src/temporal_data_loader.py`**

```python
"""Build a PyG TemporalData for the TGN fraud model.

Sorts transactions strictly by time, maps account ids to contiguous integers,
packs edge features into `msg`, sets `y = is_fraud`, and splits chronologically
(70/15/15) with NO shuffling — the split boundary is a point in time, so the
model is never trained on the future. Torch/PyG are imported lazily so importing
this module doesn't pull torch into a process that hasn't trained XGBoost yet.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

_TXN_TYPES = ["NEFT", "RTGS", "IMPS", "UPI", "Cash", "Cheque", "Wire", "Other"]


def _txn_type_onehot(series: pd.Series) -> np.ndarray:
    idx = {t: i for i, t in enumerate(_TXN_TYPES)}
    out = np.zeros((len(series), len(_TXN_TYPES)), dtype=np.float32)
    for row, val in enumerate(series.fillna("Other")):
        out[row, idx.get(str(val), idx["Other"])] = 1.0
    return out


def load_temporal_data(csv_path: str) -> Dict:
    import torch
    from torch_geometric.data import TemporalData

    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # contiguous node ids across both endpoints
    nodes = pd.Index(pd.unique(pd.concat([df["sender_id"], df["receiver_id"]])))
    node_id = {n: i for i, n in enumerate(nodes)}
    src = df["sender_id"].map(node_id).to_numpy(np.int64)
    dst = df["receiver_id"].map(node_id).to_numpy(np.int64)

    t0 = pd.to_datetime(df["timestamp"])
    t = ((t0 - t0.min()).dt.total_seconds()).to_numpy(np.int64)

    amount = df["amount"].astype(float).to_numpy(np.float32)
    amt_norm = (amount / (amount.std() + 1e-6)).astype(np.float32)
    log_amt = np.log1p(amount).astype(np.float32)
    ttype = _txn_type_onehot(df.get("transaction_type", pd.Series(["Other"] * len(df))))
    msg = np.concatenate([amt_norm[:, None], log_amt[:, None], ttype], axis=1).astype(np.float32)

    y = df["is_fraud"].astype(float).to_numpy(np.float32) if "is_fraud" in df else np.zeros(len(df), np.float32)

    data = TemporalData(
        src=torch.from_numpy(src),
        dst=torch.from_numpy(dst),
        t=torch.from_numpy(t),
        msg=torch.from_numpy(msg),
        y=torch.from_numpy(y),
    )
    train, val, test = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)

    id_to_name = {}
    for col_id, col_name in (("sender_id", "sender_name"), ("receiver_id", "receiver_name")):
        if col_name in df:
            for raw, nm in zip(df[col_id], df[col_name]):
                id_to_name[node_id[raw]] = str(nm)

    return {
        "data": data, "train": train, "val": val, "test": test,
        "num_nodes": len(nodes), "msg_dim": msg.shape[1], "id_to_name": id_to_name,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_temporal_data_loader.py -v`
Expected: PASS (or SKIP if torch absent).

- [ ] **Step 5: Commit**

```bash
git add src/temporal_data_loader.py tests/test_temporal_data_loader.py
git commit -m "feat(tgn): temporal data loader with chronological split"
```

---

### Task 2: TGN model modules

**Files:**
- Create: `src/tgn_model.py`
- Test: `tests/test_tgn_model.py`

**Interfaces:**
- Consumes: `torch_geometric.nn.TransformerConv`, `torch_geometric.nn.models.tgn.TimeEncoder`.
- Produces: `GraphAttentionEmbedding`, `FraudDecoder` (`forward(z_src, z_dst) -> logits [E,1]`), and `build_tgn(num_nodes, msg_dim, memory_dim=100, time_dim=100, embedding_dim=100) -> (memory, gnn, decoder)`.

- [ ] **Step 1: Write the failing test** (subprocess-isolated)

```python
# tests/test_tgn_model.py
"""Structural checks on the TGN modules — isolated subprocess (torch vs xgboost)."""
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def _run(body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r})
        try:
            import torch  # noqa
            from torch_geometric.nn import TransformerConv  # noqa
        except Exception:
            print("SKIP"); sys.exit(0)
    """) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def _ok(r):
    if "SKIP" in r.stdout:
        pytest.skip("torch/PyG not installed")
    assert r.returncode == 0, r.stderr
    return r


def test_build_tgn_and_decoder_shape():
    r = _ok(_run("""
        import torch
        from tgn_model import build_tgn
        memory, gnn, dec = build_tgn(num_nodes=20, msg_dim=5,
                                     memory_dim=16, time_dim=16, embedding_dim=16)
        z = torch.randn(8, 16)
        out = dec(z[:4], z[4:])
        assert out.shape == (4, 1), out.shape   # one fraud logit per edge
        print("OK")
    """))
    assert "OK" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tgn_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tgn_model'`.

- [ ] **Step 3: Implement `src/tgn_model.py`**

```python
"""TGN modules (Rossi et al. 2020), adapted for FRAUD classification.

The decoder is a single-logit fraud head over concat(Z_src, Z_dst) — not a
link-existence two-tower scorer. Memory/attention are the standard PyG TGN
primitives. Import torch lazily-safe (module import needs torch, so only import
this in a process that has already trained XGBoost, per the pipeline ordering).
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    TGNMemory, IdentityMessage, LastAggregator,
)


class GraphAttentionEmbedding(nn.Module):
    """Temporal-attention embedding: TransformerConv over recent neighbors,
    with relative-time encoding concatenated onto each edge's message."""

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class FraudDecoder(nn.Module):
    """MLP on concat(Z_src, Z_dst) -> a single fraud logit per edge."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.lin1 = nn.Linear(2 * in_channels, in_channels)
        self.lin2 = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = torch.cat([z_src, z_dst], dim=-1)
        h = self.lin1(h).relu()
        return self.lin2(h)


def build_tgn(num_nodes: int, msg_dim: int, memory_dim: int = 100,
              time_dim: int = 100, embedding_dim: int = 100):
    memory = TGNMemory(
        num_nodes, msg_dim, memory_dim, time_dim,
        message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    )
    gnn = GraphAttentionEmbedding(memory_dim, embedding_dim, msg_dim, memory.time_enc)
    decoder = FraudDecoder(embedding_dim)
    return memory, gnn, decoder
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tgn_model.py -v`
Expected: PASS (or SKIP).

- [ ] **Step 5: Commit**

```bash
git add src/tgn_model.py tests/test_tgn_model.py
git commit -m "feat(tgn): TGN memory + temporal-attention + single-logit fraud decoder"
```

---

### Task 3: Training loop + persistence

**Files:**
- Create: `src/train_tgn.py`
- Test: `tests/test_tgn_model.py` (add an end-to-end training test to the existing file)

**Interfaces:**
- Consumes: `load_temporal_data` (Task 1), `build_tgn` (Task 2), `ml_model.fbeta_optimal_threshold`.
- Produces: `train_tgn(data_dir, variant="ibm_aml", epochs=30, seed=42, batch_size=200, neighbors=10, patience=5) -> dict` (metrics); `load_tgn_metrics(data_dir, variant="ibm_aml") -> dict`; `load_tgn_predictions(data_dir, variant="ibm_aml") -> dict`. Persists `data/ml/{variant}/tgn/{metrics.json, model.pt, predictions.json}`.

- [ ] **Step 1: Write the failing test** (subprocess-isolated; a short real run on the tiny CSV)

```python
# add to tests/test_tgn_model.py
def test_train_tgn_persists_wellformed_artifacts(tmp_path=None):
    r = _ok(_run("""
        import os, json, tempfile
        import pandas as pd
        from train_tgn import train_tgn, load_tgn_metrics, load_tgn_predictions
        d = tempfile.mkdtemp()
        vdir = os.path.join(d, "tgn_demo"); os.makedirs(vdir)
        rows = []
        for i in range(400):
            rows.append(dict(
                transaction_id=f"T{i}",
                timestamp=f"2022-01-{(i%27)+1:02d} 00:00:00",
                sender_id=f"A{i%12}", receiver_id=f"B{i%15}",
                sender_name=f"A{i%12}", receiver_name=f"B{i%15}",
                amount=100.0+i, transaction_type="NEFT", is_fraud=int(i%5==0)))
        pd.DataFrame(rows).to_csv(os.path.join(vdir, "transactions.csv"), index=False)
        m = train_tgn(d, variant="tgn_demo", epochs=2, batch_size=64, neighbors=5, patience=2)
        assert "auprc" in m and "threshold" in m and "f2" in m
        assert os.path.exists(os.path.join(d, "ml", "tgn_demo", "tgn", "metrics.json"))
        loaded = load_tgn_metrics(d, variant="tgn_demo")
        assert loaded["auprc"] == m["auprc"]
        preds = load_tgn_predictions(d, variant="tgn_demo")["predictions"]
        probs = [p["prob"] for p in preds]
        assert probs == sorted(probs, reverse=True)  # ranked desc
        print("OK")
    """))
    assert "OK" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tgn_model.py -k train_tgn -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_tgn'`.

- [ ] **Step 3: Implement `src/train_tgn.py`**

```python
"""Train the TGN fraud classifier: chronological, predict-then-update, memory
detached each batch (no BPTT / VRAM blow-up), selected on val AUPRC, with an F2
operating point. Persists metrics + weights + a ranked predicted-fraud list.
Adapted from the canonical PyG TGN example (self-supervised link prediction) to
SUPERVISED fraud classification over real edges.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np


def _metrics(y_true, y_prob) -> Dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from ml_model import fbeta_optimal_threshold
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
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
    return {"auprc": round(float(average_precision_score(y_true, y_prob)), 4),
            "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "f1": round(f1, 4), "f2": round(f2, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "threshold": round(float(thr), 4)}


def train_tgn(data_dir: str, variant: str = "ibm_aml", epochs: int = 30,
              seed: int = 42, batch_size: int = 200, neighbors: int = 10,
              patience: int = 5, top_n: int = 50) -> Dict:
    import torch
    from torch_geometric.loader import TemporalDataLoader
    from torch_geometric.nn.models.tgn import LastNeighborLoader
    from temporal_data_loader import load_temporal_data
    from tgn_model import build_tgn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_temporal_data(os.path.join(data_dir, variant, "transactions.csv"))
    data = bundle["data"].to(device)
    train_data = bundle["train"].to(device)
    val_data = bundle["val"].to(device)
    test_data = bundle["test"].to(device)
    num_nodes, msg_dim = bundle["num_nodes"], bundle["msg_dim"]

    memory, gnn, decoder = build_tgn(num_nodes, msg_dim,
                                     memory_dim=100, time_dim=100, embedding_dim=100)
    memory, gnn, decoder = memory.to(device), gnn.to(device), decoder.to(device)
    nbr = LastNeighborLoader(num_nodes, size=neighbors, device=device)
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
    opt = torch.optim.Adam(list(memory.parameters()) + list(gnn.parameters())
                           + list(decoder.parameters()), lr=1e-4)
    n_pos = float((train_data.y == 1).sum()); n_neg = float((train_data.y == 0).sum())
    pos_weight = torch.tensor([max(n_neg, 1.0) / max(n_pos, 1.0)], device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _embed(n_id, edge_index, e_id):
        z = memory.memory[n_id]  # not used directly; kept for clarity
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))
        return z

    def process(split, train: bool):
        (memory.train if train else memory.eval)()
        (gnn.train if train else gnn.eval)()
        (decoder.train if train else decoder.eval)()
        loader = TemporalDataLoader(split, batch_size=batch_size)
        all_y, all_p = [], []
        for batch in loader:
            src, dst, t, msg, y = batch.src, batch.dst, batch.t, batch.msg, batch.y
            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = nbr(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
            logit = decoder(z[assoc[src]], z[assoc[dst]]).view(-1)
            if train:
                loss = crit(logit, y.float())
                opt.zero_grad(); loss.backward(); opt.step()
            # update memory + neighbors AFTER predicting (predict-then-update)
            memory.update_state(src, dst, t, msg)
            nbr.insert(src, dst)
            memory.detach()
            all_y.append(y.detach().cpu()); all_p.append(torch.sigmoid(logit).detach().cpu())
        return torch.cat(all_y).numpy(), torch.cat(all_p).numpy()

    best_auprc, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        memory.reset_state(); nbr.reset_state()
        process(train_data, train=True)              # warms memory + learns
        vy, vp = process(val_data, train=False)       # memory continues (no reset)
        v_auprc = _metrics(vy, vp)["auprc"]
        if v_auprc > best_auprc:
            best_auprc, bad = v_auprc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in
                          {**{f"m.{n}": p for n, p in memory.state_dict().items()},
                           **{f"g.{n}": p for n, p in gnn.state_dict().items()},
                           **{f"d.{n}": p for n, p in decoder.state_dict().items()}}.items()}
        else:
            bad += 1
            if bad >= patience:
                break

    # Final test with best weights: reset, warm on train, then score test.
    if best_state is not None:
        memory.load_state_dict({k[2:]: v for k, v in best_state.items() if k.startswith("m.")})
        gnn.load_state_dict({k[2:]: v for k, v in best_state.items() if k.startswith("g.")})
        decoder.load_state_dict({k[2:]: v for k, v in best_state.items() if k.startswith("d.")})
    memory.reset_state(); nbr.reset_state()
    process(train_data, train=False)
    process(val_data, train=False)
    ty, tp = process(test_data, train=False)
    metrics = _metrics(ty, tp)
    metrics.update({"model_kind": "tgn", "variant": variant, "epochs": epoch + 1,
                    "n_train": int(train_data.num_events), "n_val": int(val_data.num_events),
                    "n_test": int(test_data.num_events), "val_auprc": round(best_auprc, 4)})

    out_dir = os.path.join(data_dir, "ml", variant, "tgn")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"memory": memory.state_dict(), "gnn": gnn.state_dict(),
                "decoder": decoder.state_dict()}, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Ranked predicted-fraud list over the test window.
    id_to_name = bundle["id_to_name"]
    src_np = test_data.src.cpu().numpy(); dst_np = test_data.dst.cpu().numpy()
    order = np.argsort(-tp)[:top_n]
    preds = [{"src_id": int(src_np[i]), "dst_id": int(dst_np[i]),
              "src_name": id_to_name.get(int(src_np[i]), str(int(src_np[i]))),
              "dst_name": id_to_name.get(int(dst_np[i]), str(int(dst_np[i]))),
              "prob": round(float(tp[i]), 4), "is_fraud": int(ty[i])} for i in order]
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump({"variant": variant, "predictions": preds}, f, indent=2)
    return metrics


def load_tgn_metrics(data_dir: str, variant: str = "ibm_aml") -> Dict:
    path = os.path.join(data_dir, "ml", variant, "tgn", "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_tgn_predictions(data_dir: str, variant: str = "ibm_aml") -> Dict:
    path = os.path.join(data_dir, "ml", variant, "tgn", "predictions.json")
    if not os.path.exists(path):
        return {"predictions": []}
    with open(path) as f:
        return json.load(f)
```

> Implementer note: if `memory.state_dict()` cloning for `best_state` proves awkward, an equivalent acceptable approach is to `torch.save` the three state_dicts to a temp file on each val improvement and reload at the end — same effect (keep best-val weights). Keep the val-AUPRC selection and the predict-then-update + `memory.detach()` ordering exactly as shown.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tgn_model.py -v`
Expected: PASS (both structural + training tests; or SKIP if torch absent).

- [ ] **Step 5: Commit**

```bash
git add src/train_tgn.py tests/test_tgn_model.py
git commit -m "feat(tgn): training loop (predict-then-update, detach, AUPRC/F2) + persistence"
```

---

### Task 4: Pipeline integration

**Files:**
- Modify: `src/run_pipeline.py` (the torch section, after XGBoost — read the file to place it beside the existing `train_gnn` call)

**Interfaces:**
- Consumes: `train_tgn` (Task 3). Runs only after XGBoost has trained (torch ordering).

- [ ] **Step 1: Add the optional TGN training block**

Read `src/run_pipeline.py`. Immediately after the existing GraphSAGE (`train_gnn`) block (both are inside the post-XGBoost torch section), add:

```python
    # ── Temporal GNN (TGN) — optional, slowest trainer; skip if PyG absent ──
    try:
        from train_tgn import train_tgn
        print(f"\n[+] Training TGN (temporal fraud model, variant={dataset})...")
        tgn_m = train_tgn(data_dir, variant=dataset, epochs=30)
        print(f"  TGN AUPRC={tgn_m.get('auprc')}  F2={tgn_m.get('f2')}  "
              f"thr={tgn_m.get('threshold')}")
    except ImportError:
        print("  TGN skipped (torch/torch_geometric not installed).")
    except Exception as e:
        print(f"  TGN training failed (non-fatal): {e}")
```

- [ ] **Step 2: Smoke-check the import path (no full retrain)**

Run: `python -c "import ast; ast.parse(open('src/run_pipeline.py').read()); print('run_pipeline parses OK')"`
Expected: `run_pipeline parses OK`.

- [ ] **Step 3: Commit**

```bash
git add src/run_pipeline.py
git commit -m "feat(tgn): train TGN in the pipeline after XGBoost (optional)"
```

---

### Task 5: Backend endpoints

**Files:**
- Modify: `backend/main.py` (import; extend `/api/ml/metrics`; add `/api/tgn/predictions`)
- Test: `tests/test_api_integration.py`

**Interfaces:**
- Consumes: `load_tgn_metrics`, `load_tgn_predictions` (Task 3). `DATA_DIR`, `ACTIVE_VARIANT` exist in main.py.
- Produces: `/api/ml/metrics` gains a `tgn` key; `GET /api/tgn/predictions` → `{trained: bool, ...predictions.json}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_api_integration.py
def test_tgn_endpoints_shape(client):
    # metrics endpoint always responds; tgn key present (may be None if untrained)
    r = client.get("/api/ml/metrics", headers=INV)
    assert r.status_code == 200, r.text
    assert "tgn" in r.json()
    # predictions endpoint returns a well-formed payload either way
    p = client.get("/api/tgn/predictions", headers=INV)
    assert p.status_code == 200, p.text
    body = p.json()
    assert "trained" in body and "predictions" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_integration.py -k tgn -v`
Expected: FAIL — `"tgn" not in ...` and/or 404 on `/api/tgn/predictions`.

- [ ] **Step 3: Add the import**

Alongside `from gnn_model import load_gnn_metrics, load_gnn_edge_scores` (backend/main.py:53), add:

```python
from train_tgn import load_tgn_metrics, load_tgn_predictions
```

- [ ] **Step 4: Extend `/api/ml/metrics` and add the predictions route**

In `get_ml_metrics` (backend/main.py:1019), change the success return to include TGN:

```python
    gnn_m = load_gnn_metrics(DATA_DIR, variant=variant)
    tgn_m = load_tgn_metrics(DATA_DIR, variant=variant)
    return {"trained": True, **m, "gnn": gnn_m or None, "tgn": tgn_m or None}
```

Add near the other ML endpoints:

```python
@app.get("/api/tgn/predictions")
def get_tgn_predictions(variant: str = None):
    v = variant or ACTIVE_VARIANT
    data = load_tgn_predictions(DATA_DIR, variant=v)
    preds = data.get("predictions", [])
    return {"trained": bool(preds), "variant": v, "predictions": preds}
```

Note: the untrained-variant early return in `get_ml_metrics` (`if not m:`) has no `tgn` key — that's fine, the test only asserts `tgn` is present on the trained response; leave the untrained branch as-is.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_api_integration.py -k tgn -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py tests/test_api_integration.py
git commit -m "feat(tgn): serve TGN metrics + predicted-fraud list (JSON only)"
```

---

### Task 6: Frontend — ModelMetrics TGN column + predicted-fraud list

**Files:**
- Modify: `frontend/src/api.js` (add `getTgnPredictions`)
- Modify: `frontend/src/pages/ModelMetrics.jsx` (render `data.tgn` + the predicted-fraud list — read the file first to match its card/table style)
- Verify: browser (no component tests — lint + build + manual, per project convention)

**Interfaces:**
- Consumes: `/api/ml/metrics` (now has `tgn`) + `GET /api/tgn/predictions` (Task 5).

- [ ] **Step 1: Add the API helper**

In `frontend/src/api.js` add (matching the `fetchAPI` style):

```js
export const getTgnPredictions = (variant) =>
  fetchAPI(`/api/tgn/predictions${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`);
```

- [ ] **Step 2: Render TGN in `ModelMetrics.jsx`**

Read `frontend/src/pages/ModelMetrics.jsx`. It already renders `data` from `/api/ml/metrics` (and shows `data.gnn`). Add, matching the existing card style:
- A **TGN** row/card sourced from `data.tgn` (guard null): show `auprc`, `f2`, `precision`, `recall`, `threshold`, and a caption "Temporal GNN (Rossi 2020) — predicts fraud from the graph's time-evolution." If `data.tgn` is null, show "TGN not trained for this variant."
- A **"Predicted future fraud"** panel: on mount (and on variant change) call `getTgnPredictions(variant)`; if `trained`, render a small table of the top predictions (`src_name → dst_name`, `prob`, and a ✓/✗ for `is_fraud`); else show a muted "not trained" line. Use the same null/loading guards the page already uses.

- [ ] **Step 3: Lint + build**

Run: `cd frontend && npm run lint && npm run build`
Expected: clean (a pre-existing chunk-size warning is fine).

- [ ] **Step 4: Manual verification**

Start the stack; open ModelMetrics; confirm the TGN card shows its metrics next to XGB/SAGE/ensemble (or a clean "not trained" state before a retrain), and the predicted-fraud list renders. (A fresh clone shows "not trained" until `python src/run_pipeline.py` regenerates artifacts — acceptable.)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.js frontend/src/pages/ModelMetrics.jsx
git commit -m "feat(tgn): ModelMetrics TGN column + predicted-fraud list"
```

---

### Task 7: Regression

- [ ] **Step 1: Backend suite** (TGN tests self-skip if torch absent; run in the main process — they subprocess-isolate torch)

Run: `python -m pytest tests/ -q`
Expected: all pass (existing + new TGN tests).

- [ ] **Step 2: Frontend lint + build (final)**

Run: `cd frontend && npm run lint && npm run build`
Expected: clean.

---

## Self-Review

**Spec coverage:**
- Temporal data loader + chronological split (§4.1) → Task 1. ✓
- TGN architecture: memory/attention/decoder (§4.2) → Task 2. ✓
- Training loop: predict-then-update, `memory.detach()`, AUPRC selection, F2, persistence (§4.3) → Task 3. ✓
- Pipeline integration, XGB-before-torch, optional (§4.4, §7) → Task 4. ✓
- Serve-from-JSON metrics + predictions (§4.4) → Task 5. ✓
- ModelMetrics comparison + predicted-fraud list (§4.4) → Task 6. ✓
- Fraud (not link-existence) objective; pos_weight; no random negatives (§2) → enforced in Tasks 1/3.
- Honest eval next to other models (§6) → Task 5/6 surface TGN beside XGB/SAGE/ensemble.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". The one implementer note (Task 3 best-state save) offers an explicit equivalent, not a placeholder. ✓

**Type consistency:** `load_temporal_data` return keys (`data/train/val/test/num_nodes/msg_dim/id_to_name`) produced in Task 1, consumed verbatim in Task 3. `build_tgn(...) -> (memory, gnn, decoder)` (Task 2) consumed in Task 3. `train_tgn`/`load_tgn_metrics`/`load_tgn_predictions` (Task 3) consumed by Tasks 4 & 5. Metrics keys (`auprc, f2, precision, recall, threshold`) produced in Task 3, consumed by Tasks 5 & 6. `predictions[].{src_name,dst_name,prob,is_fraud}` produced in Task 3, consumed by Task 6. ✓
