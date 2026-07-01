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

    # PyG's quantile-based split can yield empty val/test when timestamps are
    # heavily concentrated (e.g. only a few distinct values).  Fall back to a
    # strict index-based chronological split that guarantees all three
    # partitions are non-empty.
    n = len(data.t)
    if len(val.t) == 0 or len(test.t) == 0:
        i_val = max(1, int(n * 0.70))
        i_test = max(i_val + 1, int(n * 0.85))
        i_test = min(i_test, n - 1)
        train, val, test = data[:i_val], data[i_val:i_test], data[i_test:]

    id_to_name = {}
    for col_id, col_name in (("sender_id", "sender_name"), ("receiver_id", "receiver_name")):
        if col_name in df:
            for raw, nm in zip(df[col_id], df[col_name]):
                id_to_name[node_id[raw]] = str(nm)

    return {
        "data": data, "train": train, "val": val, "test": test,
        "num_nodes": len(nodes), "msg_dim": msg.shape[1], "id_to_name": id_to_name,
    }
