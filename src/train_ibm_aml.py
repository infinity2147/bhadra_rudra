"""
IBM AML end-to-end training.

Loads the stratified 100k sample of HI-Small_Trans, builds the fund-flow
graph, runs detectors, then trains:

    1. XGBoost edge classifier      → data/ml/ibm_aml/
    2. GraphSAGE GNN                → data/ml/ibm_aml/gnn/
    3. Stacked ensemble (XGB+SAGE+GAT, LR meta-learner)
       via 3-fold OOF stacking      → data/ml/ibm_aml/ensemble/

Persists metrics + per-edge scores so the FastAPI endpoints
(/api/ml/metrics?variant=ibm_aml, /api/ml/ensemble?variant=ibm_aml) have
something real to return.

Idempotent: re-running just re-trains and overwrites the artefacts. Safe
to call from the FastAPI pipeline endpoint.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_data_loader import load_ibm_aml
from graph_engine import FundFlowGraph
from ml_model import train_and_save
from gnn_model import train_gnn
from ensemble_model import train_ensemble


def main(
    data_dir: str = None,
    sample_size: int = 100_000,
    n_folds: int = 3,
    gnn_epochs: int = 200,
):
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = data_dir or os.path.join(here, "..", "data")
    cached = os.path.join(data_dir, "real", "ibm_aml", "HI-Small_Trans_100k_sampled.csv")

    t0 = time.perf_counter()
    if os.path.exists(cached):
        print(f"[ibm_aml] Loading pre-sampled file: {cached}")
        df = pd.read_csv(cached)
    else:
        print(f"[ibm_aml] Sampling {sample_size} rows from HI-Small_Trans.csv...")
        df, _ = load_ibm_aml(sample_size=sample_size, seed=42)
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        df.to_csv(cached, index=False)

    print(f"[ibm_aml] {len(df)} rows | fraud rate {df['is_fraud'].mean():.4f} "
          f"| {df['sender_id'].nunique()} senders | {df['receiver_id'].nunique()} receivers")
    print(f"[ibm_aml] Building graph...")

    graph = FundFlowGraph().build_graph(df)
    print(f"[ibm_aml] graph: {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges")

    # ── Learned risk-score weights (T2.10) — fast, runs first ────────────────
    print("[ibm_aml] Training risk-score LR weights...")
    try:
        from risk_score_learner import train_risk_weights
        rw_m = train_risk_weights(graph, data_dir, variant="ibm_aml")
        if rw_m.get("trained"):
            print(f"   RiskLR F1 {rw_m['f1']:.3f}  AUC {rw_m['auc']:.3f}  "
                  f"({rw_m['n_train']} train / {rw_m['n_test']} test)")
        else:
            print(f"   skipped: {rw_m.get('reason')}")
    except Exception as e:
        print(f"   failed: {e}")

    # ── XGBoost baseline ─────────────────────────────────────────────────────
    print("[ibm_aml] Training XGBoost...")
    t = time.perf_counter()
    xgb_m = train_and_save(graph, df, data_dir, variant="ibm_aml", dataset_name="IBM AML 100k")
    print(f"   XGB  F1={xgb_m['f1']:.3f}  AUC={xgb_m['auc']:.3f}  "
          f"P={xgb_m['precision']:.3f}  R={xgb_m['recall']:.3f}  "
          f"({time.perf_counter() - t:.1f}s)")

    # ── GraphSAGE baseline ───────────────────────────────────────────────────
    print("[ibm_aml] Training GraphSAGE...")
    t = time.perf_counter()
    try:
        gnn_m = train_gnn(graph, data_dir, variant="ibm_aml", epochs=gnn_epochs)
        early = "(early-stopped)" if gnn_m.get("early_stopped") else ""
        print(f"   SAGE F1={gnn_m['f1']:.3f}  AUC={gnn_m['auc']:.3f}  "
              f"loss={gnn_m['final_loss']:.4f}  "
              f"epochs={gnn_m.get('completed_epochs','?')}{early}  "
              f"({time.perf_counter() - t:.1f}s)")
    except ImportError as e:
        print(f"   SAGE skipped (PyG missing): {e}")
        gnn_m = None

    # ── Stacked ensemble ─────────────────────────────────────────────────────
    print(f"[ibm_aml] Training stacked ensemble ({n_folds} folds, {gnn_epochs} epochs/fold)...")
    t = time.perf_counter()
    try:
        ens_m = train_ensemble(
            graph, df, data_dir, variant="ibm_aml", dataset_name="IBM AML 100k",
            n_folds=n_folds, gnn_epochs=gnn_epochs,
        )
        ens = ens_m["ensemble"]
        print(f"   ENS  F1={ens['f1']:.3f}  AUC={ens['auc']:.3f}  "
              f"P={ens['precision']:.3f}  R={ens['recall']:.3f}  "
              f"({time.perf_counter() - t:.1f}s)")
        print("   Base model contribution (LR coefficients):")
        for k, v in ens_m["meta_learner"]["coefficients"].items():
            print(f"     {k:<10s}{v:+.3f}")
    except Exception as e:
        print(f"   Ensemble training failed: {e}")
        ens_m = None

    print(f"[ibm_aml] DONE in {time.perf_counter() - t0:.1f}s")
    return {"xgb": xgb_m, "gnn": gnn_m, "ensemble": ens_m}


if __name__ == "__main__":
    main()
