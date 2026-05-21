"""Live per-transaction scoring + latency benchmark."""

import os

import pandas as pd


def test_score_live_txn_returns_proba_and_latency(synthetic_pipeline, temp_data_dir):
    from ml_model import train_and_save, load_model
    from live_scoring import score_live_txn
    train_and_save(synthetic_pipeline["graph"], synthetic_pipeline["df"],
                   temp_data_dir, variant="syn")
    bundle = load_model(temp_data_dir, variant="syn")
    nodes = list(synthetic_pipeline["graph"].nodes())
    res = score_live_txn(
        bundle, synthetic_pipeline["graph"],
        nodes[0], nodes[1], amount=500_000,
        channel="MobileApp", rail="NEFT",
        timestamp=pd.Timestamp.now(),
    )
    assert 0 <= res["ml_score"] <= 1
    assert "latency_ms" in res
    assert res["latency_ms"]["total"] >= 0


def test_benchmark_pipeline_runs(synthetic_pipeline, temp_data_dir):
    from ml_model import train_and_save, load_model
    from live_scoring import benchmark_pipeline
    train_and_save(synthetic_pipeline["graph"], synthetic_pipeline["df"],
                   temp_data_dir, variant="syn")
    bundle = load_model(temp_data_dir, variant="syn")
    b = benchmark_pipeline(synthetic_pipeline["graph"], synthetic_pipeline["df"], bundle, n_samples=10)
    assert b["per_txn_ms"]["mean"] >= 0
    assert b["pipeline_ms"]["total"] > 0
    assert b["vs_t_plus_1"]["speedup_factor"] > 1
