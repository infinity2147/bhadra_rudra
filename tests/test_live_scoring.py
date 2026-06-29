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


def test_live_scc_matches_training_and_caches(synthetic_pipeline):
    """in_scc_3plus at serve time must match the training definition (both
    endpoints in the union of SCCs >= 3) and must not recompute SCCs per call."""
    import networkx as nx
    from live_scoring import _build_live_features
    g = synthetic_pipeline["graph"]
    truth = set()
    for comp in nx.strongly_connected_components(g):
        if len(comp) >= 3:
            truth.update(comp)
    nodes = list(g.nodes())
    s, r = nodes[0], nodes[1]
    ts = pd.Timestamp.now()
    f_supplied = _build_live_features(g, s, r, 100_000, "MobileApp", "NEFT", ts, scc_members=truth)
    f_lazy = _build_live_features(g, s, r, 100_000, "MobileApp", "NEFT", ts)
    # serve == train definition: both endpoints in the 3+ SCC union
    assert f_supplied["in_scc_3plus"] == float(s in truth and r in truth)
    # caller-supplied and lazy-cached paths agree
    assert f_supplied["in_scc_3plus"] == f_lazy["in_scc_3plus"]
    # the lazy path cached the member set on the graph (no per-call recompute)
    assert g.graph.get("_scc3_members") == truth


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
