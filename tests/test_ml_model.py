"""ML model must train, score, and persist correctly."""

import json
import os

import numpy as np


def test_extract_features_matches_column_order(synthetic_pipeline):
    from ml_model import FEATURE_COLUMNS, extract_features
    X = extract_features(synthetic_pipeline["graph"], synthetic_pipeline["df"])
    assert list(X.columns) == FEATURE_COLUMNS, "Feature column order mismatch"
    assert not X.isnull().any().any(), "Feature matrix has NaNs"


def test_no_leakage_feature_present():
    """Sanity check that we removed the neighbour-fraud-density feature."""
    from ml_model import FEATURE_COLUMNS
    assert "neighbor_fraud_density" not in FEATURE_COLUMNS


def test_train_and_save_writes_artifacts(synthetic_pipeline, temp_data_dir):
    from ml_model import train_and_save, load_model, load_metrics, load_edge_scores
    metrics = train_and_save(
        synthetic_pipeline["graph"], synthetic_pipeline["df"],
        temp_data_dir, variant="test_synthetic",
    )
    assert metrics["model_kind"] in {"xgboost", "gradient_boosting_fallback"}
    assert metrics["f1"] >= 0 and metrics["f1"] <= 1
    # Files exist
    assert os.path.exists(os.path.join(temp_data_dir, "ml", "test_synthetic", "model.pkl"))
    assert os.path.exists(os.path.join(temp_data_dir, "ml", "test_synthetic", "metrics.json"))
    assert os.path.exists(os.path.join(temp_data_dir, "ml", "test_synthetic", "edge_scores.json"))
    # Loadable
    bundle = load_model(temp_data_dir, variant="test_synthetic")
    assert bundle is not None
    assert "model" in bundle and "feature_columns" in bundle
    # Edge scores cover every edge
    edge_scores = load_edge_scores(temp_data_dir, variant="test_synthetic")
    assert len(edge_scores) == synthetic_pipeline["graph"].number_of_edges()


def test_predict_one_returns_probability(synthetic_pipeline, temp_data_dir):
    from ml_model import train_and_save, load_model, predict_one, FEATURE_COLUMNS
    train_and_save(synthetic_pipeline["graph"], synthetic_pipeline["df"],
                   temp_data_dir, variant="test_pred")
    bundle = load_model(temp_data_dir, variant="test_pred")
    fake_row = {c: 0.0 for c in FEATURE_COLUMNS}
    p = predict_one(bundle, fake_row)
    assert 0 <= p <= 1
