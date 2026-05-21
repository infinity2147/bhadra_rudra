"""The data generator must produce a self-consistent schema with embedded fraud."""

import pandas as pd


def test_generator_produces_expected_columns(synthetic_pipeline):
    df = synthetic_pipeline["df"]
    required = {
        "transaction_id", "timestamp",
        "sender_id", "sender_name", "sender_type", "sender_branch", "sender_product",
        "receiver_id", "receiver_name", "receiver_type", "receiver_branch", "receiver_product",
        "amount", "transaction_type", "channel", "purpose_code",
        "is_fraud", "fraud_pattern",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_generator_produces_both_classes(synthetic_pipeline):
    df = synthetic_pipeline["df"]
    assert df["is_fraud"].any(), "No fraud transactions generated"
    assert (~df["is_fraud"]).any(), "No normal transactions generated"


def test_generator_includes_all_fraud_patterns(synthetic_pipeline):
    df = synthetic_pipeline["df"]
    patterns = set(df.loc[df["is_fraud"], "fraud_pattern"].unique())
    expected = {"circular_transaction", "rapid_layering", "smurfing", "shell_funnel",
                "dormant_activation"}
    assert expected.issubset(patterns), f"Missing patterns: {expected - patterns}"


def test_timestamps_are_chronological(synthetic_pipeline):
    df = synthetic_pipeline["df"]
    ts = pd.to_datetime(df["timestamp"])
    diffs = ts.diff().dropna()
    # After sort, all diffs should be >= 0
    assert (diffs.dt.total_seconds() >= 0).all(), "Transactions are not sorted"


def test_no_self_loops(synthetic_pipeline):
    df = synthetic_pipeline["df"]
    self_loops = df[df["sender_id"] == df["receiver_id"]]
    assert len(self_loops) == 0, f"Found {len(self_loops)} self-loop transactions"


def test_channels_are_valid(synthetic_pipeline):
    df = synthetic_pipeline["df"]
    valid_channels = {"Branch", "NetBanking", "MobileApp", "ATM", "UPI", "ThirdPartyAPI"}
    actual = set(df["channel"].unique())
    assert actual.issubset(valid_channels), f"Unexpected channels: {actual - valid_channels}"
