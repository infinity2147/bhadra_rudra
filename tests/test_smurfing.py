"""Foundational properties of the smurfing / structuring detector.

Two flaws the team flagged:
  1. "Single-shot fan-out" — a sender spraying sub-threshold transfers once each
     to many distinct mules evades Mode 1 (which needs ≥2 to the *same*
     receiver) and Mode 2 (which needs them inside one short time window). A
     window-independent fan-out mode closes that gap.
  2. The edge-cluster proximity band was hardcoded to [0.7, 1.0] × threshold, so
     deliberate structuring at 50% of the threshold was invisible. The band
     must be configurable; low-variance repetition is the real signature.
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from graph_engine import FundFlowGraph  # noqa: E402
from fraud_detector import FraudDetector  # noqa: E402


def _row(sid, rid, amount, ts, stype="individual", rtype="individual"):
    return {
        "transaction_id": f"T{sid}{rid}{ts:%Y%m%d%H%M%S}",
        "sender_id": sid, "sender_name": sid, "sender_type": stype, "sender_branch": "BR1",
        "receiver_id": rid, "receiver_name": rid, "receiver_type": rtype, "receiver_branch": "BR2",
        "amount": float(amount), "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_type": "NEFT", "channel": "NetBanking", "is_fraud": 0,
    }


def _detector(rows):
    df = pd.DataFrame(rows)
    g = FundFlowGraph().build_graph(df)
    return FraudDetector(g, transactions=df)


def test_single_shot_fanout_is_detected():
    """One sender, 10 sub-threshold transfers, one each to 10 distinct receivers,
    spread over 10 days — invisible to same-receiver and windowed modes."""
    rows = []
    base = datetime(2025, 3, 1, 12, 0, 0)
    for i in range(10):
        rows.append(_row("S", f"R{i}", 190_000, base + timedelta(days=i)))
    alerts = _detector(rows).detect_smurfing()

    fanout = [a for a in alerts if a.get("detection_mode") == "fan_out"]
    assert fanout, f"single-shot fan-out not detected; modes={[a.get('detection_mode') for a in alerts]}"
    assert "S" in fanout[0]["entities"][0]


def test_fanout_detected_amid_background_noise():
    """The structured fan-out (12 transfers of ~₹190k to 12 fresh mules) must be
    found even when the same sender also makes plenty of unrelated, varied
    sub-threshold transactions — a real launderer mixes structuring into normal
    activity, so a uniform-amount requirement over ALL txns isn't enough."""
    rows = []
    base = datetime(2025, 3, 1, 9, 0, 0)
    # structured cluster: 12 near-identical sub-threshold transfers, one per mule
    for i in range(12):
        rows.append(_row("S", f"MULE{i}", 190_000 + (i % 3) * 1_000, base + timedelta(days=i)))
    # background noise: varied sub-threshold transfers to a few repeat counterparties
    import random as _r
    _r.seed(1)
    for j in range(15):
        rows.append(_row("S", f"VENDOR{j % 4}", _r.randint(5_000, 150_000), base + timedelta(days=j, hours=3)))
    alerts = _detector(rows).detect_smurfing()

    fanout = [a for a in alerts if a.get("detection_mode") == "fan_out"]
    assert fanout, f"structured fan-out buried in noise not detected; modes={[a.get('detection_mode') for a in alerts]}"
    # The flagged cluster should be the ~190k mules, not the noisy vendors.
    assert fanout[0]["n_distinct_receivers"] >= 5


def test_structuring_below_70pct_band_is_detected():
    """Repeated transfers at 50% of the reporting threshold are deliberate
    structuring; the old [0.7,1.0] band made them invisible."""
    rows = []
    base = datetime(2025, 3, 1, 12, 0, 0)
    # S -> R1 (x2) and S -> R2 (x2), all exactly ₹1L (= 0.5 × ₹2L threshold)
    for rid in ("R1", "R2"):
        for k in range(2):
            rows.append(_row("S", rid, 100_000, base + timedelta(minutes=k)))
    alerts = _detector(rows).detect_smurfing()

    edge_cluster = [a for a in alerts if a.get("detection_mode") == "edge_cluster"]
    assert edge_cluster, \
        f"structuring at 50% of threshold not detected; modes={[a.get('detection_mode') for a in alerts]}"
