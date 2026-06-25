"""Foundational properties of the profile-mismatch detector.

Two flaws the team flagged:
  1. "9-transaction shell" — a shell was only anomalous on transaction *count*
     (>10), so a launderer could push 9 massive transfers through it and stay
     silent. A shell moving large *volume* (or large single transfers) is
     anomalous regardless of count.
  2. The nighttime window was hardcoded `(hours < 6) | (hours > 22)`, which is
     both un-configurable and inconsistent with the documented 22:00 boundary
     (it silently excluded the whole 22:00–22:59 hour).
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from graph_engine import FundFlowGraph  # noqa: E402
from advanced_detectors import ProfileMismatchDetector  # noqa: E402


def _row(sid, rid, amount, ts, stype, purpose="Business Payment"):
    return {
        "transaction_id": f"T{sid}{rid}{ts:%H%M%S}",
        "sender_id": sid, "sender_name": sid, "sender_type": stype, "sender_branch": "BR1",
        "receiver_id": rid, "receiver_name": rid, "receiver_type": "individual", "receiver_branch": "BR2",
        "amount": float(amount), "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_type": "NEFT", "channel": "NetBanking", "purpose_code": purpose, "is_fraud": 0,
    }


def _detect(rows):
    df = pd.DataFrame(rows)
    g = FundFlowGraph().build_graph(df)
    return ProfileMismatchDetector(g, df, risk_scores=[]).detect()


def test_high_value_shell_with_few_txns_is_flagged():
    """A shell pushing 9 large transfers (≤10 count) must still be flagged on
    volume / per-transfer size."""
    rows = []
    noon = datetime(2025, 3, 1, 12, 0, 0)
    for i in range(9):
        rows.append(_row("SH", f"R{i}", 1_500_000, noon + timedelta(minutes=i), "shell_company"))
    alerts = _detect(rows)
    flagged = {e for a in alerts for e in a["entities"]}
    assert "SH" in flagged, f"high-value shell with 9 txns not flagged; alerts={alerts}"


def test_2200_hour_counts_as_night():
    """Activity at 22:15 must register as nighttime; the old `> 22` boundary
    silently excluded the entire 22:00–22:59 hour."""
    rows = []
    night = datetime(2025, 3, 1, 22, 15, 0)
    # 5 transfers of ₹2M each (also trips the individual-averaging rule, giving
    # a 2nd mismatch so the alert clears the min-rule-score gate).
    for i in range(5):
        rows.append(_row("N", f"R{i}", 2_000_000, night + timedelta(minutes=i), "individual", purpose="Personal"))
    alerts = _detect(rows)
    n_alerts = [a for a in alerts if "N" in a["entities"]]
    assert n_alerts, f"22:15 activity did not register as night; alerts={alerts}"
    assert any("night" in m.lower() for m in n_alerts[0]["mismatches"]), \
        f"expected a nighttime mismatch; got {n_alerts[0]['mismatches']}"
