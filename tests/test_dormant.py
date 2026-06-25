"""Foundational properties of the dormant-activation detector.

Two flaws the team flagged:
  1. "Slow drip" — post-activation activity was averaged over only the first 5
     active days, so ₹1/day for 5 days then ₹500M on day 6 kept the mean near
     zero and the spike was never seen. Detection must key on the *peak* daily
     amount over a longer post-activation window.
  2. The pre-activation std fell back to a hardcoded 1 when history was
     constant, so a 20% bump on a ₹1M-regular account produced a z-score in the
     hundreds of thousands and a spurious CRITICAL. The fallback must be
     scale-relative.
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from graph_engine import FundFlowGraph  # noqa: E402
from advanced_detectors import DormantActivationDetector  # noqa: E402


def _row(amount, ts):
    return {
        "transaction_id": f"T{ts:%Y%m%d}", "sender_id": "M", "sender_name": "M",
        "sender_type": "individual", "sender_branch": "BR1",
        "receiver_id": "R", "receiver_name": "R", "receiver_type": "business", "receiver_branch": "BR2",
        "amount": float(amount), "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_type": "NEFT", "channel": "NetBanking", "is_fraud": 0,
    }


def _detect(rows):
    df = pd.DataFrame(rows)
    g = FundFlowGraph().build_graph(df)
    return DormantActivationDetector(g, df).detect()


def test_slow_drip_then_spike_is_detected():
    """₹1/day for 5 days after dormancy, then ₹500M on day 6 — the spike must
    be caught even though the first-5-day average is ~₹1."""
    rows = []
    # baseline history
    for d in range(3):
        rows.append(_row(100_000, datetime(2025, 1, 1) + timedelta(days=d)))
    # dormant ~57 days, then a slow drip
    for d in range(5):
        rows.append(_row(1, datetime(2025, 3, 1) + timedelta(days=d)))
    # the real move on day 6
    rows.append(_row(500_000_000, datetime(2025, 3, 6)))

    alerts = _detect(rows)
    flagged = {e for a in alerts for e in a["entities"]}
    assert "M" in flagged, f"slow-drip spike not detected; alerts={alerts}"


def test_constant_history_modest_bump_not_flagged():
    """A perfectly regular ₹1M/day account that bumps to ₹1.2M after a gap is
    not anomalous — the std fallback must not manufacture a giant z-score."""
    rows = []
    for d in range(4):
        rows.append(_row(1_000_000, datetime(2025, 1, 1) + timedelta(days=d)))
    rows.append(_row(1_200_000, datetime(2025, 3, 1)))   # 20% bump after a gap

    alerts = _detect(rows)
    flagged = {e for a in alerts for e in a["entities"]}
    assert "M" not in flagged, \
        f"a 20% bump on a regular account must not be CRITICAL; alerts={alerts}"
