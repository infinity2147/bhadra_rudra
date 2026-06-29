"""Fuzzy-threshold behaviour wired into the smurfing fan-out + funnel detectors.

Two guarantees under test:
  * margin == 0 reproduces today's hard-gate behaviour exactly (zero regression):
    near-miss cases stay dropped, clear hits keep their original confidence.
  * margin > 0 admits a near-miss at a confidence strictly attenuated by its
    fuzzy membership degree, so a near-miss never out-scores an equivalent
    clean hit.
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from graph_engine import FundFlowGraph  # noqa: E402
from fraud_detector import FraudDetector  # noqa: E402

_FMT = "%Y-%m-%d %H:%M:%S"


# ── smurfing fan-out: amount-near-threshold fuzzy gate ───────────────────────

def _row(sid, rid, amount, ts):
    return {
        "transaction_id": f"T{sid}{rid}{ts:%Y%m%d%H%M%S}",
        "sender_id": sid, "sender_name": sid, "sender_type": "individual", "sender_branch": "BR1",
        "receiver_id": rid, "receiver_name": rid, "receiver_type": "individual", "receiver_branch": "BR2",
        "amount": float(amount), "timestamp": ts.strftime(_FMT),
        "transaction_type": "NEFT", "channel": "NetBanking", "is_fraud": 0,
    }


def _smurf_detector(rows, config=None):
    df = pd.DataFrame(rows)
    g = FundFlowGraph().build_graph(df)
    return FraudDetector(g, transactions=df, config=config or {})


def _fanout_rows(amount):
    """10 sub-/near-threshold transfers, one each to 10 fresh mules, over 10 days."""
    base = datetime(2025, 3, 1, 12, 0, 0)
    return [_row("S", f"R{i}", amount, base + timedelta(days=i)) for i in range(10)]


def test_fanout_just_above_threshold_admitted_only_with_margin():
    # default smurfing_threshold = 200_000; 205k is just above it.
    rows = _fanout_rows(205_000)

    hard = [a for a in _smurf_detector(rows).detect_smurfing()
            if a.get("detection_mode") == "fan_out"]
    assert not hard, "near-threshold fan-out must NOT fire with the hard gate (margin 0)"

    fuzzy = [a for a in _smurf_detector(rows, {"smurfing_fuzzy_margin": 0.05}).detect_smurfing()
             if a.get("detection_mode") == "fan_out"]
    assert fuzzy, "near-threshold fan-out must fire once a fuzzy margin admits it"
    assert fuzzy[0].get("fuzzy") is True


def test_fanout_below_threshold_confidence_unchanged_by_margin():
    """Regression guard: a clearly sub-threshold fan-out scores identically with
    and without a fuzzy margin (the amounts are full-membership either way)."""
    rows = _fanout_rows(190_000)

    base = [a for a in _smurf_detector(rows).detect_smurfing()
            if a.get("detection_mode") == "fan_out"]
    fuzzy = [a for a in _smurf_detector(rows, {"smurfing_fuzzy_margin": 0.05}).detect_smurfing()
             if a.get("detection_mode") == "fan_out"]

    assert base and fuzzy
    assert base[0]["confidence"] == fuzzy[0]["confidence"]
    assert not base[0].get("fuzzy")


# ── funnel: flow-imbalance fuzzy gate ────────────────────────────────────────

def _add_edge(g, u, v, amount, t):
    g.add_edge(
        u, v,
        total_amount=float(amount), transaction_count=1,
        avg_amount=float(amount), min_amount=float(amount), max_amount=float(amount),
        std_amount=0.0,
        first_seen=t.strftime(_FMT), last_seen=t.strftime(_FMT),
        fraud_count=0,
    )


def _funnel_graph(in_total, out_total):
    """3 sources -> M -> 1 sink, with a controllable in/out flow imbalance."""
    import networkx as nx
    g = nx.DiGraph()
    t0 = datetime(2025, 1, 1, 10, 0, 0)
    share = in_total / 3.0
    for i, src in enumerate(("A", "B", "C")):
        _add_edge(g, src, "M", share, t0 + timedelta(minutes=i))
    _add_edge(g, "M", "Z", out_total, t0 + timedelta(hours=5))
    for n in ("A", "B", "C", "M", "Z"):
        g.nodes[n]["type"] = "individual"
        g.nodes[n]["name"] = n
    return g


def _flagged(alerts):
    return {a.get("funnel_entity") for a in alerts}


def test_funnel_near_miss_imbalance_admitted_only_with_margin():
    # imbalance = (4.714M - 1.0M)/(5.714M) = 0.65, below the 0.7 default gate.
    g = _funnel_graph(in_total=4_714_286, out_total=1_000_000)

    hard = FraudDetector(g, config={}).detect_shell_funnels()
    assert "M" not in _flagged(hard), "0.65 imbalance must NOT fire with the hard 0.7 gate"

    fuzzy = FraudDetector(g, config={"funnel_fuzzy_margin": 0.1}).detect_shell_funnels()
    m = next((a for a in fuzzy if a.get("funnel_entity") == "M"), None)
    assert m is not None, "0.65 imbalance must fire once a fuzzy margin admits it"
    assert m.get("fuzzy") is True


def test_funnel_fuzzy_confidence_scales_with_membership():
    """Same near-miss node, wider margin -> higher membership -> higher confidence.
    Base score is identical (same imbalance), so this isolates the fuzzy factor."""
    g = _funnel_graph(in_total=4_714_286, out_total=1_000_000)

    narrow = next(a for a in FraudDetector(g, config={"funnel_fuzzy_margin": 0.1}).detect_shell_funnels()
                  if a.get("funnel_entity") == "M")
    wide = next(a for a in FraudDetector(g, config={"funnel_fuzzy_margin": 0.3}).detect_shell_funnels()
                if a.get("funnel_entity") == "M")
    assert wide["confidence"] > narrow["confidence"]


def test_funnel_clear_hit_unchanged_by_margin():
    """Regression guard: a clear-hit imbalance (0.9) scores identically with and
    without a fuzzy margin."""
    g = _funnel_graph(in_total=9_500_000, out_total=500_000)  # imbalance 0.90

    base = next(a for a in FraudDetector(g, config={}).detect_shell_funnels()
                if a.get("funnel_entity") == "M")
    fuzzy = next(a for a in FraudDetector(g, config={"funnel_fuzzy_margin": 0.1}).detect_shell_funnels()
                 if a.get("funnel_entity") == "M")
    assert base["confidence"] == fuzzy["confidence"]
    assert not base.get("fuzzy")
