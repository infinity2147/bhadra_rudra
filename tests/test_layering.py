"""Foundational properties of the rapid-layering detector.

These tests pin the *defining* characteristics of layering, each of which the
pre-rewrite implementation got wrong:

  1. Temporal causality — money cannot leave an account before it arrives.
  2. Rapidity — a layering relay forwards funds quickly (within a window);
     hops a month apart are not "rapid layering".
  3. Follow-the-money branching — the detector must follow the largest
     transfers, not the first-N edges in graph-insertion order, so decoy
     micro-transfers can't bury the real chain.
  4. Bottleneck accounting — the reported flow is the tranche that traversed
     the whole chain (min edge), not the sum of every hop (which counts the
     same money once per hop).

The detector reads only the aggregated graph (edge total_amount +
first_seen/last_seen), so these graphs are hand-built with exactly those
attributes.
"""

import os
import sys
from datetime import datetime, timedelta

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fraud_detector import FraudDetector  # noqa: E402


_FMT = "%Y-%m-%d %H:%M:%S"


def _add_edge(g, u, v, amount, t_first, t_last=None, txns=1):
    t_last = t_last or t_first
    g.add_edge(
        u, v,
        total_amount=float(amount),
        transaction_count=txns,
        avg_amount=float(amount) / txns,
        min_amount=float(amount) / txns,
        max_amount=float(amount) / txns,
        std_amount=0.0,
        first_seen=t_first.strftime(_FMT),
        last_seen=t_last.strftime(_FMT),
        fraud_count=0,
    )


def _set_types(g, **types):
    for node, t in types.items():
        g.nodes[node]["type"] = t
        g.nodes[node].setdefault("name", node)


def _chains(alerts):
    return [tuple(a["entities"]) for a in alerts]


def test_detects_a_clean_rapid_causal_chain():
    """A → B → C → D, amounts preserved, hops ~30 min apart → one alert."""
    g = nx.DiGraph()
    t0 = datetime(2025, 3, 1, 10, 0, 0)
    _add_edge(g, "A", "B", 1_000_000, t0)
    _add_edge(g, "B", "C", 950_000, t0 + timedelta(minutes=30))
    _add_edge(g, "C", "D", 900_000, t0 + timedelta(minutes=60))
    _set_types(g, A="individual", B="business", C="shell_company", D="individual")

    alerts = FraudDetector(g).detect_rapid_layering()

    assert ("A", "B", "C", "D") in _chains(alerts), \
        f"clean rapid chain not detected; got {_chains(alerts)}"


def test_rejects_temporally_impossible_chain():
    """B forwards to C *before* it received from A — money can't move backward in time."""
    g = nx.DiGraph()
    _add_edge(g, "A", "B", 1_000_000, datetime(2025, 3, 1))
    _add_edge(g, "B", "C", 950_000, datetime(2025, 1, 1))   # a month BEFORE arrival
    _add_edge(g, "C", "D", 900_000, datetime(2025, 3, 2))
    _set_types(g, A="individual", B="business", C="business", D="individual")

    alerts = FraudDetector(g).detect_rapid_layering()

    assert alerts == [], \
        f"temporally impossible chain should not be flagged; got {_chains(alerts)}"


def test_rejects_non_rapid_chain_outside_time_window():
    """Hops a month apart are not *rapid* layering (default window 48h)."""
    g = nx.DiGraph()
    t0 = datetime(2025, 1, 1)
    _add_edge(g, "A", "B", 1_000_000, t0)
    _add_edge(g, "B", "C", 950_000, t0 + timedelta(days=30))
    _add_edge(g, "C", "D", 900_000, t0 + timedelta(days=60))
    _set_types(g, A="individual", B="business", C="business", D="individual")

    alerts = FraudDetector(g).detect_rapid_layering()

    assert alerts == [], \
        f"chain spanning months should not be flagged as rapid; got {_chains(alerts)}"


def test_follows_largest_transfer_not_insertion_order():
    """Decoy micro-transfers inserted *before* the real transfer must not hide it.

    The source fires 12 small decoys to dead-end accounts, then the real
    ₹5M transfer that starts the layering chain. With branching capped at 10,
    an insertion-order detector slices off the first 10 edges and never sees
    the real transfer. A follow-the-money detector ranks by amount and finds
    the chain regardless.
    """
    g = nx.DiGraph()
    t0 = datetime(2025, 3, 1, 9, 0, 0)
    # 12 decoys inserted FIRST (above the dust floor, but going nowhere).
    for i in range(12):
        _add_edge(g, "S", f"D{i}", 100_000, t0)
    # The real chain, inserted AFTER the decoys.
    _add_edge(g, "S", "M1", 5_000_000, t0)
    _add_edge(g, "M1", "M2", 4_800_000, t0 + timedelta(minutes=20))
    _add_edge(g, "M2", "M3", 4_600_000, t0 + timedelta(minutes=40))
    types = {"S": "individual", "M1": "shell_company", "M2": "shell_company", "M3": "individual"}
    types.update({f"D{i}": "individual" for i in range(12)})
    _set_types(g, **types)

    alerts = FraudDetector(g).detect_rapid_layering()
    chains = _chains(alerts)

    assert ("S", "M1", "M2", "M3") in chains, \
        f"real high-value chain missed behind decoys; got {chains}"
    decoy_nodes = {f"D{i}" for i in range(12)}
    for a in alerts:
        assert not (set(a["entities"]) & decoy_nodes), \
            f"decoy dead-end leaked into an alert: {a['entities']}"


def test_reports_bottleneck_flow_not_summed_flow():
    """total_flow is the tranche that traversed the whole chain (min edge),
    not the sum of every hop (which counts the same money once per hop)."""
    g = nx.DiGraph()
    t0 = datetime(2025, 3, 1, 10, 0, 0)
    _add_edge(g, "A", "B", 1_000_000, t0)
    _add_edge(g, "B", "C", 950_000, t0 + timedelta(minutes=30))
    _add_edge(g, "C", "D", 900_000, t0 + timedelta(minutes=60))
    _set_types(g, A="individual", B="business", C="business", D="individual")

    alerts = FraudDetector(g).detect_rapid_layering()
    chain = next(a for a in alerts if tuple(a["entities"]) == ("A", "B", "C", "D"))

    assert chain["total_flow"] == pytest.approx(900_000), \
        f"total_flow should be the bottleneck (900k), got {chain['total_flow']}"
    # The gross sum is still available for transparency but is clearly secondary.
    assert chain["gross_chain_flow"] == pytest.approx(2_850_000)


def test_does_not_emit_subchains_of_a_larger_chain():
    """A single 4-entity route should yield one alert, not also its 3-entity suffix."""
    g = nx.DiGraph()
    t0 = datetime(2025, 3, 1, 10, 0, 0)
    _add_edge(g, "A", "B", 1_000_000, t0)
    _add_edge(g, "B", "C", 950_000, t0 + timedelta(minutes=30))
    _add_edge(g, "C", "D", 900_000, t0 + timedelta(minutes=60))
    _set_types(g, A="individual", B="business", C="business", D="individual")

    alerts = FraudDetector(g).detect_rapid_layering()

    assert len(alerts) == 1, f"expected exactly one maximal chain, got {_chains(alerts)}"
