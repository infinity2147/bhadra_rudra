"""Persistent taint memory: when an entity is confirmed fraudulent, decaying
taint propagates to graph neighbours, PERSISTS across runs, and FLOORS future
risk scores. (The 'compounding' idea PRISM described but never built.)"""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from taint_store import TaintStore  # noqa: E402


def _path_graph():
    g = nx.DiGraph()
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "D")
    return g


def test_taint_decays_by_hop(tmp_path):
    ts = TaintStore(str(tmp_path / "t.db"))
    ts.seed(_path_graph(), ["A"], source="CASE_1", decay=0.5, max_hops=3)
    assert ts.get("A") == 1.0
    assert ts.get("B") == 0.5
    assert ts.get("C") == 0.25
    assert abs(ts.get("D") - 0.125) < 1e-9


def test_taint_persists_across_instances(tmp_path):
    db = str(tmp_path / "t.db")
    TaintStore(db).seed(_path_graph(), ["A"], source="CASE_1", decay=0.5, max_hops=2)
    # brand-new instance on the same DB file (simulates a process restart)
    reopened = TaintStore(db)
    assert reopened.get("B") == 0.5
    assert reopened.get("A") == 1.0


def test_apply_floor_raises_clean_score(tmp_path):
    ts = TaintStore(str(tmp_path / "t.db"))
    ts.seed(_path_graph(), ["A"], source="CASE_1", decay=0.5, max_hops=2)
    floored = ts.apply_floor({"A": 0.1, "B": 0.0, "C": 0.9})
    assert floored["A"] == 1.0          # tainted seed floors a low score
    assert floored["B"] == 0.5
    assert floored["C"] == 0.9          # genuine high score preserved (max)


def test_reseeding_accumulates_max(tmp_path):
    ts = TaintStore(str(tmp_path / "t.db"))
    g = _path_graph()
    ts.seed(g, ["A"], source="CASE_1", decay=0.5, max_hops=3)   # D -> 0.125
    ts.seed(g, ["D"], source="CASE_2", decay=0.5, max_hops=3)   # D -> 1.0 now
    assert ts.get("D") == 1.0           # later, stronger taint wins
    assert ts.get("A") == 1.0           # earlier taint not lost
