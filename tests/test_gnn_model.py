"""Structural checks on the GraphSAGE model — receptive-field depth and the
aggregation that resists neighbour-dilution camouflage.

These run in an ISOLATED SUBPROCESS on purpose: on this platform torch and
xgboost cannot coexist in one process (xgboost must be imported first or it
segfaults). The real pipeline always trains XGBoost before importing torch, so
it's unaffected — but a single-process test run would import torch here and then
crash the XGBoost training test. Shelling out keeps torch out of the main
pytest process entirely while still exercising the actual architecture.
"""

import os
import subprocess
import sys
import textwrap

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _run(body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r})
        try:
            import torch  # noqa
            from torch_geometric.nn import SAGEConv  # noqa
        except Exception:
            print("SKIP")
            sys.exit(0)
        from gnn_model import _build_model
    """) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_sage_depth_and_aggregation_are_configurable():
    r = _run("""
        m = _build_model(in_dim=8, hidden=16, num_layers=3, aggr="max")
        assert len(m.convs) == 3, "must have a 3-hop receptive field by default"
        assert all(c.aggr == "max" for c in m.convs), "must use max aggregation"
        print("OK")
    """)
    if "SKIP" in r.stdout:
        pytest.skip("torch_geometric not installed")
    assert "OK" in r.stdout, f"structural check failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def test_sage_forward_produces_one_score_per_edge():
    r = _run("""
        import torch
        m = _build_model(in_dim=8, hidden=16, num_layers=3, aggr="max")
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        out = m(x, edge_index, edge_pairs)
        assert tuple(out.shape) == (2,), f"expected (2,), got {tuple(out.shape)}"
        print("OK")
    """)
    if "SKIP" in r.stdout:
        pytest.skip("torch_geometric not installed")
    assert "OK" in r.stdout, f"forward check failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
