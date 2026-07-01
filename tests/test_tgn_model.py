"""Structural checks on the TGN modules — isolated subprocess (torch vs xgboost)."""
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def _run(body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r})
        try:
            import torch  # noqa
            from torch_geometric.nn import TransformerConv  # noqa
        except Exception:
            print("SKIP"); sys.exit(0)
    """) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def _ok(r):
    if "SKIP" in r.stdout:
        pytest.skip("torch/PyG not installed")
    assert r.returncode == 0, r.stderr
    return r


def test_build_tgn_and_decoder_shape():
    r = _ok(_run("""
        import torch
        from tgn_model import build_tgn
        memory, gnn, dec = build_tgn(num_nodes=20, msg_dim=5,
                                     memory_dim=16, time_dim=16, embedding_dim=16)
        z = torch.randn(8, 16)
        out = dec(z[:4], z[4:])
        assert out.shape == (4, 1), out.shape   # one fraud logit per edge
        print("OK")
    """))
    assert "OK" in r.stdout
