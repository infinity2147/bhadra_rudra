"""Chronological-split correctness for the TGN temporal loader.

Runs in an ISOLATED SUBPROCESS: torch/PyG cannot share a process with xgboost
on this platform (xgboost must import first or it segfaults), so — like
tests/test_gnn_model.py — we shell out to keep torch out of the main pytest run.
"""
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
            from torch_geometric.data import TemporalData  # noqa
        except Exception:
            print("SKIP"); sys.exit(0)
        import pandas as pd, tempfile, os
        from temporal_data_loader import load_temporal_data
        # tiny deterministic CSV, timestamps intentionally OUT of order
        rows = []
        ts = ["2022-01-03 00:00:00", "2022-01-01 00:00:00", "2022-01-02 00:00:00"]
        for i in range(30):
            rows.append(dict(
                transaction_id=f"T{{i}}", timestamp=ts[i % 3],
                sender_id=f"A{{i%5}}", receiver_id=f"B{{i%7}}",
                amount=100.0 + i, transaction_type="NEFT",
                is_fraud=int(i % 4 == 0),
            ))
        df = pd.DataFrame(rows)
        p = os.path.join(tempfile.mkdtemp(), "transactions.csv")
        df.to_csv(p, index=False)
        b = load_temporal_data(p)
    """) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def _ok(r):
    if "SKIP" in r.stdout:
        pytest.skip("torch/PyG not installed")
    assert r.returncode == 0, r.stderr
    return r


def test_split_is_chronological_and_shaped():
    r = _ok(_run("""
        d, tr, va, te = b["data"], b["train"], b["val"], b["test"]
        # strictly sorted event times
        assert bool((d.t[1:] >= d.t[:-1]).all()), "events not time-sorted"
        # chronological split, no leakage
        assert int(tr.t.max()) <= int(va.t.min()) <= int(te.t.min())
        # labels are 0/1 floats; msg present
        assert set(d.y.unique().tolist()) <= {0.0, 1.0}
        assert b["msg_dim"] == d.msg.size(-1) and b["msg_dim"] >= 2
        assert b["num_nodes"] >= 1
        print("OK")
    """))
    assert "OK" in r.stdout
