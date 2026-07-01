import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import identity_generator
import collusion_detector as cd
import json


def test_generate_plants_expected_rings():
    accts = identity_generator.generate_identity_accounts(seed=42)
    assert len(accts) == 150
    kyc_ring = [a for a in accts if a["kyc_doc_hash"] == "PAN-FORGED-7F3A"]
    dev_ring = [a for a in accts if a["device_id"] == "DEV-FARM-01"]
    assert len(kyc_ring) == 8
    assert len(dev_ring) == 12
    assert all(a["is_mule"] for a in kyc_ring + dev_ring)
    x_ring = [a for a in accts if a["device_id"] == "DEV-X-9"]
    assert len(x_ring) == 4
    assert all(a["is_mule"] for a in x_ring)
    # deterministic: same seed -> identical account_ids in order
    again = identity_generator.generate_identity_accounts(seed=42)
    assert [a["account_id"] for a in accts] == [a["account_id"] for a in again]


def test_write_identity_dataset(tmp_path):
    path = tmp_path / "sub" / "identity.json"
    accts = identity_generator.write_identity_dataset(str(path))
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert len(loaded) == len(accts) == 150


def _acc(aid, **kw):
    base = {"account_id": aid, "device_id": None, "ip": None, "kyc_doc_hash": None}
    base.update(kw)
    return base


def test_planted_kyc_ring_detected():
    accts = [_acc(f"A{i}", kyc_doc_hash="PAN-FORGED") for i in range(5)] + \
            [_acc(f"B{i}", kyc_doc_hash=f"PAN-{i}") for i in range(3)]
    rings = cd.detect_collusion_rings(accts, min_ring_size=3)
    assert len(rings) == 1
    assert rings[0]["size"] == 5
    assert any(s["type"] == "kyc_doc_hash" and s["value"] == "PAN-FORGED"
               for s in rings[0]["shared_identifiers"])


def test_transitive_merge_across_identifiers():
    # A&B share a device; B&C share a kyc doc -> one ring {A,B,C}
    accts = [
        _acc("A", device_id="D1"),
        _acc("B", device_id="D1", kyc_doc_hash="K1"),
        _acc("C", kyc_doc_hash="K1"),
    ]
    rings = cd.detect_collusion_rings(accts, min_ring_size=3)
    assert len(rings) == 1
    assert set(rings[0]["account_ids"]) == {"A", "B", "C"}


def test_below_threshold_not_flagged():
    accts = [_acc("A", device_id="D1"), _acc("B", device_id="D1")]
    assert cd.detect_collusion_rings(accts, min_ring_size=3) == []


def test_no_identifier_columns_is_noop():
    accts = [{"account_id": f"A{i}"} for i in range(10)]
    assert cd.detect_collusion_rings(accts, min_ring_size=3) == []
