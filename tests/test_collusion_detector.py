import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import identity_generator


def test_generate_plants_expected_rings():
    accts = identity_generator.generate_identity_accounts(seed=42)
    assert len(accts) == 150
    kyc_ring = [a for a in accts if a["kyc_doc_hash"] == "PAN-FORGED-7F3A"]
    dev_ring = [a for a in accts if a["device_id"] == "DEV-FARM-01"]
    assert len(kyc_ring) == 8
    assert len(dev_ring) == 12
    assert all(a["is_mule"] for a in kyc_ring + dev_ring)
    # deterministic: same seed -> identical account_ids in order
    again = identity_generator.generate_identity_accounts(seed=42)
    assert [a["account_id"] for a in accts] == [a["account_id"] for a in again]
