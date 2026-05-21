"""Case store: state machine + hash-chain integrity."""

import pytest


def _sample_alert(aid="A-1"):
    return {
        "alert_id": aid,
        "pattern_type": "Rapid Layering",
        "severity": "HIGH",
        "total_flow": 1_000_000,
        "entities": ["E1", "E2", "E3"],
    }


def test_open_case_is_idempotent(temp_data_dir):
    from case_manager import CaseStore
    store = CaseStore(temp_data_dir)
    case1 = store.open_case(_sample_alert("A-1"))
    case2 = store.open_case(_sample_alert("A-1"))
    assert case1["alert_id"] == case2["alert_id"]
    # Only one entry in the audit log
    assert len(case2["audit_log"]) == 1


def test_dispose_transitions_status_and_appends_audit(temp_data_dir):
    from case_manager import CaseStore
    store = CaseStore(temp_data_dir)
    store.open_case(_sample_alert("A-1"))
    case = store.dispose("A-1", "INVESTIGATING", note="Looking", author="alice")
    assert case["status"] == "INVESTIGATING"
    assert len(case["audit_log"]) == 2
    assert case["audit_log"][-1]["author"] == "alice"
    assert case["audit_log"][-1]["from_status"] == "OPEN"
    assert case["audit_log"][-1]["to_status"] == "INVESTIGATING"


def test_invalid_status_raises(temp_data_dir):
    from case_manager import CaseStore
    store = CaseStore(temp_data_dir)
    store.open_case(_sample_alert("A-1"))
    with pytest.raises(ValueError):
        store.dispose("A-1", "BOGUS")


def test_dispose_unknown_alert_raises(temp_data_dir):
    from case_manager import CaseStore
    store = CaseStore(temp_data_dir)
    with pytest.raises(ValueError):
        store.dispose("MISSING", "INVESTIGATING")


def test_hash_chain_intact_after_normal_use(temp_data_dir):
    from case_manager import CaseStore
    store = CaseStore(temp_data_dir)
    store.open_case(_sample_alert("A-1"))
    store.dispose("A-1", "INVESTIGATING", note="step 1")
    store.dispose("A-1", "ESCALATED", note="step 2")
    result = store.verify_chain("A-1")
    assert result["verified"] is True
    assert result["entries"] == 3
    assert result["tampered_at"] is None


def test_hash_chain_detects_tampering(temp_data_dir):
    from case_manager import CaseStore
    store = CaseStore(temp_data_dir)
    store.open_case(_sample_alert("A-1"))
    store.dispose("A-1", "INVESTIGATING", note="step 1")
    # Simulate tampering by editing the note field of the first entry
    store._test_tamper("A-1")
    result = store.verify_chain("A-1")
    assert result["verified"] is False
    assert result["tampered_at"] is not None


def test_status_counts_includes_open_for_alerts_without_cases(temp_data_dir):
    from case_manager import CaseStore
    store = CaseStore(temp_data_dir)
    store.open_case(_sample_alert("A-1"))
    counts = store.status_counts([_sample_alert("A-1"), _sample_alert("A-2")])
    assert counts["OPEN"] >= 1  # A-2 has no case row but is OPEN by default
