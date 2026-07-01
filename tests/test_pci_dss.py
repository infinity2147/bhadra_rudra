import pytest
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from main import app, state
from case_manager import CaseStore

@pytest.fixture
def client(tmp_path):
    # Setup state manually for test using a temp db
    state["cases"] = CaseStore(str(tmp_path))
    
    # Need mock rbac roles if they are enforced in main? 
    # Actually get_role uses a default but we override it via headers.
    
    # Create a dummy alert
    alert = {
        "alert_id": "TEST_ALERT_123",
        "severity": "HIGH",
        "pattern_type": "Smurfing",
        "total_flow": 100000,
        "entities": ["A", "B"]
    }
    state["alerts"] = [alert]
    state["cases"].open_case(alert)
    return TestClient(app)


def test_pci_dss_investigator_blocked(client):
    """PCI DSS Req 7: Access to cardholder data must be restricted to business need-to-know."""
    response = client.post(
        "/api/cases/TEST_ALERT_123/reveal-card",
        headers={"X-User-Role": "INVESTIGATOR", "X-User-Name": "john_inv"}
    )
    assert response.status_code == 403
    assert "PCI DSS Requirement 7" in response.json()["detail"]


def test_pci_dss_supervisor_allowed_and_audited(client):
    """PCI DSS Req 7 & 10: Supervisor can unmask, and access is tracked in audit log."""
    response = client.post(
        "/api/cases/TEST_ALERT_123/reveal-card",
        headers={"X-User-Role": "SUPERVISOR", "X-User-Name": "jane_sup"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["card_number"] == "4532718899012345"

    # Verify audit log was created
    case = state["cases"].get("TEST_ALERT_123")
    assert case is not None
    audit = case["audit_log"]
    reveal_entries = [e for e in audit if e["action"] == "REVEAL_CARD"]
    assert len(reveal_entries) == 1
    
    entry = reveal_entries[0]
    assert entry["author"] == "jane_sup"
    assert "PCI DSS compliance" in entry["note"]
