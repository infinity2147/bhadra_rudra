"""Mocks for AA + DiliSense must return well-formed shapes."""


def test_aa_consent_returns_handle():
    from aa_kyc_mock import aa_create_consent, aa_pull_data, aa_revoke_consent
    c = aa_create_consent("CUST-1", ["FIP-A", "FIP-B"], purpose_code="103", duration_days=7)
    assert c["status"] == "ACTIVE"
    assert "consent_handle" in c
    assert c["_mock_disclaimer"]
    # Pull works
    pull = aa_pull_data(c["consent_handle"], days_back=5)
    assert pull["transaction_count"] > 0
    # Revoke flips status
    rev = aa_revoke_consent(c["consent_handle"])
    assert rev["status"] == "REVOKED"
    # Pull after revoke errors
    pull2 = aa_pull_data(c["consent_handle"])
    assert pull2.get("error")


def test_dilisense_returns_deterministic_score():
    from aa_kyc_mock import dilisense_screen
    r1 = dilisense_screen("Thunder Bolt Exports", "shell_company")
    r2 = dilisense_screen("Thunder Bolt Exports", "shell_company")
    assert r1["risk_score_0_100"] == r2["risk_score_0_100"]
    assert r1["risk"] == "CRITICAL"
    assert any(h["type"] == "SANCTIONS" for h in r1["hits"])


def test_aa_consent_expiry_blocks_pull():
    from aa_kyc_mock import aa_create_consent, aa_pull_data, _AA_CONSENT_STORE
    from datetime import datetime, timedelta

    c = aa_create_consent("CUST-EXP", ["FIP-A"], duration_days=1)
    handle = c["consent_handle"]
    _AA_CONSENT_STORE[handle]["expires_at"] = (datetime.now() - timedelta(days=1)).isoformat()

    pull = aa_pull_data(handle)
    assert pull.get("error")
    assert pull.get("status") == 401
