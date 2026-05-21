"""RBAC matrix tests."""

import pytest


def test_investigator_can_open_and_note():
    from rbac import can
    for action in ["case.open", "case.note", "case.move.INVESTIGATING", "case.move.ESCALATED"]:
        assert can(action, "INVESTIGATOR"), f"INVESTIGATOR should allow {action}"


def test_investigator_cannot_file_sar_or_dismiss():
    from rbac import can
    assert not can("case.move.SAR_FILED", "INVESTIGATOR")
    assert not can("case.move.DISMISSED", "INVESTIGATOR")


def test_supervisor_can_file_sar_and_dismiss():
    from rbac import can
    assert can("case.move.SAR_FILED", "SUPERVISOR")
    assert can("case.move.DISMISSED", "SUPERVISOR")


def test_only_admin_can_edit_config_or_retrain():
    from rbac import can
    assert not can("config.write", "INVESTIGATOR")
    assert not can("config.write", "SUPERVISOR")
    assert can("config.write", "ADMIN")
    assert not can("ml.retrain", "SUPERVISOR")
    assert can("ml.retrain", "ADMIN")


def test_require_raises_for_disallowed():
    from rbac import require
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        require("ml.retrain", "INVESTIGATOR")
    assert exc_info.value.status_code == 403
