"""
Lightweight role-based access control for the demo.

Three roles model what a real PSB fraud-ops team looks like:

  - INVESTIGATOR — can triage cases, add notes, move OPEN/INVESTIGATING/ESCALATED.
                   Cannot file SAR (which leaves the bank) or change config.
  - SUPERVISOR   — everything INVESTIGATOR can do, plus FILE_SAR and DISMISS.
                   The actual "money out the door" steps require a senior eye.
  - ADMIN        — everything, plus threshold edits + ML retraining.

This is a demo gate, not a real auth system. We use a header (X-User-Role)
instead of a login screen so evaluators don't have to manage credentials —
the role can be flipped from the UI dropdown. A production deployment
would replace this with the bank's existing IDP (LDAP / Okta / Keycloak).
"""

from __future__ import annotations

from typing import Set, Optional

from fastapi import HTTPException, Header


VALID_ROLES = {"INVESTIGATOR", "SUPERVISOR", "ADMIN"}
DEFAULT_ROLE = "INVESTIGATOR"

# What each action requires
PERMISSIONS = {
    "case.open": {"INVESTIGATOR", "SUPERVISOR", "ADMIN"},
    "case.note": {"INVESTIGATOR", "SUPERVISOR", "ADMIN"},
    "case.move.INVESTIGATING": {"INVESTIGATOR", "SUPERVISOR", "ADMIN"},
    "case.move.ESCALATED": {"INVESTIGATOR", "SUPERVISOR", "ADMIN"},
    "case.move.SAR_FILED": {"SUPERVISOR", "ADMIN"},
    "case.move.DISMISSED": {"SUPERVISOR", "ADMIN"},
    "fiu.download": {"INVESTIGATOR", "SUPERVISOR", "ADMIN"},
    "config.read": {"INVESTIGATOR", "SUPERVISOR", "ADMIN"},
    "config.write": {"ADMIN"},
    "ml.retrain": {"ADMIN"},
    "pipeline.run": {"ADMIN"},
    # Driving the live monitoring stream (start/stop the local consumer, replay
    # already-loaded transactions onto the bus) is an operational *view* action,
    # not a money-movement or config change — every role that can triage cases
    # can also watch the live feed. Distinct from pipeline.run (full regen, ADMIN).
    "stream.control": {"INVESTIGATOR", "SUPERVISOR", "ADMIN"},
    "audit.verify": {"SUPERVISOR", "ADMIN"},
}


def get_role(x_user_role: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency: pull the role from the X-User-Role header.

    No header → DEFAULT_ROLE (demo convenience). But a header that's PRESENT and
    not a recognised role is rejected with 403 rather than silently elevated to
    the default — an unknown role must never inherit privileges.
    """
    if x_user_role is None or x_user_role.strip() == "":
        return DEFAULT_ROLE
    role = x_user_role.upper()
    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=403,
            detail=f"Unknown role '{x_user_role}'. Valid roles: {sorted(VALID_ROLES)}",
        )
    return role


def require(action: str, role: str) -> None:
    """Raise 403 if the role can't perform the action."""
    allowed = PERMISSIONS.get(action)
    if allowed is None:
        raise HTTPException(500, f"Unknown action {action}")
    if role not in allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role}' cannot perform '{action}'. Required role: any of {sorted(allowed)}",
        )


def can(action: str, role: str) -> bool:
    """Boolean variant for the UI."""
    return role in PERMISSIONS.get(action, set())


def role_capabilities(role: str) -> dict:
    """Return a flat dict of what this role can do, for the frontend nav gating."""
    return {action: (role in allowed) for action, allowed in PERMISSIONS.items()}
