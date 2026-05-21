"""
Case Manager — per-alert investigator workflow.

Each fraud alert can be opened as a "case" with a status (Open / Investigating /
SAR Filed / Dismissed / Escalated) and an audit log of decisions. This is the
piece that turns RUDRA from a dashboard into something a bank compliance officer
would actually use: every state change is logged with author + timestamp +
note, which is exactly what regulators expect to see during an inspection.

Storage: a single JSON file (data/cases.json). For a hackathon POC this beats
spinning up Postgres, and the structure maps cleanly to a real DB schema later.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

# Valid statuses and the transitions we permit. Closed states (SAR_FILED,
# DISMISSED) can still receive notes but no further status change without
# re-opening — kept simple here.
VALID_STATUSES = {"OPEN", "INVESTIGATING", "SAR_FILED", "DISMISSED", "ESCALATED"}


class CaseStore:
    """File-backed case store. Loads everything on init, persists on each write."""

    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "cases.json")
        self.cases: Dict[str, Dict] = {}
        self.load()

    # ── Persistence ─────────────────────────────────────────────
    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.cases = json.load(f)
        else:
            self.cases = {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.cases, f, indent=2)

    # ── Reads ───────────────────────────────────────────────────
    def get(self, alert_id: str) -> Optional[Dict]:
        return self.cases.get(alert_id)

    def list(self, status: Optional[str] = None) -> List[Dict]:
        items = list(self.cases.values())
        if status:
            items = [c for c in items if c.get("status") == status]
        items.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        return items

    def status_counts(self, all_alerts: List[Dict]) -> Dict[str, int]:
        """Return counts for each status, defaulting to OPEN for any alert
        that doesn't have a case row yet."""
        counts = {s: 0 for s in VALID_STATUSES}
        seen = set()
        for c in self.cases.values():
            seen.add(c["alert_id"])
            counts[c.get("status", "OPEN")] = counts.get(c.get("status", "OPEN"), 0) + 1
        # Alerts without an explicit case = OPEN
        for a in all_alerts:
            aid = a.get("alert_id")
            if aid and aid not in seen:
                counts["OPEN"] += 1
        return counts

    # ── Writes ──────────────────────────────────────────────────
    def open_case(self, alert: Dict) -> Dict:
        """Create a case in OPEN status from an alert. Idempotent."""
        aid = alert.get("alert_id")
        if not aid:
            raise ValueError("Alert is missing alert_id")
        if aid in self.cases:
            return self.cases[aid]

        now = datetime.now().isoformat()
        case = {
            "alert_id": aid,
            "pattern_type": alert.get("pattern_type", ""),
            "severity": alert.get("severity", ""),
            "total_flow": alert.get("total_flow", 0),
            "entities": alert.get("entities", []),
            "status": "OPEN",
            "assigned_to": None,
            "created_at": now,
            "updated_at": now,
            "audit_log": [
                {
                    "timestamp": now,
                    "author": "system",
                    "action": "OPENED",
                    "from_status": None,
                    "to_status": "OPEN",
                    "note": "Case auto-opened from triggered alert.",
                }
            ],
        }
        self.cases[aid] = case
        self.save()
        return case

    def dispose(
        self,
        alert_id: str,
        status: str,
        note: str = "",
        author: str = "investigator",
        assigned_to: Optional[str] = None,
    ) -> Dict:
        """Update the case status and append an audit-log entry."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}.")

        case = self.cases.get(alert_id)
        if not case:
            raise ValueError(f"No case exists for alert {alert_id}. Open it first.")

        from_status = case.get("status")
        now = datetime.now().isoformat()
        case["status"] = status
        case["updated_at"] = now
        if assigned_to is not None:
            case["assigned_to"] = assigned_to
        case["audit_log"].append({
            "timestamp": now,
            "author": author,
            "action": f"{from_status}->{status}",
            "from_status": from_status,
            "to_status": status,
            "note": note,
        })
        self.save()
        return case

    def add_note(self, alert_id: str, note: str, author: str = "investigator") -> Dict:
        """Append a note without changing status."""
        case = self.cases.get(alert_id)
        if not case:
            raise ValueError(f"No case exists for alert {alert_id}.")
        now = datetime.now().isoformat()
        case["updated_at"] = now
        case["audit_log"].append({
            "timestamp": now,
            "author": author,
            "action": "NOTE",
            "from_status": case["status"],
            "to_status": case["status"],
            "note": note,
        })
        self.save()
        return case

    def bulk_open_all(self, alerts: List[Dict]) -> int:
        """One-shot helper: open a case for every alert that doesn't have one yet."""
        opened = 0
        for a in alerts:
            aid = a.get("alert_id")
            if aid and aid not in self.cases:
                self.open_case(a)
                opened += 1
        return opened
