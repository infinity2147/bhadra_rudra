"""
Case Manager — per-alert investigator workflow.

Storage moved from JSON-file to SQLite (single file, no external service).
Schema:

  cases
    alert_id TEXT PRIMARY KEY
    pattern_type TEXT
    severity TEXT
    total_flow REAL
    entities_json TEXT        -- JSON array of entity ids
    status TEXT               -- OPEN / INVESTIGATING / SAR_FILED / DISMISSED / ESCALATED
    assigned_to TEXT NULL
    created_at TEXT           -- ISO timestamp
    updated_at TEXT           -- ISO timestamp
    incident_id TEXT NULL     -- set when alert is part of a clustered incident

  audit_log
    id INTEGER PRIMARY KEY AUTOINCREMENT
    alert_id TEXT
    timestamp TEXT
    author TEXT
    action TEXT
    from_status TEXT NULL
    to_status TEXT
    note TEXT
    prev_hash TEXT            -- hash of the previous entry in this case's chain
    this_hash TEXT            -- SHA-256(prev_hash || canonical_entry_json)

Every state change appends to audit_log with a hash that depends on the
previous entry, so any tampering (insertion, edit, deletion) is detectable
by walking the chain. This is the tamper-evidence property a real bank
needs for FIU / RBI inspection.

To verify a case's chain: walk entries in id order, recompute each this_hash
from prev_hash + canonical_entry_json, compare. Mismatch = tampering.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional


VALID_STATUSES = {"OPEN", "INVESTIGATING", "SAR_FILED", "DISMISSED", "ESCALATED"}

# Fields that are part of the canonical-form hash. Stable order matters.
_HASH_FIELDS = ("alert_id", "timestamp", "author", "action", "from_status", "to_status", "note")

GENESIS_HASH = "GENESIS"


def _synchronized(method):
    """Serialize access to the SQLite connection across FastAPI worker threads.

    sqlite3 with check_same_thread=False permits sharing a connection across
    threads but explicitly requires the caller to serialize — two concurrent
    cursors on one connection scramble each other's row descriptions
    (we saw `IndexError: tuple index out of range` on Row column access).
    RLock so internal calls between methods don't self-deadlock.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


def _canonical_entry(row: Dict) -> str:
    """JSON-serialise an audit entry in a deterministic way for hashing."""
    return json.dumps(
        {k: row.get(k, "") for k in _HASH_FIELDS},
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )


def _compute_hash(prev_hash: str, entry: Dict) -> str:
    payload = (prev_hash + "|" + _canonical_entry(entry)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CaseStore:
    """SQLite-backed case store with hash-chain audit log.

    The interface mirrors the previous JSON-file implementation so the rest of
    the codebase doesn't need to change. All methods are safe to call from
    FastAPI workers (sqlite3 has check_same_thread=False here).
    """

    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "rudra.db")
        os.makedirs(data_dir, exist_ok=True)
        # check_same_thread=False because FastAPI handlers may run on different threads;
        # the @_synchronized decorator serializes actual access (see module top).
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()
        self._maybe_migrate_from_json(data_dir)

    # ── Schema ──────────────────────────────────────────────────
    @_synchronized
    def _init_schema(self) -> None:
        c = self.conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS cases (
                alert_id      TEXT PRIMARY KEY,
                pattern_type  TEXT,
                severity      TEXT,
                total_flow    REAL,
                entities_json TEXT,
                status        TEXT,
                assigned_to   TEXT,
                created_at    TEXT,
                updated_at    TEXT,
                incident_id   TEXT
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id     TEXT NOT NULL,
                timestamp    TEXT,
                author       TEXT,
                action       TEXT,
                from_status  TEXT,
                to_status    TEXT,
                note         TEXT,
                prev_hash    TEXT,
                this_hash    TEXT,
                FOREIGN KEY(alert_id) REFERENCES cases(alert_id)
            );
            CREATE INDEX IF NOT EXISTS idx_audit_alert ON audit_log(alert_id, id);
            CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
        """)
        self.conn.commit()

    @_synchronized
    def _maybe_migrate_from_json(self, data_dir: str) -> None:
        """One-time migration: import old cases.json if present and table is empty."""
        json_path = os.path.join(data_dir, "cases.json")
        if not os.path.exists(json_path):
            return
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM cases")
        if c.fetchone()[0] > 0:
            return
        try:
            with open(json_path) as f:
                cases = json.load(f)
            for case in cases.values() if isinstance(cases, dict) else cases:
                self._upsert_case(case)
                for entry in case.get("audit_log", []):
                    self._raw_insert_audit(entry, case["alert_id"])
            self.conn.commit()
            # Rename old file so we know it's been migrated
            os.rename(json_path, json_path + ".migrated")
        except Exception:
            # If migration fails, don't crash — just leave both stores in place
            self.conn.rollback()

    # ── Internal helpers ────────────────────────────────────────
    @_synchronized
    def _upsert_case(self, case: Dict) -> None:
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO cases (alert_id, pattern_type, severity, total_flow,
                               entities_json, status, assigned_to, created_at,
                               updated_at, incident_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alert_id) DO UPDATE SET
                pattern_type = excluded.pattern_type,
                severity = excluded.severity,
                total_flow = excluded.total_flow,
                entities_json = excluded.entities_json,
                status = excluded.status,
                assigned_to = excluded.assigned_to,
                updated_at = excluded.updated_at,
                incident_id = excluded.incident_id
        """, (
            case.get("alert_id"),
            case.get("pattern_type", ""),
            case.get("severity", ""),
            case.get("total_flow", 0),
            json.dumps(case.get("entities", [])),
            case.get("status", "OPEN"),
            case.get("assigned_to"),
            case.get("created_at"),
            case.get("updated_at"),
            case.get("incident_id"),
        ))

    @_synchronized
    def _raw_insert_audit(self, entry: Dict, alert_id: str) -> str:
        """Insert an audit entry computing its hash from the case's previous one."""
        c = self.conn.cursor()
        c.execute("SELECT this_hash FROM audit_log WHERE alert_id = ? ORDER BY id DESC LIMIT 1",
                  (alert_id,))
        row = c.fetchone()
        prev_hash = row["this_hash"] if row else GENESIS_HASH

        entry_full = {
            "alert_id": alert_id,
            "timestamp": entry.get("timestamp"),
            "author": entry.get("author"),
            "action": entry.get("action"),
            "from_status": entry.get("from_status"),
            "to_status": entry.get("to_status"),
            "note": entry.get("note", ""),
        }
        this_hash = _compute_hash(prev_hash, entry_full)

        c.execute("""
            INSERT INTO audit_log (alert_id, timestamp, author, action,
                                    from_status, to_status, note, prev_hash, this_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert_id,
            entry_full["timestamp"],
            entry_full["author"],
            entry_full["action"],
            entry_full["from_status"],
            entry_full["to_status"],
            entry_full["note"],
            prev_hash,
            this_hash,
        ))
        return this_hash

    @_synchronized
    def _audit_rows(self, alert_id: str) -> List[Dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM audit_log WHERE alert_id = ? ORDER BY id", (alert_id,))
        return [dict(r) for r in c.fetchall()]

    def _row_to_case(self, row: sqlite3.Row) -> Dict:
        return {
            "alert_id": row["alert_id"],
            "pattern_type": row["pattern_type"],
            "severity": row["severity"],
            "total_flow": row["total_flow"],
            "entities": json.loads(row["entities_json"] or "[]"),
            "status": row["status"],
            "assigned_to": row["assigned_to"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "incident_id": row["incident_id"],
            "audit_log": self._audit_rows(row["alert_id"]),
        }

    # ── Public API ──────────────────────────────────────────────
    @_synchronized
    def get(self, alert_id: str) -> Optional[Dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM cases WHERE alert_id = ?", (alert_id,))
        row = c.fetchone()
        return self._row_to_case(row) if row else None

    @_synchronized
    def list(self, status: Optional[str] = None) -> List[Dict]:
        c = self.conn.cursor()
        if status:
            c.execute("SELECT * FROM cases WHERE status = ? ORDER BY updated_at DESC", (status,))
        else:
            c.execute("SELECT * FROM cases ORDER BY updated_at DESC")
        return [self._row_to_case(r) for r in c.fetchall()]

    @_synchronized
    def status_counts(self, all_alerts: List[Dict]) -> Dict[str, int]:
        c = self.conn.cursor()
        c.execute("SELECT status, COUNT(*) as cnt FROM cases GROUP BY status")
        counts = {s: 0 for s in VALID_STATUSES}
        seen = set()
        for r in c.fetchall():
            counts[r["status"]] = r["cnt"]
        # Any alert without a case row counts as OPEN
        c.execute("SELECT alert_id FROM cases")
        for r in c.fetchall():
            seen.add(r["alert_id"])
        for a in all_alerts:
            aid = a.get("alert_id")
            if aid and aid not in seen:
                counts["OPEN"] += 1
        return counts

    @_synchronized
    def open_case(self, alert: Dict) -> Dict:
        aid = alert.get("alert_id")
        if not aid:
            raise ValueError("Alert is missing alert_id")
        existing = self.get(aid)
        if existing:
            return existing
        now = datetime.now().isoformat()
        case_row = {
            "alert_id": aid,
            "pattern_type": alert.get("pattern_type", ""),
            "severity": alert.get("severity", ""),
            "total_flow": alert.get("total_flow", 0),
            "entities": alert.get("entities", []),
            "status": "OPEN",
            "assigned_to": None,
            "created_at": now,
            "updated_at": now,
            "incident_id": None,
        }
        self._upsert_case(case_row)
        self._raw_insert_audit({
            "timestamp": now,
            "author": "system",
            "action": "OPENED",
            "from_status": None,
            "to_status": "OPEN",
            "note": "Case auto-opened from triggered alert.",
        }, aid)
        self.conn.commit()
        return self.get(aid)

    @_synchronized
    def dispose(
        self,
        alert_id: str,
        status: str,
        note: str = "",
        author: str = "investigator",
        assigned_to: Optional[str] = None,
    ) -> Dict:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}.")
        case = self.get(alert_id)
        if not case:
            raise ValueError(f"No case exists for alert {alert_id}. Open it first.")

        from_status = case["status"]
        now = datetime.now().isoformat()

        c = self.conn.cursor()
        c.execute("""
            UPDATE cases SET status = ?, updated_at = ?, assigned_to = COALESCE(?, assigned_to)
            WHERE alert_id = ?
        """, (status, now, assigned_to, alert_id))
        self._raw_insert_audit({
            "timestamp": now,
            "author": author,
            "action": f"{from_status}->{status}",
            "from_status": from_status,
            "to_status": status,
            "note": note,
        }, alert_id)
        self.conn.commit()
        return self.get(alert_id)

    @_synchronized
    def add_note(self, alert_id: str, note: str, author: str = "investigator") -> Dict:
        case = self.get(alert_id)
        if not case:
            raise ValueError(f"No case exists for alert {alert_id}.")
        now = datetime.now().isoformat()
        c = self.conn.cursor()
        c.execute("UPDATE cases SET updated_at = ? WHERE alert_id = ?", (now, alert_id))
        self._raw_insert_audit({
            "timestamp": now,
            "author": author,
            "action": "NOTE",
            "from_status": case["status"],
            "to_status": case["status"],
            "note": note,
        }, alert_id)
        self.conn.commit()
        return self.get(alert_id)

    @_synchronized
    def set_incident(self, alert_id: str, incident_id: Optional[str]) -> None:
        c = self.conn.cursor()
        c.execute("UPDATE cases SET incident_id = ? WHERE alert_id = ?", (incident_id, alert_id))
        self.conn.commit()

    @_synchronized
    def bulk_open_all(self, alerts: List[Dict]) -> int:
        opened = 0
        for a in alerts:
            aid = a.get("alert_id")
            if not aid:
                continue
            if not self.get(aid):
                self.open_case(a)
                opened += 1
        return opened

    # ── Hash-chain verification ─────────────────────────────────
    @_synchronized
    def verify_chain(self, alert_id: str) -> Dict:
        """Recompute every hash in the chain and report tamper status."""
        rows = self._audit_rows(alert_id)
        if not rows:
            return {"alert_id": alert_id, "verified": True, "entries": 0,
                    "tampered_at": None, "message": "No audit entries yet."}

        prev_hash = GENESIS_HASH
        for r in rows:
            expected = _compute_hash(prev_hash, {
                "alert_id": r["alert_id"],
                "timestamp": r["timestamp"],
                "author": r["author"],
                "action": r["action"],
                "from_status": r["from_status"],
                "to_status": r["to_status"],
                "note": r["note"] or "",
            })
            if r["prev_hash"] != prev_hash:
                return {"alert_id": alert_id, "verified": False, "entries": len(rows),
                        "tampered_at": r["id"],
                        "message": f"prev_hash mismatch at entry {r['id']}"}
            if r["this_hash"] != expected:
                return {"alert_id": alert_id, "verified": False, "entries": len(rows),
                        "tampered_at": r["id"],
                        "message": f"this_hash mismatch at entry {r['id']}"}
            prev_hash = r["this_hash"]

        return {"alert_id": alert_id, "verified": True, "entries": len(rows),
                "tampered_at": None, "message": "Chain intact.",
                "head_hash": prev_hash}

    # ── Test helper: simulate tampering (only used by tests) ────
    @_synchronized
    def _test_tamper(self, alert_id: str) -> None:  # pragma: no cover
        c = self.conn.cursor()
        c.execute("UPDATE audit_log SET note = 'TAMPERED' WHERE alert_id = ? AND id = "
                  "(SELECT MIN(id) FROM audit_log WHERE alert_id = ?)",
                  (alert_id, alert_id))
        self.conn.commit()
